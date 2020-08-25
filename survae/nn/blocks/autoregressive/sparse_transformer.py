import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint
from survae.nn.layers import act_module


# Adapted from the official PyTorch implementation:
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py

class DenseTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=512, dropout=0.1,
                 activation="gelu", kdim=None, vdim=None,
                 attn_bias=True, checkpoint_blocks=False):
        super(DenseTransformer, self).__init__()

        decoder_layer = DenseTransformerBlock(d_model=d_model,
                                              nhead=nhead,
                                              dim_feedforward=dim_feedforward,
                                              dropout=dropout,
                                              activation=activation,
                                              kdim=kdim,
                                              vdim=vdim,
                                              attn_bias=attn_bias,
                                              checkpoint=checkpoint_blocks)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.out_norm = nn.LayerNorm(d_model)

        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead

        self._reset_parameters()

    def forward(self, x, key_padding_mask=None):
        if x.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        attn_mask = self.generate_square_subsequent_mask(x.shape[0]).to(x.device)
        for decoder_layer in self.layers:
            x = decoder_layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return self.out_norm(x)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        # The initialization is done as described in:
        # - Paragraph 3 of Section "6. Training".
        # - Final paragraph of Section "5.2. Scaling to hundreds of layers".

        # Initialization has already been done in each block.
        # Here, that initialization is adjusted based on num_layers.

        # Adjust initialization based on num_layers as descibed in Section 5.2:
        for decoder_layer in self.layers:
            decoder_layer.linear2.weight.data /= math.sqrt(2*self.num_layers)
            decoder_layer.self_attn.out_proj.weight.data /= math.sqrt(2*self.num_layers)


class DenseTransformerBlock(nn.Module):
    '''
    The residual block from Sparse Transformers (https://arxiv.org/abs/1904.10509) using dense self-attention.
    Differences from the regular decoder-only Transformer layer include:
    - No dropout in the attention module or in feed-forward network.
    - ReLU activation replaced by GELU.
    - Reordering of the operations.
    '''

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation="gelu", kdim=None, vdim=None, attn_bias=True, checkpoint=False):
        super(DenseTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=kdim, vdim=vdim, bias=attn_bias)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = act_module(activation)
        self.checkpoint = checkpoint

        self._reset_parameters()

    def _reset_parameters(self):
        # The initialization is done as described in paragraph 3 of Section "6. Training".

        # Initialize feedforward weights to N(0, 0.125/sqrt(d_in)):
        nn.init.normal_(self.linear1.weight, std=0.125/math.sqrt(self.linear1.weight.shape[1]))
        nn.init.normal_(self.linear2.weight, std=0.125/math.sqrt(self.linear2.weight.shape[1]))
        # Initialize feedforward biases to 0:
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

        # Initialize Q, K, V transform weights to N(0, 0.125/sqrt(d_in)):
        nn.init.normal_(self.self_attn.in_proj_weight, std=0.125/math.sqrt(self.self_attn.in_proj_weight.shape[1]))
        if not self.self_attn._qkv_same_embed_dim:
            nn.init.normal_(self.self_attn.q_proj_weight, std=0.125/math.sqrt(self.self_attn.q_proj_weight.shape[1]))
            nn.init.normal_(self.self_attn.k_proj_weight, std=0.125/math.sqrt(self.self_attn.k_proj_weight.shape[1]))
            nn.init.normal_(self.self_attn.v_proj_weight, std=0.125/math.sqrt(self.self_attn.v_proj_weight.shape[1]))
        # Initialize Q, K, V transform biases to 0:
        if self.self_attn.in_proj_bias is not None:
            nn.init.zeros_(self.self_attn.in_proj_bias)

        # Initialize output transform weights to N(0, 0.125/sqrt(d_in)):
        nn.init.normal_(self.self_attn.out_proj.weight, std=0.125/math.sqrt(self.self_attn.out_proj.weight.shape[1]))
        # Initialize output transform biases to 0:
        if self.self_attn.out_proj.bias is not None:
            nn.init.zeros_(self.self_attn.out_proj.bias)

    def _attn_block(self, x, attn_mask=None, key_padding_mask=None):
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        x = self.dropout1(x)
        return x

    def _ff_block(self, x):
        x = self.norm2(x)
        x = self.linear2(self.activation(self.linear1(x)))
        x = self.dropout2(x)
        return x

    def _forward(self, x, attn_mask=None, key_padding_mask=None):
        ax = self._attn_block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        bx = self._ff_block(x+ax)
        return x + ax + bx

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if not self.checkpoint:
            return self._forward(x, attn_mask, key_padding_mask)
        else:
            x.requires_grad_(True)
            return checkpoint.checkpoint(self._forward, x, attn_mask, key_padding_mask)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
