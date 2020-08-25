import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint
from survae.nn.layers import act_module


# Adapted from the official PyTorch implementation:
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", kdim=None, vdim=None,
                 attn_bias=True, checkpoint_blocks=False):
        super(DecoderOnlyTransformer, self).__init__()

        decoder_layer = DecoderOnlyTransformerBlock(d_model=d_model,
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

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

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
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class DecoderOnlyTransformerBlock(nn.Module):
    '''The residual block from Transformers (https://arxiv.org/abs/1706.03762) in self-attention, decoder-only mode.'''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", kdim=None, vdim=None, attn_bias=True, checkpoint=False):
        super(DecoderOnlyTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=kdim, vdim=vdim, bias=attn_bias)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = act_module(activation)
        self.checkpoint = checkpoint

    def _attn_block(self, x, attn_mask=None, key_padding_mask=None):
        x2 = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        return x

    def _ff_block(self, x):
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

    def _forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self._attn_block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self._ff_block(x)
        return x

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if not self.checkpoint:
            return self._forward(x, attn_mask, key_padding_mask)
        else:
            x.requires_grad_(True)
            return checkpoint.checkpoint(self._forward, x, attn_mask, key_padding_mask)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
