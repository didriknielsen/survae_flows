import math
import copy
import torch
import torch.nn as nn
from survae.nn.layers import LambdaLayer
from survae.nn.blocks import DenseTransformerBlock

class PositionalEncoding1d(nn.Module):
    '''
    Learning positional embeddings.

    Args:
        shape: Iterable, the shape of the input.
        embedding_dim: int, the size of each embedding vector.
    '''

    def __init__(self, size, embedding_dim):
        super(PositionalEncoding1d, self).__init__()
        self.size = size
        self.embedding_dim = embedding_dim
        self.encode_l = nn.Parameter(torch.Tensor(size, 1, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize position embedding layers to N(0, 0.125/sqrt(3*d))
        # as described in paragraph 3 of Section "6. Training":
        nn.init.normal_(self.encode_l, std=0.125/math.sqrt(self.embedding_dim))

    def forward(self, x):
        return x + self.encode_l


class PositionalDenseTransformer(nn.Module):
    def __init__(self, l_input=50, d_input=1, d_output=2, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=512, dropout=0.1,
                 activation="gelu", kdim=None, vdim=None,
                 attn_bias=True, checkpoint_blocks=False,
                 in_lambda=lambda x: x,
                 out_lambda=lambda x: x):
        super(PositionalDenseTransformer, self).__init__()

        decoder_layer = DenseTransformerBlock(d_model=d_model,
                                              nhead=nhead,
                                              dim_feedforward=dim_feedforward,
                                              dropout=dropout,
                                              activation=activation,
                                              kdim=kdim,
                                              vdim=vdim,
                                              attn_bias=attn_bias,
                                              checkpoint=checkpoint_blocks)


        self.in_lambda = LambdaLayer(in_lambda)
        self.in_linear = nn.Linear(d_input, d_model)
        self.encode = PositionalEncoding1d(l_input, d_model)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_linear = nn.Linear(d_model, d_output)
        self.out_lambda = LambdaLayer(out_lambda)

        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead

        self._reset_parameters()

    def forward(self, x):

        # Initial ops
        x = self.in_lambda(x)
        x = x.permute(2,0,1)

        # Input layer
        x = self.in_linear(x)
        x = self.encode(x)

        # Main layers
        for decoder_layer in self.layers:
            x = decoder_layer(x, attn_mask=None, key_padding_mask=None)

        # Output layer
        x = self.out_norm(x)
        x = self.out_linear(x)

        # Final ops
        x = x.permute(1,2,0)
        x = self.out_lambda(x)
        return x

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

        # Initialize output weight matrix to 0:
        nn.init.zeros_(self.out_linear.weight)
        if self.out_linear.bias is not None:
            nn.init.zeros_(self.out_linear.bias)


class DenseTransformer(nn.Module):
    def __init__(self, d_input=1, d_output=2, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=512, dropout=0.1,
                 activation="gelu", kdim=None, vdim=None,
                 attn_bias=True, checkpoint_blocks=False,
                 in_lambda=lambda x: x,
                 out_lambda=lambda x: x):
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


        self.in_lambda = LambdaLayer(in_lambda)
        self.in_linear = nn.Linear(d_input, d_model)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_linear = nn.Linear(d_model, d_output)
        self.out_lambda = LambdaLayer(out_lambda)

        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead

        self._reset_parameters()

    def forward(self, x):

        # Initial ops
        x = self.in_lambda(x)
        x = x.permute(2,0,1)

        # Input layer
        x = self.in_linear(x)

        # Main layers
        for decoder_layer in self.layers:
            x = decoder_layer(x, attn_mask=None, key_padding_mask=None)

        # Output layer
        x = self.out_norm(x)
        x = self.out_linear(x)

        # Final ops
        x = x.permute(1,2,0)
        x = self.out_lambda(x)
        return x

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

        # Initialize output weight matrix to 0:
        nn.init.zeros_(self.out_linear.weight)
        if self.out_linear.bias is not None:
            nn.init.zeros_(self.out_linear.bias)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
