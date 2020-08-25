import torch
import torch.nn as nn
from survae.nn.layers import gelu, swish, concat_relu, concat_elu, gated_tanh


class GELU(nn.Module):
    '''GELU activation (https://arxiv.org/abs/1606.08415) as used in Sparse Transformers (https://arxiv.org/abs/1904.10509).'''

    def forward(self, input):
        return gelu(input)


class Swish(nn.Module):
    '''Swish activation (https://arxiv.org/abs/1710.05941).'''

    def forward(self, input):
        return swish(input)


class ConcatReLU(nn.Module):
    '''Concatenated ReLU (http://arxiv.org/abs/1603.05201).'''

    def forward(self, input):
        return concat_relu(input)


class ConcatELU(nn.Module):
    '''Like concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU instead.'''

    def forward(self, input):
        return concat_elu(input)


class GatedTanhUnit(nn.Module):
    '''Gated Tanh activation.'''

    def __init__(self, dim=-1):
        super(GatedTanhUnit, self).__init__()
        self.dim = dim

    def forward(self, x):
        return gated_tanh(x, dim=self.dim)
