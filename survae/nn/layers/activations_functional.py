import torch
import torch.nn.functional as F


def gelu(x):
    '''GELU activation (https://arxiv.org/abs/1606.08415) as used in Sparse Transformers (https://arxiv.org/abs/1904.10509).'''
    return x * torch.sigmoid(1.702 * x)


def swish(x):
    '''Swish activation (https://arxiv.org/abs/1710.05941).'''
    return x * torch.sigmoid(x)


def concat_relu(x):
    '''Concatenated ReLU (http://arxiv.org/abs/1603.05201).'''
    return F.relu(torch.cat([x, -x], dim=1))


def concat_elu(x):
    '''Like concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU instead.'''
    return F.elu(torch.cat([x, -x], dim=1))


def gated_tanh(x, dim):
    '''Gated Tanh activation.'''
    x_tanh, x_sigmoid = torch.chunk(x, 2, dim=dim)
    return torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)
