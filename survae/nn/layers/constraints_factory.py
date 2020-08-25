import torch
import torch.nn.functional as F


def scale_fn(scale_str):
    assert scale_str in {'exp', 'softplus', 'sigmoid', 'tanh_exp'}
    if scale_str == 'exp':          return lambda s: torch.exp(s)
    elif scale_str == 'softplus':   return lambda s: F.softplus(s)
    elif scale_str == 'sigmoid':    return lambda s: torch.sigmoid(s + 2.) + 1e-3
    elif scale_str == 'tanh_exp':   return lambda s: torch.exp(2.*torch.tanh(s/2.))
