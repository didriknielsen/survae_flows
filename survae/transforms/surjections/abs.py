import math
import torch
import torch.nn.functional as F
from survae.transforms.surjections import Surjection
from survae.utils import sum_except_batch


class SimpleAbsSurjection(Surjection):
    '''
    An absolute value layer.
    Uses a fixed inverse which flips the sign with probability 0.5.
    This enforces symmetry across all axes.
    '''

    stochastic_forward = False

    def forward(self, x):
        z = x.abs()
        ldj = - x.new_ones(x.shape[0]) * math.log(2) * x.shape[1:].numel()
        return z, ldj

    def inverse(self, z):
        s = torch.bernoulli(0.5*torch.ones_like(z))
        x = (2*s-1)*z
        return x
