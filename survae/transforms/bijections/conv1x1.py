import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.transforms.bijections import Bijection


class Conv1x1(Bijection):
    """
    Invertible 1x1 Convolution [1].
    The weight matrix is initialized as a random rotation matrix
    as described in Section 3.2 of [1].

    Args:
        num_channels (int): Number of channels in the input and output.
        orthogonal_init (bool): If True, initialize weights to be a random orthogonal matrix (default=True).
        slogdet_cpu (bool): If True, compute slogdet on cpu (default=True).

    Note:
        torch.slogdet appears to run faster on CPU than on GPU.
        slogdet_cpu is thus set to True by default.

    References:
        [1] Glow: Generative Flow with Invertible 1×1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
    """
    def __init__(self, num_channels, orthogonal_init=True, slogdet_cpu=True):
        super(Conv1x1, self).__init__()
        self.num_channels = num_channels
        self.slogdet_cpu = slogdet_cpu
        self.weight = nn.Parameter(torch.Tensor(num_channels, num_channels))
        self.reset_parameters(orthogonal_init)

    def reset_parameters(self, orthogonal_init):
        self.orthogonal_init = orthogonal_init

        if self.orthogonal_init:
            nn.init.orthogonal_(self.weight)
        else:
            bound = 1.0 / np.sqrt(self.num_channels)
            nn.init.uniform_(self.weight, -bound, bound)

    def _conv(self, weight, v):
        return F.conv2d(v, weight.unsqueeze(-1).unsqueeze(-1))

    def _logdet(self, x_shape):
        b, c, h, w = x_shape
        if self.slogdet_cpu:
            _, ldj_per_pixel = torch.slogdet(self.weight.to('cpu'))
        else:
            _, ldj_per_pixel = torch.slogdet(self.weight)
        ldj = ldj_per_pixel * h * w
        return ldj.expand([b]).to(self.weight.device)

    def forward(self, x):
        z = self._conv(self.weight, x)
        ldj = self._logdet(x.shape)
        return z, ldj

    def inverse(self, z):
        weight_inv = torch.inverse(self.weight)
        x = self._conv(weight_inv, z)
        return x
