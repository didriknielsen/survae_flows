import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.transforms.bijections import Bijection


class Linear(Bijection):
    """
    Linear bijection y=Wx.

    Costs:
        forward = O(BD^2)
        inverse = O(BD^2 + D^3)
        ldj = O(D^3)
    where:
        B = batch size
        D = number of features

    Args:
        num_features: int, Number of features in the input and output.
        orthogonal_init: bool, if True initialize weights to be a random orthogonal matrix (default=True).
        bias: bool, if True a bias is included (default=False).
    """
    def __init__(self, num_features, orthogonal_init=True, bias=False):
        super(Linear, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features, num_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(orthogonal_init)

    def reset_parameters(self, orthogonal_init):
        self.orthogonal_init = orthogonal_init

        if self.orthogonal_init:
            nn.init.orthogonal_(self.weight)
        else:
            bound = 1.0 / np.sqrt(self.num_features)
            nn.init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        z = F.linear(x, self.weight, self.bias)
        _, ldj = torch.slogdet(self.weight)
        ldj = ldj.expand([x.shape[0]])
        return z, ldj

    def inverse(self, z):
        weight_inv = torch.inverse(self.weight)
        if self.bias is not None: z = z - self.bias
        x = F.linear(z, weight_inv)
        return x
