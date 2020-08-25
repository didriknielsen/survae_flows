import torch
from collections.abc import Iterable
from survae.transforms.bijections import Bijection


class PermuteAxes(Bijection):

    def __init__(self, permutation):
        super(PermuteAxes, self).__init__()
        assert isinstance(permutation, Iterable), 'permutation must be an Iterable'
        assert permutation[0] == 0, 'First element of permutation must be 0 (such that batch dimension stays intact)'
        self.permutation = permutation
        self.inverse_permutation = torch.argsort(torch.tensor(self.permutation)).tolist()

    def forward(self, x):
        z = x.permute(self.permutation).contiguous()
        ldj = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        x = z.permute(self.inverse_permutation).contiguous()
        return x
