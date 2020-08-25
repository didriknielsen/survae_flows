import torch
from survae.transforms.stochastic import StochasticTransform


class StochasticPermutation(StochasticTransform):
    '''A stochastic permutation layer.'''

    def __init__(self, dim=1):
        super(StochasticPermutation, self).__init__()
        self.register_buffer('buffer', torch.zeros(1))
        self.dim = dim

    def forward(self, x):
        rand = torch.rand(x.shape[0], x.shape[self.dim], device=x.device)
        permutation = rand.argsort(dim=1)
        for d in range(1, self.dim):
            permutation = permutation.unsqueeze(1)
        for d in range(self.dim+1, x.dim()):
            permutation = permutation.unsqueeze(-1)
        permutation = permutation.expand_as(x)
        z = torch.gather(x, self.dim, permutation)
        ldj = self.buffer.new_zeros(x.shape[0])
        return z, ldj

    def inverse(self, z):
        rand = torch.rand(z.shape[0], z.shape[self.dim], device=z.device)
        permutation = rand.argsort(dim=1)
        for d in range(1, self.dim):
            permutation = permutation.unsqueeze(1)
        for d in range(self.dim+1, z.dim()):
            permutation = permutation.unsqueeze(-1)
        permutation = permutation.expand_as(z)
        x = torch.gather(z, self.dim, permutation)
        return x
