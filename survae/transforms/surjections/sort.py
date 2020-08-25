import torch
from survae.transforms.surjections import Surjection


class SimpleSortSurjection(Surjection):
    '''
    A sorting layer. Sorts along `dim` for element extracted using `lambd`.
    The inverse is a random permutation.

    Args:
        dim: int, the dimension along which the tensor is sorted.
        lambd: callable, a callable which extracts a subset of x which is used to determine the sorting order.

    Example for (1,4) tensor using (dim=1, lambd=lambda x: x):
    # Input x, output z
    tensor([[0.6268, 0.0913, 0.8587, 0.2548]])
    tensor([[0.0913, 0.2548, 0.6268, 0.8587]])

    Example for (1,4,2) tensor using (dim=1, lambd=lambda x: x[:,:,0]):
    # Input x, output z
    tensor([[[0.6601, 0.0948],
             [0.9293, 0.1715],
             [0.5511, 0.7153],
             [0.3567, 0.7232]]])
    tensor([[[0.3567, 0.7232],
             [0.5511, 0.7153],
             [0.6601, 0.0948],
             [0.9293, 0.1715]]])

    '''
    stochastic_forward = False

    def __init__(self, dim=1, lambd=lambda x: x):
        super(SimpleSortSurjection, self).__init__()
        self.register_buffer('buffer', torch.zeros(1))
        self.dim = dim
        self.lambd = lambd

    def forward(self, x):
        x_order = self.lambd(x)
        assert x_order.dim() == 2, 'lambd should return a tensor of shape (batch_size, dim_size) = ({}, {}), not {}'.format(x.shape[0], x.shape[self.dim], x_order.shape)
        assert x_order.shape[0] == x.shape[0], 'lambd should return a tensor of shape (batch_size, dim_size) = ({}, {}), not {}'.format(x.shape[0], x.shape[self.dim], x_order.shape)
        assert x_order.shape[1] == x.shape[self.dim], 'lambd should return a tensor of shape (batch_size, dim_size) = ({}, {}), not {}'.format(x.shape[0], x.shape[self.dim], x_order.shape)
        permutation = torch.argsort(x_order, dim=1)
        for d in range(1, self.dim):
            permutation = permutation.unsqueeze(1)
        for d in range(self.dim+1, x.dim()):
            permutation = permutation.unsqueeze(-1)
        permutation = permutation.expand_as(x)
        z = torch.gather(x, self.dim, permutation)
        ldj = - self.buffer.new_ones(x.shape[0]) * torch.arange(1, 1+x.shape[self.dim]).type(self.buffer.dtype).log().sum()
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
