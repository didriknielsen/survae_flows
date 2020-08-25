import torch
from survae.distributions import Distribution
from survae.utils import mean_except_batch


class StandardUniform(Distribution):
    """A multivariate Uniform with boundaries (0,1)."""

    def __init__(self, shape):
        super().__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('one', torch.ones(1))

    def log_prob(self, x):
        lb = mean_except_batch(x.ge(self.zero).type(self.zero.dtype))
        ub = mean_except_batch(x.le(self.one).type(self.one.dtype))
        return torch.log(lb*ub)

    def sample(self, num_samples):
        return torch.rand((num_samples,) + self.shape, device=self.zero.device, dtype=self.zero.dtype)
