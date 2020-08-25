import math
import torch
from survae.distributions import Distribution
from survae.utils import sum_except_batch


class StandardHalfNormal(Distribution):
    """A standard half-Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super(StandardHalfNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))

    def log_prob(self, x):
        log_scaling = math.log(2)
        log_base =    - 0.5 * math.log(2 * math.pi)
        log_inner =   - 0.5 * x**2
        log_probs = log_scaling+log_base+log_inner
        log_probs[x < 0] = -math.inf
        return sum_except_batch(log_probs)

    def sample(self, num_samples):
        return torch.randn(num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype).abs()
