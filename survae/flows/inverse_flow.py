import torch
from torch import nn
from collections.abc import Iterable
from survae.distributions import Distribution
from survae.transforms import Transform


class InverseFlow(Distribution):
    """
    Base class for InverseFlow.
    Inverse flows use the forward transforms to transform noise to samples.
    These are typically useful as variational distributions.
    Here, we are not interested in the log probability of novel samples.
    However, using .sample_with_log_prob(), samples can be obtained together
    with their log probability.
    """

    def __init__(self, base_dist, transforms):
        super(InverseFlow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)

    def log_prob(self, x):
        raise RuntimeError("InverseFlow does not support log_prob, see Flow instead.")

    def sample(self, num_samples):
        z = self.base_dist.sample(num_samples)
        for transform in self.transforms:
            z, _ = transform(z)
        return z

    def sample_with_log_prob(self, num_samples):
        z, log_prob = self.base_dist.sample_with_log_prob(num_samples)
        for transform in self.transforms:
            z, ldj = transform(z)
            log_prob -= ldj
        return z, log_prob
