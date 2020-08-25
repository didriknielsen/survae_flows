import torch
from torch import nn
from collections.abc import Iterable
from survae.utils import context_size
from survae.distributions import Distribution, ConditionalDistribution
from survae.transforms import Transform, ConditionalTransform


class ConditionalInverseFlow(ConditionalDistribution):
    """
    Base class for ConditionalFlow.
    Inverse flows use the forward transforms to transform noise to samples.
    These are typically useful as variational distributions.
    Here, we are not interested in the log probability of novel samples.
    However, using .sample_with_log_prob(), samples can be obtained together
    with their log probability.
    """

    def __init__(self, base_dist, transforms, context_init=None):
        super(ConditionalInverseFlow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.context_init = context_init

    def log_prob(self, x, context):
        raise RuntimeError("ConditionalInverseFlow does not support log_prob, see ConditionalFlow instead.")

    def sample(self, context):
        if self.context_init: context = self.context_init(context)
        if isinstance(self.base_dist, ConditionalDistribution):
            z = self.base_dist.sample(context)
        else:
            z = self.base_dist.sample(context_size(context))
        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                z, _ = transform(z, context)
            else:
                z, _ = transform(z)
        return z

    def sample_with_log_prob(self, context):
        if self.context_init: context = self.context_init(context)
        if isinstance(self.base_dist, ConditionalDistribution):
            z, log_prob = self.base_dist.sample_with_log_prob(context)
        else:
            z, log_prob = self.base_dist.sample_with_log_prob(context_size(context))
        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                z, ldj = transform(z, context)
            else:
                z, ldj = transform(z)
            log_prob -= ldj
        return z, log_prob
