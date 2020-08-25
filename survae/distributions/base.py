import math
import torch
from torch import nn


class Distribution(nn.Module):
    """Distribution base class."""

    def log_prob(self, x):
        """Calculate log probability under the distribution.

        Args:
            x: Tensor, shape (batch_size, ...)

        Returns:
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def sample(self, num_samples):
        """Generates samples from the distribution.

        Args:
            num_samples: int, number of samples to generate.

        Returns:
            samples: Tensor, shape (num_samples, ...)
        """
        raise NotImplementedError()

    def sample_with_log_prob(self, num_samples):
        """Generates samples from the distribution together with their log probability.

        Args:
            num_samples: int, number of samples to generate.

        Returns:
            samples: Tensor, shape (num_samples, ...)
            log_prob: Tensor, shape (num_samples,)
        """
        samples = self.sample(num_samples)
        log_prob = self.log_prob(samples)
        return samples, log_prob

    def forward(self, *args, mode, **kwargs):
        '''
        To allow Distribution objects to be wrapped by DataParallelDistribution,
        which parallelizes .forward() of replicas on subsets of data.

        DataParallelDistribution.log_prob() calls DataParallel.forward().
        DataParallel.forward() calls Distribution.forward() for different
        data subsets on each device and returns the combined outputs.
        '''
        if mode == 'log_prob':
            return self.log_prob(*args, **kwargs)
        else:
            raise RuntimeError("Mode {} not supported.".format(mode))
