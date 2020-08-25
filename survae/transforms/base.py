import torch
from torch import nn
from collections.abc import Iterable


class Transform(nn.Module):
    """Base class for Transform"""

    has_inverse = True

    @property
    def bijective(self):
        raise NotImplementedError()

    @property
    def stochastic_forward(self):
        raise NotImplementedError()

    @property
    def stochastic_inverse(self):
        raise NotImplementedError()

    @property
    def lower_bound(self):
        return self.stochastic_forward

    def forward(self, x):
        """
        Forward transform.
        Computes `z <- x` and the log-likelihood contribution term `log C`
        such that `log p(x) = log p(z) + log C`.

        Args:
            x: Tensor, shape (batch_size, ...)

        Returns:
            z: Tensor, shape (batch_size, ...)
            ldj: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def inverse(self, z):
        """
        Inverse transform.
        Computes `x <- z`.

        Args:
            z: Tensor, shape (batch_size, ...)

        Returns:
            x: Tensor, shape (batch_size, ...)
        """
        raise NotImplementedError()


class SequentialTransform(Transform):
    """
    Chains multiple Transform objects sequentially.

    Args:
        transforms: Transform or iterable with each element being a Transform object
    """

    def __init__(self, transforms):
        super(SequentialTransform, self).__init__()
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.has_inverse = all(transform.has_inverse for transform in transforms)
        self.transforms = nn.ModuleList(transforms)

    @property
    def bijective(self):
        return all(transform.bijective for transform in self.transforms)

    @property
    def stochastic_forward(self):
        return any(transform.stochastic_forward for transform in self.transforms)

    @property
    def stochastic_inverse(self):
        return any(transform.stochastic_inverse for transform in self.transforms)

    def forward(self, x):
        batch_size = x.shape[0]
        x, ldj = self.transforms[0].forward(x)
        for transform in self.transforms[1:]:
            x, l = transform.forward(x)
            ldj += l
        return x, ldj

    def inverse(self, z):
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z
