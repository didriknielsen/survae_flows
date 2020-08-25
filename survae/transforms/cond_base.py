import torch
from torch import nn
from collections.abc import Iterable
from survae.transforms import Transform


class ConditionalTransform(Transform):
    """Base class for ConditionalTransform"""

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

    def forward(self, x, context):
        """
        Forward transform.
        Computes `z = f(x|context)` and `log|det J|` for `J = df(x|context)/dx`
        such that `log p_x(x|context) = log p_z(f(x|context)) + log|det J|`.

        Args:
            x: Tensor, shape (batch_size, ...)
            context: Tensor, shape (batch_size, ...).

        Returns:
            z: Tensor, shape (batch_size, ...)
            ldj: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def inverse(self, z, context):
        """
        Inverse transform.
        Computes `x = f^{-1}(z|context)`.

        Args:
            z: Tensor, shape (batch_size, ...)
            context: Tensor, shape (batch_size, ...).

        Returns:
            x: Tensor, shape (batch_size, ...)
        """
        raise NotImplementedError()
