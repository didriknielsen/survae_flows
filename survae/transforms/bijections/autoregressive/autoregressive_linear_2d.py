import torch
from survae.utils import sum_except_batch
from survae.transforms.bijections.autoregressive import AutoregressiveBijection2d


class AdditiveAutoregressiveBijection2d(AutoregressiveBijection2d):
    '''Additive autoregressive bijection.'''

    def _output_dim_multiplier(self):
        return 1

    def _elementwise_forward(self, x, elementwise_params):
        return x + elementwise_params, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def _elementwise_inverse(self, z, elementwise_params):
        return z - elementwise_params


class AffineAutoregressiveBijection2d(AutoregressiveBijection2d):
    '''
    Affine autoregressive bijection.

    Args:
        autoregressive_net: nn.Module, an autoregressive network such that
            elementwise_params = autoregressive_net(x)
        autoregressive_order: str or Iterable, the order in which to sample.
            One of `{'raster_cwh', 'raster_wh'}`
        scale_fn: callable, the transform to obtain the scale.
    '''

    def __init__(self, autoregressive_net, autoregressive_order='raster_cwh', scale_fn=lambda s: torch.exp(s)):
        super(AffineAutoregressiveBijection2d, self).__init__(autoregressive_net=autoregressive_net, autoregressive_order=autoregressive_order)
        assert callable(scale_fn)
        self.scale_fn = scale_fn

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        scale = self.scale_fn(unconstrained_scale)
        z = scale * x + shift
        ldj = sum_except_batch(torch.log(scale))
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        scale = self.scale_fn(unconstrained_scale)
        x = (z - shift) / scale
        return x

    def _unconstrained_scale_and_shift(self, elementwise_params):
        unconstrained_scale = elementwise_params[..., 0]
        shift = elementwise_params[..., 1]
        return unconstrained_scale, shift
