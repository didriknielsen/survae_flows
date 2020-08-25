import torch
from collections.abc import Iterable
from survae.transforms.bijections import Bijection


class AutoregressiveBijection(Bijection):
    """Transforms each input variable with an invertible elementwise bijection.

    The parameters of each invertible elementwise bijection can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.

    Args:
        autoregressive_net: nn.Module, an autoregressive network such that
            elementwise_params = autoregressive_net(x)
        autoregressive_order: str or Iterable, the order in which to sample.
            One of `{'ltr'}`
    """
    def __init__(self, autoregressive_net, autoregressive_order='ltr'):
        super(AutoregressiveBijection, self).__init__()
        assert isinstance(autoregressive_order, str) or isinstance(autoregressive_order, Iterable)
        assert autoregressive_order in {'ltr'}
        self.autoregressive_net = autoregressive_net
        self.autoregressive_order = autoregressive_order

    def forward(self, x):
        elementwise_params = self.autoregressive_net(x)
        z, ldj = self._elementwise_forward(x, elementwise_params)
        return z, ldj

    def inverse(self, z):
        with torch.no_grad():
            if self.autoregressive_order == 'ltr': return self._inverse_ltr(z)

    def _inverse_ltr(self, z):
        x = torch.zeros_like(z)
        for d in range(x.shape[1]):
            elementwise_params = self.autoregressive_net(x)
            x[:,d] = self._elementwise_inverse(z[:,d], elementwise_params[:,d])
        return x

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, x, elementwise_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, z, elementwise_params):
        raise NotImplementedError()
