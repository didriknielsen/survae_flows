import torch
from survae.transforms.bijections.conditional import ConditionalBijection


class ConditionalAutoregressiveBijection2d(ConditionalBijection):
    """Transforms each input variable with an invertible elementwise bijection.

    The parameters of each invertible elementwise bijection can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.

    Args:
        autoregressive_net: nn.Module, an autoregressive network such that
            elementwise_params = autoregressive_net(x)
        autoregressive_order: str or Iterable, the order in which to sample.
            One of `{'raster_cwh', 'raster_wh'}`
    """
    def __init__(self, autoregressive_net, autoregressive_order='raster_cwh'):
        super(ConditionalAutoregressiveBijection2d, self).__init__()
        assert isinstance(autoregressive_order, str) or isinstance(autoregressive_order, Iterable)
        assert autoregressive_order in {'raster_cwh', 'raster_wh'}
        self.autoregressive_net = autoregressive_net
        self.autoregressive_order = autoregressive_order

    def forward(self, x, context):
        elementwise_params = self.autoregressive_net(x, context=context)
        z, ldj = self._elementwise_forward(x, elementwise_params)
        return z, ldj

    def inverse(self, z, context):
        with torch.no_grad():
            if self.autoregressive_order == 'raster_cwh': return self._inverse_raster_cwh(z, context=context)
            if self.autoregressive_order == 'raster_wh': return self._inverse_raster_wh(z, context=context)

    def _inverse_raster_cwh(self, z, context):
        x = torch.zeros_like(z)
        for h in range(x.shape[2]):
            for w in range(x.shape[3]):
                for c in range(x.shape[1]):
                    elementwise_params = self.autoregressive_net(x, context=context)
                    x[:,c,h,w] = self._elementwise_inverse(z[:,c,h,w], elementwise_params[:,c,h,w])
        return x

    def _inverse_raster_wh(self, z, context):
        x = torch.zeros_like(z)
        for h in range(x.shape[2]):
            for w in range(x.shape[3]):
                elementwise_params = self.autoregressive_net(x, context=context)
                x[:,:,h,w] = self._elementwise_inverse(z[:,:,h,w], elementwise_params[:,:,h,w])
        return x

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, x, elementwise_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, z, elementwise_params):
        raise NotImplementedError()
