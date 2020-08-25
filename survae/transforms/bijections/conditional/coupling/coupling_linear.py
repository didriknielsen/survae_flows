import torch
from survae.utils import sum_except_batch
from survae.transforms.bijections.conditional.coupling import ConditionalCouplingBijection


class ConditionalAdditiveCouplingBijection(ConditionalCouplingBijection):
    '''Conditional additive coupling bijection.'''

    def _output_dim_multiplier(self):
        return 1

    def _elementwise_forward(self, x, elementwise_params):
        return x + elementwise_params, torch.zeros(x.shape[0], device=x.device)

    def _elementwise_inverse(self, z, elementwise_params):
        return z - elementwise_params


class ConditionalAffineCouplingBijection(ConditionalCouplingBijection):
    '''
    Conditional affine coupling bijection.

    Args:
        coupling_net: nn.Module, a coupling network such that for x = [x1,x2]
            elementwise_params = coupling_net([x1,context])
        context_net: nn.Module or None, a network to process the context.
        split_dim: int, dimension to split the input (default=1).
        num_condition: int or None, number of parameters to condition on.
            If None, the first half is conditioned on:
            - For even inputs (1,2,3,4), (1,2) will be conditioned on.
            - For odd inputs (1,2,3,4,5), (1,2,3) will be conditioned on.
        scale_fn: callable, the transform to obtain the scale.
    '''

    def __init__(self, coupling_net, context_net=None, split_dim=1, num_condition=None, scale_fn=lambda s: torch.exp(s)):
        super(ConditionalAffineCouplingBijection, self).__init__(coupling_net=coupling_net, context_net=context_net, split_dim=split_dim, num_condition=num_condition)
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
