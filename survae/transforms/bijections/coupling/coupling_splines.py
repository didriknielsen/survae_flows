import torch
from collections.abc import Iterable
from survae.utils import sum_except_batch
from survae.transforms.bijections.functional import splines
from survae.transforms.bijections.coupling import CouplingBijection


class LinearSplineCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_bins, split_dim=1, num_condition=None):
        super(LinearSplineCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return self.num_bins

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        z, ldj_elementwise = splines.linear_spline(x, elementwise_params, inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        x, _ = splines.linear_spline(z, elementwise_params, inverse=True)
        return x


class QuadraticSplineCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_bins, split_dim=1, num_condition=None):
        super(QuadraticSplineCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return 2 * self.num_bins + 1

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths, unnormalized_heights = elementwise_params[..., :self.num_bins], elementwise_params[..., self.num_bins:]
        z, ldj_elementwise = splines.quadratic_spline(x, unnormalized_widths=unnormalized_widths, unnormalized_heights=unnormalized_heights, inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths, unnormalized_heights = elementwise_params[..., :self.num_bins], elementwise_params[..., self.num_bins:]
        x, _ = splines.quadratic_spline(z, unnormalized_widths=unnormalized_widths, unnormalized_heights=unnormalized_heights, inverse=True)
        return x


class CubicSplineCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_bins, split_dim=1, num_condition=None):
        super(CubicSplineCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return 2 * self.num_bins + 2

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnorm_derivatives_left = elementwise_params[..., 2*self.num_bins:2*self.num_bins+1]
        unnorm_derivatives_right = elementwise_params[..., 2*self.num_bins+1:]
        z, ldj_elementwise = splines.cubic_spline(x,
                                                  unnormalized_widths=unnormalized_widths,
                                                  unnormalized_heights=unnormalized_heights,
                                                  unnorm_derivatives_left=unnorm_derivatives_left,
                                                  unnorm_derivatives_right=unnorm_derivatives_right,
                                                  inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnorm_derivatives_left = elementwise_params[..., 2*self.num_bins:2*self.num_bins+1]
        unnorm_derivatives_right = elementwise_params[..., 2*self.num_bins+1:]
        x, _ = splines.cubic_spline(z,
                                    unnormalized_widths=unnormalized_widths,
                                    unnormalized_heights=unnormalized_heights,
                                    unnorm_derivatives_left=unnorm_derivatives_left,
                                    unnorm_derivatives_right=unnorm_derivatives_right,
                                    inverse=True)
        return x


class RationalQuadraticSplineCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_bins, split_dim=1, num_condition=None):
        super(RationalQuadraticSplineCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return 3 * self.num_bins + 1

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = elementwise_params[..., 2*self.num_bins:]
        z, ldj_elementwise = splines.rational_quadratic_spline(x,
                                                               unnormalized_widths=unnormalized_widths,
                                                               unnormalized_heights=unnormalized_heights,
                                                               unnormalized_derivatives=unnormalized_derivatives,
                                                               inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = elementwise_params[..., 2*self.num_bins:]
        x, _ = splines.rational_quadratic_spline(z,
                                                 unnormalized_widths=unnormalized_widths,
                                                 unnormalized_heights=unnormalized_heights,
                                                 unnormalized_derivatives=unnormalized_derivatives,
                                                 inverse=True)
        return x
