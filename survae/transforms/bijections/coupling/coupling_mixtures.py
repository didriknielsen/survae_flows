import torch
from survae.utils import sum_except_batch
from survae.transforms.bijections.functional.mixtures import gaussian_mixture_transform, logistic_mixture_transform, censored_logistic_mixture_transform
from survae.transforms.bijections.functional.mixtures import get_mixture_params
from survae.transforms.bijections.coupling import CouplingBijection


class GaussianMixtureCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_mixtures, split_dim=1, num_condition=None):
        super(GaussianMixtureCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
        self.num_mixtures = num_mixtures
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, elementwise_params, inverse):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)

        x = gaussian_mixture_transform(inputs=inputs,
                                       logit_weights=logit_weights,
                                       means=means,
                                       log_scales=log_scales,
                                       eps=self.eps,
                                       max_iters=self.max_iters,
                                       inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)


class LogisticMixtureCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_mixtures, split_dim=1, num_condition=None):
        super(LogisticMixtureCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
        self.num_mixtures = num_mixtures
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, elementwise_params, inverse):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)

        x = logistic_mixture_transform(inputs=inputs,
                                       logit_weights=logit_weights,
                                       means=means,
                                       log_scales=log_scales,
                                       eps=self.eps,
                                       max_iters=self.max_iters,
                                       inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)


class CensoredLogisticMixtureCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_mixtures, num_bins, split_dim=1, num_condition=None):
        super(CensoredLogisticMixtureCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
        self.num_mixtures = num_mixtures
        self.num_bins = num_bins
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, elementwise_params, inverse):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)

        x = censored_logistic_mixture_transform(inputs=inputs,
                                                logit_weights=logit_weights,
                                                means=means,
                                                log_scales=log_scales,
                                                num_bins=self.num_bins,
                                                eps=self.eps,
                                                max_iters=self.max_iters,
                                                inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)
