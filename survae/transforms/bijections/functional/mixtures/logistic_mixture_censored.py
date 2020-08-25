import torch
import torch.nn.functional as F
from survae.transforms.bijections.functional.iterative_inversion import bisection_inverse
from survae.transforms.bijections.functional.mixtures.utils_logistic_censored import cmol_cdf, cmol_log_pdf


def censored_logistic_mixture_transform(inputs, logit_weights, means, log_scales, num_bins=256, eps=1e-10, max_iters=100, inverse=False):
    '''
    Censored univariate mixture of logistics transform.

    Args:
        inputs: torch.Tensor, shape (shape,)
        logit_weights: torch.Tensor, shape (shape, num_mixtures)
        means: torch.Tensor, shape (shape, num_mixtures)
        log_scales: torch.Tensor, shape (shape, num_mixtures)
        num_bins: int, the number of bins
        eps: float, tolerance for bisection |f(x) - z_est| < eps
        max_iters: int, maximum iterations for bisection
        inverse: bool, if True, return inverse
    '''

    log_weights = F.log_softmax(logit_weights, dim=-1)
    log_scales = log_scales.clamp(min=-7.0)

    def mix_cdf(x):
        return torch.sum(log_weights.exp() * cmol_cdf(x, means, log_scales, num_bins), dim=-1)

    def mix_log_pdf(x):
        return torch.logsumexp(log_weights + cmol_log_pdf(x, means, log_scales, num_bins), dim=-1)

    if inverse:
        return bisection_inverse(fn=lambda x: mix_cdf(x),
                                 z=inputs,
                                 init_x=torch.ones_like(inputs) * 0.5,
                                 init_lower=torch.zeros_like(inputs),
                                 init_upper=torch.ones_like(inputs),
                                 eps=eps,
                                 max_iters=max_iters)
    else:
        z = mix_cdf(inputs)
        ldj = mix_log_pdf(inputs)
        return z, ldj
