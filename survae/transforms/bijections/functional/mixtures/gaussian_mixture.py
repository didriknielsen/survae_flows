import torch
import torch.nn.functional as F
from torch.distributions import Normal
from survae.transforms.bijections.functional.iterative_inversion import bisection_inverse


def gaussian_mixture_transform(inputs, logit_weights, means, log_scales, eps=1e-10, max_iters=100, inverse=False):
    '''
    Univariate mixture of Gaussians transform.

    Args:
        inputs: torch.Tensor, shape (shape,)
        logit_weights: torch.Tensor, shape (shape, num_mixtures)
        means: torch.Tensor, shape (shape, num_mixtures)
        log_scales: torch.Tensor, shape (shape, num_mixtures)
        eps: float, tolerance for bisection |f(x) - z_est| < eps
        max_iters: int, maximum iterations for bisection
        inverse: bool, if True, return inverse
    '''

    log_weights = F.log_softmax(logit_weights, dim=-1)
    dist = Normal(means, log_scales.exp())

    def mix_cdf(x):
        return torch.sum(log_weights.exp() * dist.cdf(x.unsqueeze(-1)), dim=-1)

    def mix_log_pdf(x):
        return torch.logsumexp(log_weights + dist.log_prob(x.unsqueeze(-1)), dim=-1)

    if inverse:
        max_scales = torch.sum(torch.exp(log_scales), dim=-1, keepdim=True)
        init_lower, _ = (means - 20 * max_scales).min(dim=-1)
        init_upper, _ = (means + 20 * max_scales).max(dim=-1)
        return bisection_inverse(fn=lambda x: mix_cdf(x),
                                 z=inputs,
                                 init_x=torch.zeros_like(inputs),
                                 init_lower=init_lower,
                                 init_upper=init_upper,
                                 eps=eps,
                                 max_iters=max_iters)
    else:
        z = mix_cdf(inputs)
        ldj = mix_log_pdf(inputs)
        return z, ldj
