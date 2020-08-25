import math
import torch
from survae.transforms.bijections.functional.mixtures.utils_logistic import logistic_log_cdf, logistic_log_one_minus_cdf, logistic_log_pdf


def cmol_cdf(x, means, log_scales, K):
    x = x.unsqueeze(-1)
    # For logistic_eval_x:
    # Matching the eval. locations of the logistic distribution from the PixelCNN++ code base:
    # Lower: From {0,1,...,255}/256 -> {0,1,...,255}/255 -> 2 * {0,1,...,255}/255 - 1 -> 2 * {0,1,...,255}/255 - 1 - 1/255 (min = -1-1/255)
    # Upper: From {1,2,...,256}/256 -> {1,2,...,256}/255 -> 2 * {1,2,...,256}/255 - 1 -> 2 * {1,2,...,256}/255 - 1 - 1/255 (max = 1+1/255)
    logistic_eval_x = K/(K-1) *(2 * x - 1) # From [0,1] -> [-1-1/255,1+1/255]
    logistic_eval_lower = - (1 + 1/(K-1)) * torch.ones(torch.Size(), dtype=x.dtype, device=x.device)[(None,)*x.dim()]
    logistic_eval_upper = (1 + 1/(K-1)) * torch.ones(torch.Size(), dtype=x.dtype, device=x.device)[(None,)*x.dim()]
    cdf_mid = logistic_log_cdf(logistic_eval_x, means, log_scales).exp()
    cdf_lower = (1-K*x) * logistic_log_cdf(logistic_eval_lower, means, log_scales).exp() * torch.lt(x, 1/K).type(x.dtype)
    cdf_upper = (K*x-(K-1)) * logistic_log_one_minus_cdf(logistic_eval_upper, means, log_scales).exp() * torch.gt(x, (K-1)/K).type(x.dtype)
    return cdf_mid + cdf_upper - cdf_lower


def cmol_log_pdf(x, means, log_scales, K):
    x = x.unsqueeze(-1)
    # For logistic_eval_x:
    # Matching the eval. locations of the logistic distribution from the PixelCNN++ code base:
    # Lower: From {0,1,...,255}/256 -> {0,1,...,255}/255 -> 2 * {0,1,...,255}/255 - 1 -> 2 * {0,1,...,255}/255 - 1 - 1/255 (min = -1-1/255)
    # Upper: From {1,2,...,256}/256 -> {1,2,...,256}/255 -> 2 * {1,2,...,256}/255 - 1 -> 2 * {1,2,...,256}/255 - 1 - 1/255 (max = 1+1/255)
    logistic_eval_x = K/(K-1) *(2 * x - 1) # From [0,1] -> [-1-1/255,1+1/255]
    logistic_eval_lower = - (1 + 1/(K-1)) * torch.ones(torch.Size(), dtype=x.dtype, device=x.device)[(None,)*x.dim()]
    logistic_eval_upper = (1 + 1/(K-1)) * torch.ones(torch.Size(), dtype=x.dtype, device=x.device)[(None,)*x.dim()]
    log_pdf_mid = math.log(2) + math.log(K) - math.log(K-1) + logistic_log_pdf(logistic_eval_x, means, log_scales)
    log_pdf_lower = math.log(K) + logistic_log_cdf(logistic_eval_lower, means, log_scales)
    log_pdf_upper = math.log(K) + logistic_log_one_minus_cdf(logistic_eval_upper, means, log_scales)
    log_pdf_lower.masked_fill_(~torch.lt(x, 1/K), value=-float('inf'))
    log_pdf_upper.masked_fill_(~torch.gt(x, (K-1)/K), value=-float('inf'))
    log_pdf_stack = torch.stack([log_pdf_lower, log_pdf_mid, log_pdf_upper], dim=-1)
    return torch.logsumexp(log_pdf_stack, dim=-1)
