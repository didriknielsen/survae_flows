import torch
import torch.nn.functional as F


def logistic_log_cdf(x, means, log_scales):
    """
    Logistic logistic logistic function.

    Args:
        x: (array): write your description
        means: (array): write your description
        log_scales: (todo): write your description
    """
    return F.logsigmoid(torch.exp(-log_scales) * (x - means))


def logistic_log_one_minus_cdf(x, means, log_scales):
    '''
    Uses that:
    `log(1-sigmoid(x)) = - softplus(x)`
    '''
    return -F.softplus(torch.exp(-log_scales) * (x - means))


def logistic_log_pdf(x, means, log_scales):
    '''
    Uses that:
    pdf(x) = dcdf(x)/dx
           = dsigmoid((x-m)/s)/dx
           = 1/s * sigmoid((x-m)/s) * (1-sigmoid((x-m)/s))
    '''
    return - log_scales + logistic_log_cdf(x, means, log_scales) + logistic_log_one_minus_cdf(x, means, log_scales)
