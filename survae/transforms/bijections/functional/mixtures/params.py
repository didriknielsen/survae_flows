import torch


def get_mixture_params(params, num_mixtures):
    '''Get parameters for mixture transforms.'''
    assert params.shape[-1] == 3 * num_mixtures

    unnormalized_weights = params[..., :num_mixtures]
    means = params[..., num_mixtures:2*num_mixtures]
    log_scales = params[..., 2*num_mixtures:3*num_mixtures]

    return unnormalized_weights, means, log_scales
