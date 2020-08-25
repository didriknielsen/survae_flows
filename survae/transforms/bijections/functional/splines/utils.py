import torch


def gather_elementwise(tensor, idx_tensor):
    '''
    For `tensor.shape = tensor_shape + (K,)`
    and `idx_tensor.shape = tensor_shape` with elements in {0,1,...,K-1}
    '''
    return tensor.gather(-1, idx_tensor[..., None])[..., 0]



# Taken from https://github.com/bayesiains/nsf/blob/master/utils/torchutils.py

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


# Taken from https://github.com/bayesiains/nsf/blob/master/utils/torchutils.py

def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)
