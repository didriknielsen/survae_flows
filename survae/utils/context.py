import torch


def context_size(context):
    """
    Returns the size of the size.

    Args:
        context: (dict): write your description
    """
    while not isinstance(context, torch.Tensor):
        first_key = list(context.keys())[0]
        context = context[first_key]
    return context.shape[0]
