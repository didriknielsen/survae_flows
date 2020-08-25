import torch


def mask_conv2d_spatial(mask_type, height, width):
    """
    Creates a mask for Conv2d such that it becomes autoregressive in
    the spatial dimensions.

    Input:
        mask_type: str
            Either 'A' or 'B'. 'A' for first layer of network, 'B' for all others.
        height: int
            Kernel height for layer.
        width: int
            Kernel width for layer.
    Output:
        mask: torch.FloatTensor
            Shape (1, 1, height, width).
            A mask with 0 in places for masked elements.
    """
    mask = torch.ones([1, 1, height, width])
    mask[:, :, height // 2, width // 2 + (mask_type == 'B'):] = 0
    mask[:, :, height // 2 + 1:] = 0
    return mask


def mask_channels(mask_type, in_channels, out_channels, data_channels=3):
    """
    Creates an autoregressive channel mask.

    Input:
        mask_type: str
            Either 'A' or 'B'. 'A' for first layer of network, 'B' for all others.
        in_channels: int
            Number of input channels to layer.
        out_channels: int
            Number of output channels of layer.
        data_channels: int
            Number of channels in the input data, e.g. 3 for RGB images. (default = 3).
    Output:
        mask: torch.FloatTensor
            Shape (out_channels, in_channels).
            A mask with 0 in places for masked elements.
    """
    in_factor = in_channels // data_channels + 1
    out_factor = out_channels // data_channels + 1

    base_mask = torch.ones([data_channels,data_channels])
    if mask_type == 'A':
        base_mask = base_mask.tril(-1)
    else:
        base_mask = base_mask.tril(0)

    mask_p1 = torch.cat([base_mask]*in_factor, dim=1)
    mask_p2 = torch.cat([mask_p1]*out_factor, dim=0)

    mask = mask_p2[0:out_channels,0:in_channels]
    return mask


def mask_conv2d(mask_type, in_channels, out_channels, height, width, data_channels=3):
    r"""
    Creates a mask for Conv2d such that it becomes autoregressive in both
    the spatial dimensions and the channel dimension.

    Input:
        mask_type: str
            Either 'A' or 'B'. 'A' for first layer of network, 'B' for all others.
        in_channels: int
            Number of input channels to layer.
        out_channels: int
            Number of output channels of layer.
        height: int
            Kernel height for layer.
        width: int
            Kernel width for layer.
        data_channels: int
            Number of channels in the input data, e.g. 3 for RGB images. (default = 3).
    Output:
        mask: torch.FloatTensor
            Shape (out_channels, in_channels, height, width).
            A mask with 0 in places for masked elements.
    """
    mask = torch.ones([out_channels,in_channels,height,width])
    # RGB masking in central pixel
    mask[:, :, height // 2, width // 2] = mask_channels(mask_type, in_channels, out_channels, data_channels)
    # Masking all pixels to the right of the central pixel
    mask[:, :, height // 2, width // 2 + 1:] = 0
    # Masking all pixels below the central pixel
    mask[:, :, height // 2 + 1:] = 0
    return mask
