import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.nn.layers.autoregressive.utils import mask_conv2d_spatial, mask_conv2d


class _MaskedConv2d(nn.Conv2d):
    """
    A masked version of nn.Conv2d.
    """

    def register_mask(self, mask):
        """
        Registers mask to be used in forward pass.

        Input:
            mask: torch.FloatTensor
                Shape needs to be broadcastable with self.weight.
        """
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(_MaskedConv2d, self).forward(x)


class SpatialMaskedConv2d(_MaskedConv2d):
    """
    A version of nn.Conv2d masked to be autoregressive in the spatial dimensions.
    Uses mask of shape (1, 1, height, width).

    Input:
        *args: Arguments passed to the constructor of nn.Conv2d.
        mask_type: str
            Either 'A' or 'B'. 'A' for first layer of network, 'B' for all others.
        **kwargs: Keyword arguments passed to the constructor of nn.Conv2d.
    """

    def __init__(self, *args, mask_type, **kwargs):
        super(SpatialMaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        _, _, height, width = self.weight.size()
        mask = mask_conv2d_spatial(mask_type, height, width)
        self.register_mask(mask)


class MaskedConv2d(_MaskedConv2d):
    """
    A version of nn.Conv2d masked to be autoregressive in
    the spatial dimensions and in the channel dimension.
    This is constructed specifically for data that
    has any number of input channels.
    Uses mask of shape (out_channels, in_channels, height, width).

    Input:
        *args: Arguments passed to the constructor of nn.Conv2d.
        mask_type: str
            Either 'A' or 'B'. 'A' for first layer of network, 'B' for all others.
        data_channels: int
            Number of channels in the input data, e.g. 3 for RGB images. Default: 3.
            This will be used to mask channels throughout the newtork such that
            all feature maps will have order (R, G, B, R, G, B, ...).
            In the case of mask_type B, for the central pixel:
            Outputs in position R can only access inputs in position R.
            Outputs in position G can access inputs in position R and G.
            Outputs in position B can access inputs in position R, G and B.
            In the case of mask_type A, for the central pixel:
            Outputs in position G can only access inputs in position R.
            Outputs in position B can access inputs in position R and G.
        **kwargs: Keyword arguments passed to the constructor of nn.Conv2d.
    """

    def __init__(self, *args, mask_type, data_channels=3, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        out_channels, in_channels, height, width = self.weight.size()
        mask = mask_conv2d(mask_type, in_channels, out_channels, height, width, data_channels)
        self.register_mask(mask)
