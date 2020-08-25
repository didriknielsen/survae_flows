import torch
import torch.nn as nn
from survae.nn.blocks import ResidualDenseBlock


class DenseNet(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_blocks,
                 mid_channels, depth, growth, dropout,
                 gated_conv=False, zero_init=False):

        layers = [nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)] +\
                 [ResidualDenseBlock(in_channels=mid_channels,
                                     out_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=False) for _ in range(num_blocks)] +\
                 [nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)]

        if zero_init:
            nn.init.zeros_(layers[-1].weight)
            if hasattr(layers[-1], 'bias'):
                nn.init.zeros_(layers[-1].bias)

        super(DenseNet, self).__init__(*layers)
