import torch
import torch.nn as nn
import torch.nn.functional as F

from survae.nn.layers import LambdaLayer, ElementwiseParams2d
from survae.nn.layers.autoregressive import MaskedConv2d
from survae.nn.blocks.autoregressive import MaskedResidualBlock2d


class PixelCNN(nn.Sequential):
    '''PixelCNN (van den Oord et al., 2016) (https://arxiv.org/abs/1601.06759).'''

    def __init__(self, in_channels, num_params, filters=128, num_blocks=15, output_filters=1024, kernel_size=3, kernel_size_in=7, init_transforms=lambda x: 2*x-1):

        layers = [LambdaLayer(init_transforms)] +\
                 [MaskedConv2d(in_channels, 2 * filters, kernel_size=kernel_size_in, padding=kernel_size_in//2, mask_type='A', data_channels=in_channels)] +\
                 [MaskedResidualBlock2d(filters, data_channels=in_channels, kernel_size=kernel_size) for _ in range(num_blocks)] +\
                 [nn.ReLU(True), MaskedConv2d(2 * filters, output_filters, kernel_size=1, mask_type='B', data_channels=in_channels)] +\
                 [nn.ReLU(True), MaskedConv2d(output_filters, num_params * in_channels, kernel_size=1, mask_type='B', data_channels=in_channels)] +\
                 [ElementwiseParams2d(num_params)]

        super(PixelCNN, self).__init__(*layers)
