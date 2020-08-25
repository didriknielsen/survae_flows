import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.nn.layers.autoregressive import MaskedConv2d, SpatialMaskedConv2d


class MaskedResidualBlock2d(nn.Module):

    def __init__(self, h, kernel_size=3, data_channels=3):
        super(MaskedResidualBlock2d, self).__init__()

        self.conv1 = MaskedConv2d(2 * h, h, kernel_size=1, mask_type='B', data_channels=data_channels)
        self.conv2 = MaskedConv2d(h, h, kernel_size=kernel_size, padding=kernel_size//2, mask_type='B', data_channels=data_channels)
        self.conv3 = MaskedConv2d(h, 2 * h, kernel_size=1, mask_type='B', data_channels=data_channels)

    def forward(self, x):
        identity = x

        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))

        return x + identity


class SpatialMaskedResidualBlock2d(nn.Module):

    def __init__(self, h, kernel_size=3):
        super(SpatialMaskedResidualBlock2d, self).__init__()

        self.conv1 = nn.Conv2d(2 * h, h, kernel_size=1)
        self.conv2 = SpatialMaskedConv2d(h, h, kernel_size=kernel_size, padding=kernel_size//2, mask_type='B')
        self.conv3 = nn.Conv2d(h, 2 * h, kernel_size=1)

    def forward(self, x):
        identity = x

        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))

        return x + identity
