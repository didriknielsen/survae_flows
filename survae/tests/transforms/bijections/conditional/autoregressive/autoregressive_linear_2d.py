import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms.bijections.conditional.autoregressive import *
from survae.nn.layers.autoregressive import SpatialMaskedConv2d, MaskedConv2d
from survae.nn.layers import ElementwiseParams, ElementwiseParams2d
from survae.tests.transforms.bijections.conditional import ConditionalBijectionTest


class CondNet(nn.Module):

    def __init__(self, channels, context_channels, params, autoregressive_order):
        super(CondNet, self).__init__()
        if autoregressive_order=='raster_cwh':
            self.conv = MaskedConv2d(channels, channels*params, kernel_size=3, padding=1, mask_type='A')
        elif autoregressive_order=='raster_wh':
            self.conv = SpatialMaskedConv2d(channels, channels*params, kernel_size=3, padding=1, mask_type='A')
        self.context = nn.Conv2d(context_channels, channels*params, kernel_size=1)
        self.out = ElementwiseParams2d(params)

    def forward(self, x, context):
        h = self.context(context) + self.conv(x)
        return 0.1 * self.out(h)


class ConditionalAdditiveAutoregressiveBijection2dTest(ConditionalBijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = (3,8,8)
        x = torch.randn(batch_size, *shape)
        context = torch.randn(batch_size, *shape)
        net_spatial = CondNet(3, context_channels=3, params=1, autoregressive_order='raster_wh')
        net = CondNet(3, context_channels=3, params=1, autoregressive_order='raster_cwh')

        self.eps = 1e-6
        bijection = ConditionalAdditiveAutoregressiveBijection2d(net, autoregressive_order='raster_cwh')
        self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))

        bijection = ConditionalAdditiveAutoregressiveBijection2d(net_spatial, autoregressive_order='raster_wh')
        self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))


class ConditionalAffineAutoregressiveBijection2dTest(ConditionalBijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = (3,8,8)
        x = torch.randn(batch_size, *shape)
        context = torch.randn(batch_size, *shape)
        net_spatial = CondNet(3, context_channels=3, params=2, autoregressive_order='raster_wh')
        net = CondNet(3, context_channels=3, params=2, autoregressive_order='raster_cwh')

        self.eps = 1e-6
        for autoregressive_order in ['raster_cwh', 'raster_wh']:
            with self.subTest(autoregressive_order=autoregressive_order):
                if autoregressive_order=='raster_cwh': autoreg_net = net
                elif autoregressive_order=='raster_wh': autoreg_net = net_spatial
                bijection = ConditionalAffineAutoregressiveBijection2d(autoreg_net, autoregressive_order=autoregressive_order)
                self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))


if __name__ == '__main__':
    unittest.main()
