import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms.bijections.autoregressive import *
from survae.nn.layers.autoregressive import SpatialMaskedConv2d, MaskedConv2d
from survae.nn.layers import ElementwiseParams, ElementwiseParams2d
from survae.tests.transforms.bijections import BijectionTest


class AdditiveAutoregressiveBijection2dTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = (3,8,8)
        x = torch.randn(batch_size, *shape)
        net_spatial = SpatialMaskedConv2d(3,3, kernel_size=3, padding=1, mask_type='A')
        net = MaskedConv2d(3,3, kernel_size=3, padding=1, mask_type='A')

        self.eps = 1e-6
        bijection = AdditiveAutoregressiveBijection2d(net, autoregressive_order='raster_cwh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

        bijection = AdditiveAutoregressiveBijection2d(net_spatial, autoregressive_order='raster_wh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))


class AffineAutoregressiveBijection2dTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = (3,8,8)
        x = torch.randn(batch_size, *shape)
        net_spatial = nn.Sequential(SpatialMaskedConv2d(3,3*2, kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(2))
        net = nn.Sequential(MaskedConv2d(3,3*2, kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(2))

        self.eps = 1e-6
        for autoregressive_order in ['raster_cwh', 'raster_wh']:
            with self.subTest(autoregressive_order=autoregressive_order):
                if autoregressive_order=='raster_cwh': autoreg_net = net
                elif autoregressive_order=='raster_wh': autoreg_net = net_spatial
                bijection = AffineAutoregressiveBijection2d(autoreg_net, autoregressive_order=autoregressive_order)
                self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))


if __name__ == '__main__':
    unittest.main()
