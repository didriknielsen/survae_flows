import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms.bijections.autoregressive import *
from survae.nn.layers.autoregressive import SpatialMaskedConv2d, MaskedConv2d
from survae.nn.layers import ElementwiseParams, ElementwiseParams2d
from survae.tests.transforms.bijections import BijectionTest


class LinearSplineAutoregressiveBijection2dTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 16
        batch_size = 10
        shape = (3,8,8)
        x = torch.rand(batch_size, *shape)
        net_spatial = nn.Sequential(SpatialMaskedConv2d(3,3*num_bins, kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(num_bins))
        net = nn.Sequential(MaskedConv2d(3,3*num_bins, kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(num_bins))

        self.eps = 1e-3
        bijection = LinearSplineAutoregressiveBijection2d(net, num_bins=num_bins, autoregressive_order='raster_cwh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

        bijection = LinearSplineAutoregressiveBijection2d(net_spatial, num_bins=num_bins, autoregressive_order='raster_wh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))


class QuadraticSplineAutoregressiveBijection2dTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 16
        batch_size = 10
        shape = (3,8,8)
        x = torch.rand(batch_size, *shape)
        net_spatial = nn.Sequential(SpatialMaskedConv2d(3,3*(2*num_bins+1), kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(2*num_bins+1))
        net = nn.Sequential(MaskedConv2d(3,3*(2*num_bins+1), kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(2*num_bins+1))

        self.eps = 5e-3
        bijection = QuadraticSplineAutoregressiveBijection2d(net, num_bins=num_bins, autoregressive_order='raster_cwh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

        bijection = QuadraticSplineAutoregressiveBijection2d(net_spatial, num_bins=num_bins, autoregressive_order='raster_wh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))


class CubicSplineAutoregressiveBijection2dTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 16
        batch_size = 10
        shape = (3,8,8)
        x = torch.rand(batch_size, *shape)
        net_spatial = nn.Sequential(SpatialMaskedConv2d(3,3*(2*num_bins+2), kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(2*num_bins+2))
        net = nn.Sequential(MaskedConv2d(3,3*(2*num_bins+2), kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(2*num_bins+2))

        self.eps = 5e-3
        bijection = CubicSplineAutoregressiveBijection2d(net, num_bins=num_bins, autoregressive_order='raster_cwh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

        bijection = CubicSplineAutoregressiveBijection2d(net_spatial, num_bins=num_bins, autoregressive_order='raster_wh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))


class RationalQuadraticSplineAutoregressiveBijection2dTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 16
        batch_size = 10
        shape = (3,8,8)
        x = torch.rand(batch_size, *shape)
        net_spatial = nn.Sequential(SpatialMaskedConv2d(3,3*(3*num_bins+1), kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(3*num_bins+1))
        net = nn.Sequential(MaskedConv2d(3,3*(3*num_bins+1), kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(3*num_bins+1))

        self.eps = 1e-5
        bijection = RationalQuadraticSplineAutoregressiveBijection2d(net, num_bins=num_bins, autoregressive_order='raster_cwh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

        bijection = RationalQuadraticSplineAutoregressiveBijection2d(net_spatial, num_bins=num_bins, autoregressive_order='raster_wh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))


if __name__ == '__main__':
    unittest.main()
