import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms.bijections.autoregressive import *
from survae.nn.layers.autoregressive import SpatialMaskedConv2d, MaskedConv2d
from survae.nn.layers import ElementwiseParams, ElementwiseParams2d
from survae.tests.transforms.bijections import BijectionTest


class GaussianMixtureAutoregressiveBijection2dTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_mix = 8
        batch_size = 10
        shape = (3,4,4)
        elementwise_params = 3 * num_mix
        x = torch.randn(batch_size, *shape)
        net_spatial = nn.Sequential(SpatialMaskedConv2d(3,3*elementwise_params, kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(elementwise_params))
        net = nn.Sequential(MaskedConv2d(3,3*elementwise_params, kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(elementwise_params))

        self.eps = 1e-4
        bijection = GaussianMixtureAutoregressiveBijection2d(net, num_mixtures=num_mix, autoregressive_order='raster_cwh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

        bijection = GaussianMixtureAutoregressiveBijection2d(net_spatial, num_mixtures=num_mix, autoregressive_order='raster_wh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))


class LogisticMixtureAutoregressiveBijection2dTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_mix = 8
        batch_size = 10
        shape = (3,4,4)
        elementwise_params = 3 * num_mix
        x = torch.randn(batch_size, *shape)
        net_spatial = nn.Sequential(SpatialMaskedConv2d(3,3*elementwise_params, kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(elementwise_params))
        net = nn.Sequential(MaskedConv2d(3,3*elementwise_params, kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(elementwise_params))

        self.eps = 5e-5
        bijection = LogisticMixtureAutoregressiveBijection2d(net, num_mixtures=num_mix, autoregressive_order='raster_cwh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

        bijection = LogisticMixtureAutoregressiveBijection2d(net_spatial, num_mixtures=num_mix, autoregressive_order='raster_wh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))


class CensoredLogisticMixtureAutoregressiveBijection2dTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 4
        num_mix = 8
        batch_size = 10
        shape = (3,4,4)
        elementwise_params = 3 * num_mix
        x = torch.rand(batch_size, *shape)
        net_spatial = nn.Sequential(SpatialMaskedConv2d(3,3*elementwise_params, kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(elementwise_params))
        net = nn.Sequential(MaskedConv2d(3,3*elementwise_params, kernel_size=3, padding=1, mask_type='A'), ElementwiseParams2d(elementwise_params))

        self.eps = 1e-6
        bijection = CensoredLogisticMixtureAutoregressiveBijection2d(net, num_mixtures=num_mix, num_bins=num_bins, autoregressive_order='raster_cwh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

        bijection = CensoredLogisticMixtureAutoregressiveBijection2d(net_spatial, num_mixtures=num_mix, num_bins=num_bins, autoregressive_order='raster_wh')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))




if __name__ == '__main__':
    unittest.main()
