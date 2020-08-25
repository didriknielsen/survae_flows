import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms.bijections.autoregressive import *
from survae.nn.nets.autoregressive import MADE
from survae.tests.transforms.bijections import BijectionTest


class GaussianMixtureAutoregressiveBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_mix = 4
        batch_size = 10
        features = 7
        x = torch.randn(batch_size, features)
        net = MADE(features, num_params=3*num_mix, hidden_features=[21])

        self.eps = 1e-4
        bijection = GaussianMixtureAutoregressiveBijection(net, num_mixtures=num_mix, autoregressive_order='ltr')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, features))


class LogisticMixtureAutoregressiveBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_mix = 4
        batch_size = 10
        features = 7
        x = torch.randn(batch_size, features)
        net = MADE(features, num_params=3*num_mix, hidden_features=[21])

        self.eps = 5e-5
        bijection = LogisticMixtureAutoregressiveBijection(net, num_mixtures=num_mix, autoregressive_order='ltr')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, features))


class CensoredLogisticMixtureAutoregressiveBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 16
        num_mix = 4
        batch_size = 10
        features = 7
        x = torch.rand(batch_size, features)
        net = MADE(features, num_params=3*num_mix, hidden_features=[21])

        self.eps = 1e-6
        bijection = CensoredLogisticMixtureAutoregressiveBijection(net, num_mixtures=num_mix, num_bins=num_bins, autoregressive_order='ltr')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, features))


if __name__ == '__main__':
    unittest.main()
