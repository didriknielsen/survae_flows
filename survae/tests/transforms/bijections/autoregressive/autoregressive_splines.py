import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms.bijections.autoregressive import *
from survae.nn.nets.autoregressive import MADE
from survae.tests.transforms.bijections import BijectionTest


class LinearSplineAutoregressiveBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 4
        batch_size = 10
        features = 7
        x = torch.rand(batch_size, features)
        net = MADE(features, num_params=num_bins, hidden_features=[21])

        self.eps = 1e-3
        bijection = LinearSplineAutoregressiveBijection(net, num_bins=num_bins, autoregressive_order='ltr')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, features))


class QuadraticSplineAutoregressiveBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 4
        batch_size = 10
        features = 7
        x = torch.rand(batch_size, features)
        net = MADE(features, num_params=2*num_bins+1, hidden_features=[21])

        self.eps = 5e-3
        bijection = QuadraticSplineAutoregressiveBijection(net, num_bins=num_bins, autoregressive_order='ltr')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, features))


class CubicSplineAutoregressiveBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 4
        batch_size = 10
        features = 7
        x = torch.rand(batch_size, features)
        net = MADE(features, num_params=2*num_bins+2, hidden_features=[21])

        self.eps = 5e-3
        bijection = CubicSplineAutoregressiveBijection(net, num_bins=num_bins, autoregressive_order='ltr')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, features))


class RationalQuadraticSplineAutoregressiveBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 4
        batch_size = 10
        features = 7
        x = torch.rand(batch_size, features)
        net = MADE(features, num_params=3*num_bins+1, hidden_features=[21])

        self.eps = 1e-5
        bijection = RationalQuadraticSplineAutoregressiveBijection(net, num_bins=num_bins, autoregressive_order='ltr')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, features))


if __name__ == '__main__':
    unittest.main()
