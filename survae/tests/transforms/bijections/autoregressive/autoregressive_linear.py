import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms.bijections.autoregressive import *
from survae.nn.nets.autoregressive import MADE
from survae.tests.transforms.bijections import BijectionTest


class AdditiveAutoregressiveBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        features = 7
        x = torch.randn(batch_size, features)
        net = MADE(features, num_params=1, hidden_features=[21])

        self.eps = 1e-6
        bijection = AdditiveAutoregressiveBijection(net, autoregressive_order='ltr')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, features))


class AffineAutoregressiveBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        features = 7
        x = torch.randn(batch_size, features)
        net = MADE(features, num_params=2, hidden_features=[21])

        self.eps = 1e-6
        bijection = AffineAutoregressiveBijection(net, autoregressive_order='ltr')
        self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, features))


if __name__ == '__main__':
    unittest.main()
