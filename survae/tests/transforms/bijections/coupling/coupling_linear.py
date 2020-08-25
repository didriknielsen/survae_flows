import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms.bijections.coupling import *
from survae.nn.layers import ElementwiseParams, ElementwiseParams2d
from survae.tests.transforms.bijections import BijectionTest


class AdditiveCouplingBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10

        self.eps = 1e-6
        for shape in [(6,),
                      (6,8,8)]:
            for num_condition in [None, 1]:
                with self.subTest(shape=shape, num_condition=num_condition):
                    x = torch.randn(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Linear(3,3)
                        if len(shape) == 3: net = nn.Conv2d(3,3, kernel_size=3, padding=1)
                    else:
                        if len(shape) == 1: net = nn.Linear(1,5)
                        if len(shape) == 3: net = nn.Conv2d(1,5, kernel_size=3, padding=1)
                    bijection = AdditiveCouplingBijection(net, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


class AffineCouplingBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10

        self.eps = 5e-6
        for shape in [(6,),
                      (6,8,8)]:
            for num_condition in [None, 1]:
                with self.subTest(shape=shape, num_condition=num_condition):
                    x = torch.randn(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3,3*2), ElementwiseParams(2))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3,3*2, kernel_size=3, padding=1), ElementwiseParams2d(2))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1,5*2), ElementwiseParams(2))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1,5*2, kernel_size=3, padding=1), ElementwiseParams2d(2))
                    bijection = AffineCouplingBijection(net, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


if __name__ == '__main__':
    unittest.main()
