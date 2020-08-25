import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms.bijections.coupling import *
from survae.nn.layers import ElementwiseParams, ElementwiseParams2d
from survae.tests.transforms.bijections import BijectionTest


class LinearSplineCouplingBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 16
        batch_size = 10

        self.eps = 1e-3
        for shape in [(6,),
                      (6,8,8)]:
            for num_condition in [None, 1]:
                with self.subTest(shape=shape, num_condition=num_condition):
                    x = torch.rand(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3,3*num_bins), ElementwiseParams(num_bins))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3,3*num_bins, kernel_size=3, padding=1), ElementwiseParams2d(num_bins))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1,5*num_bins), ElementwiseParams(num_bins))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1,5*num_bins, kernel_size=3, padding=1), ElementwiseParams2d(num_bins))
                    bijection = LinearSplineCouplingBijection(net, num_bins=num_bins, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


class QuadraticSplineCouplingBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 16
        batch_size = 10

        num_params = 2 * num_bins + 1

        self.eps = 5e-3
        for shape in [(6,),
                      (6,8,8)]:
            for num_condition in [None, 1]:
                with self.subTest(shape=shape, num_condition=num_condition):
                    x = torch.rand(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3,3*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3,3*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1,5*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1,5*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    bijection = QuadraticSplineCouplingBijection(net, num_bins=num_bins, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


class CubicSplineCouplingBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 16
        batch_size = 10

        num_params = 2 * num_bins + 2

        self.eps = 5e-3
        for shape in [(6,),
                      (6,8,8)]:
            for num_condition in [None, 1]:
                with self.subTest(shape=shape, num_condition=num_condition):
                    x = torch.rand(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3,3*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3,3*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1,5*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1,5*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    bijection = CubicSplineCouplingBijection(net, num_bins=num_bins, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


class RationalQuadraticSplineCouplingBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 16
        batch_size = 10

        num_params = 3 * num_bins + 1

        self.eps = 1e-5
        for shape in [(6,),
                      (6,8,8)]:
            for num_condition in [None, 1]:
                with self.subTest(shape=shape, num_condition=num_condition):
                    x = torch.rand(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3,3*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3,3*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1,5*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1,5*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    bijection = RationalQuadraticSplineCouplingBijection(net, num_bins=num_bins, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


if __name__ == '__main__':
    unittest.main()
