import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms.bijections.conditional.coupling import *
from survae.nn.layers import ElementwiseParams, ElementwiseParams2d, scale_fn
from survae.tests.transforms.bijections.conditional import ConditionalBijectionTest


class ConditionalLinearSplineCouplingBijectionTest(ConditionalBijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 16
        batch_size = 10

        self.eps = 1e-3
        for shape in [(6,),
                      (6,8,8)]:
            for num_condition in [None, 1]:
                with self.subTest(shape=shape, num_condition=num_condition):
                    x = torch.rand(batch_size, *shape)
                    context = torch.randn(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3+6,3*num_bins), ElementwiseParams(num_bins))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3+6,3*num_bins, kernel_size=3, padding=1), ElementwiseParams2d(num_bins))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1+6,5*num_bins), ElementwiseParams(num_bins))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1+6,5*num_bins, kernel_size=3, padding=1), ElementwiseParams2d(num_bins))
                    bijection = ConditionalLinearSplineCouplingBijection(net, num_bins=num_bins, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x, context=context)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


class ConditionalQuadraticSplineCouplingBijectionTest(ConditionalBijectionTest):

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
                    context = torch.randn(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3+6,3*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3+6,3*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1+6,5*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1+6,5*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    bijection = ConditionalQuadraticSplineCouplingBijection(net, num_bins=num_bins, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x, context=context)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


class ConditionalCubicSplineCouplingBijectionTest(ConditionalBijectionTest):

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
                    context = torch.randn(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3+6,3*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3+6,3*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1+6,5*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1+6,5*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    bijection = ConditionalCubicSplineCouplingBijection(net, num_bins=num_bins, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x, context=context)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


class ConditionalRationalQuadraticSplineCouplingBijectionTest(ConditionalBijectionTest):

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
                    context = torch.randn(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3+6,3*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3+6,3*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1+6,5*num_params), ElementwiseParams(num_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1+6,5*num_params, kernel_size=3, padding=1), ElementwiseParams2d(num_params))
                    bijection = ConditionalRationalQuadraticSplineCouplingBijection(net, num_bins=num_bins, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x, context=context)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


if __name__ == '__main__':
    unittest.main()
