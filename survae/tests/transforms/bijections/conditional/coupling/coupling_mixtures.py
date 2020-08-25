import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms.bijections.conditional.coupling import *
from survae.nn.layers import ElementwiseParams, ElementwiseParams2d, scale_fn
from survae.tests.transforms.bijections.conditional import ConditionalBijectionTest


class ConditionalGaussianMixtureCouplingBijectionTest(ConditionalBijectionTest):

    def test_bijection_is_well_behaved(self):
        num_mix = 8
        batch_size = 10
        elementwise_params = 3 * num_mix

        self.eps = 5e-5
        for shape in [(6,),
                      (6,4,4)]:
            for num_condition in [None, 1]:
                with self.subTest(shape=shape, num_condition=num_condition):
                    x = torch.randn(batch_size, *shape)
                    context = torch.randn(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3+6,3*elementwise_params), ElementwiseParams(elementwise_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3+6,3*elementwise_params, kernel_size=3, padding=1), ElementwiseParams2d(elementwise_params))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1+6,5*elementwise_params), ElementwiseParams(elementwise_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1+6,5*elementwise_params, kernel_size=3, padding=1), ElementwiseParams2d(elementwise_params))
                    bijection = ConditionalGaussianMixtureCouplingBijection(net, num_mixtures=num_mix, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x, context=context)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


class ConditionalLogisticMixtureCouplingBijectionTest(ConditionalBijectionTest):

    def test_bijection_is_well_behaved(self):
        num_mix = 8
        batch_size = 10
        elementwise_params = 3 * num_mix

        self.eps = 5e-5
        for shape in [(6,),
                      (6,4,4)]:
            for num_condition in [None, 1]:
                with self.subTest(shape=shape, num_condition=num_condition):
                    x = torch.randn(batch_size, *shape)
                    context = torch.randn(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3+6,3*elementwise_params), ElementwiseParams(elementwise_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3+6,3*elementwise_params, kernel_size=3, padding=1), ElementwiseParams2d(elementwise_params))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1+6,5*elementwise_params), ElementwiseParams(elementwise_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1+6,5*elementwise_params, kernel_size=3, padding=1), ElementwiseParams2d(elementwise_params))
                    bijection = ConditionalLogisticMixtureCouplingBijection(net, num_mixtures=num_mix, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x, context=context)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


class ConditionalCensoredLogisticMixtureCouplingBijectionTest(ConditionalBijectionTest):

    def test_bijection_is_well_behaved(self):
        num_bins = 16
        num_mix = 8
        batch_size = 10
        elementwise_params = 3 * num_mix

        self.eps = 1e-6
        for shape in [(6,),
                      (6,4,4)]:
            for num_condition in [None, 1]:
                with self.subTest(shape=shape, num_condition=num_condition):
                    x = torch.rand(batch_size, *shape)
                    context = torch.randn(batch_size, *shape)
                    if num_condition is None:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(3+6,3*elementwise_params), ElementwiseParams(elementwise_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(3+6,3*elementwise_params, kernel_size=3, padding=1), ElementwiseParams2d(elementwise_params))
                    else:
                        if len(shape) == 1: net = nn.Sequential(nn.Linear(1+6,5*elementwise_params), ElementwiseParams(elementwise_params))
                        if len(shape) == 3: net = nn.Sequential(nn.Conv2d(1+6,5*elementwise_params, kernel_size=3, padding=1), ElementwiseParams2d(elementwise_params))
                    bijection = ConditionalCensoredLogisticMixtureCouplingBijection(net, num_mixtures=num_mix, num_bins=num_bins, num_condition=num_condition)
                    self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))
                    z, _ = bijection.forward(x, context=context)
                    if num_condition is None:
                        self.assertEqual(x[:,:3], z[:,:3])
                    else:
                        self.assertEqual(x[:,:1], z[:,:1])


if __name__ == '__main__':
    unittest.main()
