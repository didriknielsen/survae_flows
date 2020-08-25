import numpy as np
import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms import ConditionalAdditiveBijection, ConditionalAffineBijection
from survae.tests.transforms.bijections.conditional import ConditionalBijectionTest


class ConditionalAdditiveBijectionTest(ConditionalBijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = (6,)

        x = torch.randn(batch_size, 6)
        context = torch.randn(batch_size, 3)

        self.eps = 1e-6
        bijection = ConditionalAdditiveBijection(nn.Linear(3,6))
        self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))


class ConditionalAffineBijectionTest(ConditionalBijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = (6,)

        x = torch.randn(batch_size, 6)
        context = torch.randn(batch_size, 3)

        self.eps = 1e-6
        bijection = ConditionalAffineBijection(nn.Linear(3,6*2))
        self.assert_bijection_is_well_behaved(bijection, x, context, z_shape=(batch_size, *shape))


if __name__ == '__main__':
    unittest.main()
