import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import Linear
from survae.tests.transforms.bijections import BijectionTest


class LinearTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = [3]
        x = torch.rand(batch_size, *shape)
        bijections = [
            Linear(3, orthogonal_init=True, bias=True),
            Linear(3, orthogonal_init=True, bias=False),
            Linear(3, orthogonal_init=False, bias=True),
            Linear(3, orthogonal_init=False, bias=False),
        ]

        self.eps = 1e-5
        for bijection in bijections:
            with self.subTest(bijection=bijection):
                self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))


if __name__ == '__main__':
    unittest.main()
