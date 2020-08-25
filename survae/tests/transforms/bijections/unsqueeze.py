import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import Unsqueeze2d
from survae.tests.transforms.bijections import BijectionTest


class Unsqueeze2dTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        setups = [
            (2, [1, 2, 2], [4, 1, 1]),
            (2, [1, 4, 4], [4, 2, 2]),
            (2, [2, 4, 4], [8, 2, 2]),
            (2, [3, 4, 4], [12, 2, 2]),
            (2, [1, 2, 2], [4, 1, 1]),
            (2, [1, 4, 4], [4, 2, 2]),
            (2, [2, 4, 4], [8, 2, 2]),
            (2, [3, 4, 4], [12, 2, 2]),
            (3, [1, 3, 3], [9, 1, 1]),
            (3, [1, 9, 9], [9, 3, 3]),
            (3, [2, 9, 9], [18, 3, 3]),
            (3, [3, 9, 9], [27, 3, 3]),
            (3, [1, 3, 3], [9, 1, 1]),
            (3, [1, 9, 9], [9, 3, 3]),
            (3, [2, 9, 9], [18, 3, 3]),
            (3, [3, 9, 9], [27, 3, 3]),
        ]

        for ordered in (False, True):
            for factor, expected_z_shape, x_shape in setups:
                with self.subTest(factor=factor, ordered=ordered, x_shape=x_shape, expected_z_shape=expected_z_shape):
                    x = torch.randn(batch_size, *x_shape)
                    bijection = Unsqueeze2d(factor, ordered=ordered)
                    self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *expected_z_shape))


if __name__ == '__main__':
    unittest.main()
