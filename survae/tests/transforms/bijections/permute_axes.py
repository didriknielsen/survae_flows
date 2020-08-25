import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import PermuteAxes
from survae.tests.transforms.bijections import BijectionTest


class PermuteAxesTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = [2, 3, 4]
        x = torch.randn(batch_size, *shape)
        bijections = [
            (PermuteAxes([0,1,2,3]), (2,3,4)),
            (PermuteAxes([0,1,3,2]), (2,4,3)),
            (PermuteAxes([0,2,1,3]), (3,2,4)),
            (PermuteAxes([0,2,3,1]), (3,4,2)),
            (PermuteAxes([0,3,1,2]), (4,2,3)),
            (PermuteAxes([0,3,2,1]), (4,3,2)),
        ]

        for bijection, new_shape in bijections:
            with self.subTest(bijection=bijection):
                self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *new_shape))


if __name__ == '__main__':
    unittest.main()
