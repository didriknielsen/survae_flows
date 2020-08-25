import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import Permute, Reverse, Shuffle
from survae.tests.transforms.bijections import BijectionTest


class PermuteTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = [2, 3, 4]
        x = torch.randn(batch_size, *shape)
        bijections = [
            Permute(torch.tensor([0, 1]), 1),
            Permute([0, 1], 1),
            Permute([1, 0], 1),
            Permute([0, 1, 2], 2),
            Permute([2, 1, 0], 2),
            Permute([0, 1, 2, 3], 3),
            Permute([3, 2, 1, 0], 3),
            Permute([0, 2, 1], 2),
            Permute([2, 0, 1], 2),
            Permute([1, 0, 2], 2),
            Permute([1, 2, 0], 2),
            Reverse(dim_size=2, dim=1),
            Reverse(dim_size=3, dim=2),
            Reverse(dim_size=4, dim=3),
            Shuffle(dim_size=2, dim=1),
            Shuffle(dim_size=3, dim=2),
            Shuffle(dim_size=4, dim=3),
        ]

        for bijection in bijections:
            with self.subTest(bijection=bijection):
                self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))


if __name__ == '__main__':
    unittest.main()
