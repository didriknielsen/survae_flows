import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import Conv1x1
from survae.tests.transforms.bijections import BijectionTest


class Conv1x1Test(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = [3, 32, 32]
        x = torch.rand(batch_size, *shape)
        bijections = [
            Conv1x1(3, orthogonal_init=True),
            Conv1x1(3, orthogonal_init=False),
        ]

        self.eps = 1e-4
        for bijection in bijections:
            with self.subTest(bijection=bijection):
                self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))


if __name__ == '__main__':
    unittest.main()
