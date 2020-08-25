import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import Reshape
from survae.tests.transforms.bijections import BijectionTest


class ReshapeTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = [2, 3, 4]
        x = torch.randn(batch_size, *shape)
        bijections = [
            Reshape(shape, [2*3*4]),
            Reshape(shape, [2, 3*4]),
            Reshape(shape, [2*3, 4]),
            Reshape(shape, [2, 3, 4, 1]),
            Reshape(shape, [1, 2, 3, 4]),
            Reshape(shape, [2*3*2, 2]),
        ]

        for bijection in bijections:
            with self.subTest(bijection=bijection):
                self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *bijection.output_shape))


if __name__ == '__main__':
    unittest.main()
