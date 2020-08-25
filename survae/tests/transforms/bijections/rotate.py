import copy
import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import Rotate
from survae.tests.transforms.bijections import BijectionTest


class RotateTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = [2, 3, 4]
        x = torch.randn(batch_size, *shape)
        bijections = [
            Rotate(90, 1, 2),
            Rotate(90, 1, 3),
            Rotate(90, 2, 3),
            Rotate(180, 1, 2),
            Rotate(180, 1, 3),
            Rotate(180, 2, 3),
            Rotate(270, 1, 2),
            Rotate(270, 1, 3),
            Rotate(270, 2, 3),
        ]

        for bijection in bijections:
            with self.subTest(bijection=bijection):
                self.assert_bijection_is_well_behaved(bijection, x, z_shape=compute_z_shape(bijection, x))

    def test_bijection_output(self):
        batch_size = 10
        shape = [2, 3, 4]
        x = torch.randn(batch_size, *shape)

        bijection = Rotate(90, 2, 3)
        z, _ = bijection(x)
        self.assertEqual(x[:,:,0,0], z[:,:,3,0])

        bijection = Rotate(180, 2, 3)
        z, _ = bijection(x)
        self.assertEqual(x[:,:,0,0], z[:,:,2,3])

        bijection = Rotate(270, 2, 3)
        z, _ = bijection(x)
        self.assertEqual(x[:,:,0,0], z[:,:,0,2])


def compute_z_shape(bijection, x):
    if bijection.degrees == 180: return torch.Size(x.shape)
    z_shape = list(x.shape)
    z_shape[bijection.dim1] = x.shape[bijection.dim2]
    z_shape[bijection.dim2] = x.shape[bijection.dim1]
    return torch.Size(z_shape)


if __name__ == '__main__':
    unittest.main()
