import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms import SimpleMaxPoolSurjection2d
from survae.distributions import StandardHalfNormal
from survae.tests.transforms.surjections import SurjectionTest


class SimpleMaxPoolSurjection2dTest(SurjectionTest):

    def test_surjection_is_well_behaved(self):
        batch_size = 10

        surjections = [
            ((1,4,4), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*1,2,2))), (1,2,2)),
            ((3,2,2), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*1,1,1))), (3,1,1)),
            ((3,4,4), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*3,2,2))), (3,2,2)),
            ((4,10,10), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*4,5,5))), (4,5,5)),
        ]

        self.eps = 5e-7
        for x_shape, surjection, z_shape in surjections:
            with self.subTest(surjection=surjection):
                x = torch.randn(batch_size, *x_shape)
                self.assert_surjection_is_well_behaved(surjection, x, z_shape=(batch_size, *z_shape), z_dtype=x.dtype)

    def test_matching_maxpool(self):
        batch_size = 10
        maxpool = nn.MaxPool2d(2)
        surjections = [
            ((1,4,4), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*1,2,2)))),
            ((3,2,2), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*1,1,1)))),
            ((3,4,4), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*3,2,2)))),
            ((4,10,10), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*4,5,5)))),
        ]

        for x_shape, surjection in surjections:
            with self.subTest(surjection=surjection):
                x = torch.randn(batch_size, *x_shape)
                z, _ = surjection.forward(x)
                zt = maxpool(x)
                self.assertEqual(z, zt)

    def test_utils(self):
        batch_size = 10

        surjections = [
            ((1,4,4), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*1,2,2))), (1,2,2)),
            ((3,2,2), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*1,1,1))), (3,1,1)),
            ((3,4,4), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*3,2,2))), (3,2,2)),
            ((4,10,10), SimpleMaxPoolSurjection2d(decoder=StandardHalfNormal((3*4,5,5))), (4,5,5)),
        ]

        self.eps = 5e-7
        for x_shape, surjection, z_shape in surjections:
            with self.subTest(surjection=surjection):
                x = torch.randn(batch_size, *x_shape)
                z, xd, k = surjection._deconstruct_x(x)
                x_ = surjection._construct_x(z, xd, k)
                self.assertEqual(x, x_)


if __name__ == '__main__':
    unittest.main()
