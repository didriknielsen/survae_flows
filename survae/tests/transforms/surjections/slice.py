import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms import Slice
from survae.distributions import StandardNormal, ConditionalMeanNormal
from survae.tests.transforms.surjections import SurjectionTest


class SliceTest(SurjectionTest):

    def test_surjection_is_well_behaved(self):
        batch_size = 10
        shape = [8, 4, 4]
        x = torch.randn(batch_size, *shape)
        surjections = [
            (Slice(StandardNormal([1, 4, 4]), num_keep=7, dim=1), (7,4,4)),
            (Slice(StandardNormal([4, 4, 4]), num_keep=4, dim=1), (4,4,4)),
            (Slice(StandardNormal([7, 4, 4]), num_keep=1, dim=1), (1,4,4)),
            (Slice(StandardNormal([8, 2, 4]), num_keep=2, dim=2), (8,2,4)),
            (Slice(StandardNormal([8, 4, 2]), num_keep=2, dim=3), (8,4,2)),
            (Slice(ConditionalMeanNormal(nn.Conv2d(7, 1, kernel_size=1)), num_keep=7, dim=1), (7,4,4)),
            (Slice(ConditionalMeanNormal(nn.Conv2d(4, 4, kernel_size=1)), num_keep=4, dim=1), (4,4,4)),
            (Slice(ConditionalMeanNormal(nn.Conv2d(1, 7, kernel_size=1)), num_keep=1, dim=1), (1,4,4))
        ]

        for surjection, new_shape in surjections:
            with self.subTest(surjection=surjection):
                self.assert_surjection_is_well_behaved(surjection, x, z_shape=(batch_size, *new_shape), z_dtype=x.dtype)


if __name__ == '__main__':
    unittest.main()
