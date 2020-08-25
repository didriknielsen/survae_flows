import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms import Augment
from survae.distributions import StandardNormal, ConditionalMeanNormal
from survae.tests.transforms.surjections import SurjectionTest


class AugmentTest(SurjectionTest):

    def test_surjection_is_well_behaved(self):
        batch_size = 10
        shape = [8, 4, 4]
        x = torch.randn(batch_size, *shape)
        surjections = [
            (Augment(StandardNormal([1, 4, 4]), x_size=8, split_dim=1), (9,4,4)),
            (Augment(StandardNormal([4, 4, 4]), x_size=8, split_dim=1), (12,4,4)),
            (Augment(StandardNormal([7, 4, 4]), x_size=8, split_dim=1), (15,4,4)),
            (Augment(StandardNormal([8, 2, 4]), x_size=4, split_dim=2), (8,6,4)),
            (Augment(StandardNormal([8, 4, 2]), x_size=4, split_dim=3), (8,4,6)),
            (Augment(ConditionalMeanNormal(nn.Conv2d(8, 1, kernel_size=1)), x_size=8, split_dim=1), (9,4,4)),
            (Augment(ConditionalMeanNormal(nn.Conv2d(8, 4, kernel_size=1)), x_size=8, split_dim=1), (12,4,4)),
            (Augment(ConditionalMeanNormal(nn.Conv2d(8, 7, kernel_size=1)), x_size=8, split_dim=1), (15,4,4))
        ]

        for surjection, new_shape in surjections:
            with self.subTest(surjection=surjection):
                self.assert_surjection_is_well_behaved(surjection, x, z_shape=(batch_size, *new_shape), z_dtype=x.dtype)


if __name__ == '__main__':
    unittest.main()
