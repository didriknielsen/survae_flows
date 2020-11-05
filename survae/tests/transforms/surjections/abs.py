import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms import SimpleAbsSurjection
from survae.tests.transforms.surjections import SurjectionTest


class SimpleAbsSurjectionTest(SurjectionTest):

    def test_surjection_is_well_behaved(self):
        """
        Test if well formed well formed in - place.

        Args:
            self: (todo): write your description
        """
        batch_size = 10
        x_shape = (20,)
        x = torch.randn(batch_size, *x_shape)
        surjection = SimpleAbsSurjection()

        self.assert_surjection_is_well_behaved(surjection, x, z_shape=(batch_size, *x_shape), z_dtype=x.dtype)


if __name__ == '__main__':
    unittest.main()
