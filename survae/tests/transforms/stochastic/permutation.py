import torch
import torchtestcase
import unittest
import torch.nn as nn
from survae.transforms import StochasticPermutation
from survae.tests.transforms.stochastic import StochasticTransformTest


class StochasticPermutationTest(StochasticTransformTest):

    def setUp(self):
        batch_size = 8
        data_shape = (12,4,4)

        self.x = torch.randn(batch_size, *data_shape)
        self.transform1 = StochasticPermutation(1)
        self.transform2 = StochasticPermutation(2)
        self.transform3 = StochasticPermutation(3)

    def test_stochastic_transform_is_well_behaved(self):
        self.assert_stochastic_transform_is_well_behaved(self.transform1, self.x, z_shape=self.x.shape, z_dtype=self.x.dtype)
        self.assert_stochastic_transform_is_well_behaved(self.transform2, self.x, z_shape=self.x.shape, z_dtype=self.x.dtype)
        self.assert_stochastic_transform_is_well_behaved(self.transform3, self.x, z_shape=self.x.shape, z_dtype=self.x.dtype)

    def test_same_elements(self):
        self.assertEqual(self.x.sort(1).values, self.transform1(self.x)[0].sort(1).values)
        self.assertEqual(self.x.sort(2).values, self.transform2(self.x)[0].sort(2).values)
        self.assertEqual(self.x.sort(3).values, self.transform3(self.x)[0].sort(3).values)


if __name__ == '__main__':
    unittest.main()
