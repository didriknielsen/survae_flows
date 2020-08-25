import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms import SimpleSortSurjection
from survae.tests.transforms.surjections import SurjectionTest


class SimpleSortTest(SurjectionTest):

    def setUp(self):
        batch_size = 8
        data_shape = (12,4,4)

        self.x = torch.rand(batch_size, *data_shape)
        self.transform1 = SimpleSortSurjection(1, lambd=lambda x: x[:,:,0,0])
        self.transform2 = SimpleSortSurjection(2, lambd=lambda x: x[:,0,:,0])
        self.transform3 = SimpleSortSurjection(3, lambd=lambda x: x[:,0,0,:])

    def test_surjection_is_well_behaved(self):
        self.assert_surjection_is_well_behaved(self.transform1, self.x, z_shape=self.x.shape, z_dtype=self.x.dtype)
        self.assert_surjection_is_well_behaved(self.transform2, self.x, z_shape=self.x.shape, z_dtype=self.x.dtype)
        self.assert_surjection_is_well_behaved(self.transform3, self.x, z_shape=self.x.shape, z_dtype=self.x.dtype)

    def test_order_on_tensor_extracted_from_lambd(self):
        self.assertEqual(self.x[:,:,0,0].sort(1).values, self.transform1(self.x)[0][:,:,0,0])
        self.assertEqual(self.x[:,0,:,0].sort(1).values, self.transform2(self.x)[0][:,0,:,0])
        self.assertEqual(self.x[:,0,0,:].sort(1).values, self.transform3(self.x)[0][:,0,0,:])


if __name__ == '__main__':
    unittest.main()
