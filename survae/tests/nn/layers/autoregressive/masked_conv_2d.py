import torch
import torchtestcase
import unittest
import copy
from survae.tests.nn import ModuleTest
from survae.nn.layers.autoregressive import SpatialMaskedConv2d, MaskedConv2d


class SpatialMaskedConv2dTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        batch_size = 10
        shape = (2,8,8)
        x = torch.randn(batch_size, *shape)

        module = SpatialMaskedConv2d(2,2, kernel_size=3, padding=1, mask_type='A')
        self.assert_layer_is_well_behaved(module, x)

        module = SpatialMaskedConv2d(2,2, kernel_size=3, padding=1, mask_type='B')
        self.assert_layer_is_well_behaved(module, x)

    def test_autoregressive_type_A(self):
        batch_size = 10
        shape = (2,8,8)
        x = torch.randn(batch_size, *shape)
        x_altered = copy.deepcopy(x)
        x_altered[:,1,4,2] += 100.0 # Alter feature 2/2 in position (4,2)

        module = SpatialMaskedConv2d(2,2, kernel_size=3, padding=1, mask_type='A')
        y = module(x)
        y_altered = module(x_altered)

        # Assert every element up to and including (4,2) is unaltered
        self.assertEqual(y[:,:,:4], y_altered[:,:,:4])
        self.assertEqual(y[:,:,4,:3], y_altered[:,:,4,:3])

        # Assert element (4,3) is altered
        self.assertFalse((y[:,:,4,3] == y_altered[:,:,4,3]).view(-1).any())

    def test_autoregressive_type_B(self):
        batch_size = 10
        shape = (2,8,8)
        x = torch.randn(batch_size, *shape)
        x_altered = copy.deepcopy(x)
        x_altered[:,1,4,2] += 100.0 # Alter feature 2/2 in position (4,2)

        module = SpatialMaskedConv2d(2,2, kernel_size=3, padding=1, mask_type='B')
        y = module(x)
        y_altered = module(x_altered)

        # Assert every element up to (but not including) (4,2) is unaltered
        self.assertEqual(y[:,:,:4], y_altered[:,:,:4])
        self.assertEqual(y[:,:,4,:2], y_altered[:,:,4,:2])

        # Assert element (4,2) is altered
        self.assertFalse((y[:,:,4,2] == y_altered[:,:,4,2]).view(-1).any())


class MaskedConv2dTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        batch_size = 10
        shape = (6,8,8)
        x = torch.randn(batch_size, *shape)

        module = MaskedConv2d(6,6, kernel_size=3, padding=1, mask_type='A')
        self.assert_layer_is_well_behaved(module, x)

        module = MaskedConv2d(6,6, kernel_size=3, padding=1, mask_type='B')
        self.assert_layer_is_well_behaved(module, x)

    def test_autoregressive_type_A(self):
        batch_size = 10
        shape = (6,8,8)
        x = torch.randn(batch_size, *shape)
        x_altered = copy.deepcopy(x)
        x_altered[:,4,4,2] += 100.0 # Alter channel G of feature 2/2 in position (4,2)

        module = MaskedConv2d(6,6, kernel_size=3, padding=1, mask_type='A')
        y = module(x)
        y_altered = module(x_altered)

        # Assert all pixels up to (4,2) are unaltered
        self.assertEqual(y[:,:,:4], y_altered[:,:,:4])
        self.assertEqual(y[:,:,4,:2], y_altered[:,:,4,:2])

        # Assert channel R is unaltered
        self.assertEqual(y[:,0,4,2], y_altered[:,0,4,2])
        self.assertEqual(y[:,3,4,2], y_altered[:,3,4,2])

        # Assert channel G is unaltered
        self.assertEqual(y[:,1,4,2], y_altered[:,1,4,2])
        self.assertEqual(y[:,4,4,2], y_altered[:,4,4,2])

        # Assert channel B is altered
        self.assertFalse((y[:,2,4,2] == y_altered[:,2,4,2]).view(-1).any())
        self.assertFalse((y[:,5,4,2] == y_altered[:,5,4,2]).view(-1).any())

        # Assert all elements in next pixel are altered
        self.assertFalse((y[:,:,4,3] == y_altered[:,:,4,3]).view(-1).any())

    def test_autoregressive_type_B(self):
        batch_size = 10
        shape = (6,8,8)
        x = torch.randn(batch_size, *shape)
        x_altered = copy.deepcopy(x)
        x_altered[:,4,4,2] += 100.0 # Alter channel G of feature 2/2 in position (4,2)

        module = MaskedConv2d(6,6, kernel_size=3, padding=1, mask_type='B')
        y = module(x)
        y_altered = module(x_altered)

        # Assert all pixels up to (:,4,2) are unaltered
        self.assertEqual(y[:,:,:4], y_altered[:,:,:4])
        self.assertEqual(y[:,:,4,:2], y_altered[:,:,4,:2])

        # Assert channel R is unaltered
        self.assertEqual(y[:,0,4,2], y_altered[:,0,4,2])
        self.assertEqual(y[:,3,4,2], y_altered[:,3,4,2])

        # Assert channel G is altered
        self.assertFalse((y[:,1,4,2] == y_altered[:,1,4,2]).view(-1).any())
        self.assertFalse((y[:,4,4,2] == y_altered[:,4,4,2]).view(-1).any())

        # Assert channel B is altered
        self.assertFalse((y[:,2,4,2] == y_altered[:,2,4,2]).view(-1).any())
        self.assertFalse((y[:,5,4,2] == y_altered[:,5,4,2]).view(-1).any())

        # Assert all elements in next pixel are altered
        self.assertFalse((y[:,:,4,3] == y_altered[:,:,4,3]).view(-1).any())


if __name__ == '__main__':
    unittest.main()
