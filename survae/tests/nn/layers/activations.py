import torch
import torchtestcase
import unittest
from survae.tests.nn import ModuleTest
from survae.nn.layers import GELU, Swish, ConcatReLU, ConcatELU, GatedTanhUnit


class GELUTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        batch_size = 10
        shape = (6,)
        x = torch.randn(batch_size, *shape)

        module = GELU()
        self.assert_layer_is_well_behaved(module, x)


class SwishTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        batch_size = 10
        shape = (6,)
        x = torch.randn(batch_size, *shape)

        module = Swish()
        self.assert_layer_is_well_behaved(module, x)


class ConcatReLUTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        batch_size = 10
        shape = (6,)
        x = torch.randn(batch_size, *shape)

        module = ConcatReLU()
        self.assert_layer_is_well_behaved(module, x)

        y = module(x)
        expected_shape = (batch_size, 12)
        self.assertEqual(y.shape, expected_shape)


class ConcatELUTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        batch_size = 10
        shape = (6,)
        x = torch.randn(batch_size, *shape)

        module = ConcatELU()
        self.assert_layer_is_well_behaved(module, x)

        y = module(x)
        expected_shape = (batch_size, 12)
        self.assertEqual(y.shape, expected_shape)


class GatedTanhUnitTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        batch_size = 10
        shape = (6,)
        x = torch.randn(batch_size, *shape)

        module = GatedTanhUnit()
        self.assert_layer_is_well_behaved(module, x)

        y = module(x)
        expected_shape = (batch_size, 3)
        self.assertEqual(y.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
