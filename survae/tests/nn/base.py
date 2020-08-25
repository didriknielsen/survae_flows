import torch
import unittest
import torchtestcase


class ModuleTest(torchtestcase.TorchTestCase):
    """Base test for all nn modules."""

    def assert_tensor_is_good(self, tensor, shape=None):
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertTrue(tensor.dtype.is_floating_point)
        self.assertFalse(torch.isnan(tensor).any())
        self.assertFalse(torch.isinf(tensor).any())
        if shape is not None:
            self.assertEqual(tensor.shape, torch.Size(shape))

    def assert_layer_is_well_behaved(self, module, x):
        y = module(x)
        self.assert_tensor_is_good(y)
