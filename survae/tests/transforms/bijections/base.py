import torch
import unittest
import torchtestcase


class BijectionTest(torchtestcase.TorchTestCase):
    """Base test for all bijections."""

    def assert_tensor_is_good(self, tensor, shape=None, dtype=None):
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertFalse(torch.isnan(tensor).any())
        self.assertFalse(torch.isinf(tensor).any())
        if shape is not None:
            self.assertEqual(tensor.shape, torch.Size(shape))
        if dtype is not None:
            self.assertEqual(tensor.dtype, dtype)

    def assert_bijection_is_well_behaved(self, bijection, x, z_shape=None):
        z, ldj = bijection.forward(x)
        xr = bijection.inverse(z)

        self.assert_tensor_is_good(z, shape=z_shape, dtype=x.dtype)
        self.assert_tensor_is_good(ldj, shape=(x.shape[0],))
        self.assert_tensor_is_good(xr, shape=x.shape, dtype=x.dtype)
        self.assertEqual(x, xr)
