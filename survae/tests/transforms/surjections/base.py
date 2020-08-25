import torch
import unittest
import torchtestcase


class SurjectionTest(torchtestcase.TorchTestCase):
    """Base test for all surjections."""

    def assert_tensor_is_good(self, tensor, shape=None, dtype=None):
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertFalse(torch.isnan(tensor).any())
        self.assertFalse(torch.isinf(tensor).any())
        if shape is not None:
            self.assertEqual(tensor.shape, torch.Size(shape))
        if dtype is not None:
            self.assertEqual(tensor.dtype, dtype)

    def assert_surjection_is_well_behaved(self, surjection, x, z_shape=None, z_dtype=None):

        if surjection.stochastic_forward:

            z, ldj = surjection.forward(x)
            xr = surjection.inverse(z)

            self.assertEqual(x, xr)

            self.assert_tensor_is_good(z, shape=z_shape, dtype=z_dtype)
            self.assert_tensor_is_good(ldj, shape=(x.shape[0],))
            self.assert_tensor_is_good(xr, shape=x.shape, dtype=x.dtype)

        else:

            z, ldj = surjection.forward(x)
            xr = surjection.inverse(z)
            zr, ldjr = surjection.forward(xr)

            self.assertEqual(z, zr)

            self.assert_tensor_is_good(z, shape=z_shape, dtype=z_dtype)
            self.assert_tensor_is_good(ldj, shape=(x.shape[0],))
            self.assert_tensor_is_good(xr, shape=x.shape, dtype=x.dtype)
