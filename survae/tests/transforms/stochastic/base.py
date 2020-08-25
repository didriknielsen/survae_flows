import torch
import unittest
import torchtestcase


class StochasticTransformTest(torchtestcase.TorchTestCase):
    """Base test for all stochastic transforms."""

    def assert_tensor_is_good(self, tensor, shape=None, dtype=None):
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertFalse(torch.isnan(tensor).any())
        self.assertFalse(torch.isinf(tensor).any())
        if shape is not None:
            self.assertEqual(tensor.shape, torch.Size(shape))
        if dtype is not None:
            self.assertEqual(tensor.dtype, dtype)

    def assert_stochastic_transform_is_well_behaved(self, transform, x, z_shape=None, z_dtype=None):

        z, ldj = transform.forward(x)
        xr = transform.inverse(z)

        self.assert_tensor_is_good(z, shape=z_shape, dtype=z_dtype)
        self.assert_tensor_is_good(ldj, shape=(x.shape[0],))
        self.assert_tensor_is_good(xr, shape=x.shape, dtype=x.dtype)
