import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import LinearLU
from survae.tests.transforms.bijections import BijectionTest


class LinearLUTest(BijectionTest):

    def setUp(self):
        self.features = 3
        self.bijection = LinearLU(num_features=self.features, bias=True)

        L, U = self.bijection._create_lower_upper()
        self.weight = L @ U
        self.weight_inverse = torch.inverse(self.weight)
        _, self.logabsdet = torch.slogdet(self.weight)

        self.eps = 1e-5

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        x = torch.rand(batch_size, self.features)
        bijections = [
            LinearLU(3, identity_init=True, bias=True),
            LinearLU(3, identity_init=True, bias=False),
            LinearLU(3, identity_init=False, bias=True),
            LinearLU(3, identity_init=False, bias=False),
        ]

        for bijection in bijections:
            with self.subTest(bijection=bijection):
                self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, self.features))

    def test_forward(self):
        batch_size = 10
        x = torch.randn(batch_size, self.features)
        z, ldj = self.bijection.forward(x)

        z_ref = x @ self.weight.t()
        ldj_ref = torch.full([batch_size], self.logabsdet.item(), dtype=torch.float)

        self.assertEqual(z, z_ref)
        self.assertEqual(ldj, ldj_ref)

    def test_inverse(self):
        batch_size = 10
        z = torch.randn(batch_size, self.features)
        x = self.bijection.inverse(z)

        x_ref = (z - self.bijection.bias) @ self.weight_inverse.t()

        self.assertEqual(x, x_ref)

    def test_weight(self):
        weight = self.bijection.weight()
        self.assert_tensor_is_good(weight, [self.features, self.features])
        self.assertEqual(weight, self.weight)

    def test_weight_inverse(self):
        weight_inverse = self.bijection.weight_inverse()
        self.assert_tensor_is_good(weight_inverse, [self.features, self.features])
        self.assertEqual(weight_inverse, self.weight_inverse)


if __name__ == '__main__':
    unittest.main()
