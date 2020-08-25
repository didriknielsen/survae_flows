import numpy as np
import torch
import torch.nn.functional as F
import torchtestcase
import unittest
from survae.transforms import LeakyReLU, SneakyReLU, Tanh, Sigmoid, Logit, Softplus, SoftplusInverse
from survae.tests.transforms.bijections import BijectionTest


class ElementwiseNonlinearTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = [2, 3, 4]
        x_normal = torch.randn(batch_size, *shape)
        x_uniform = torch.rand(batch_size, *shape)
        bijections = [
            (LeakyReLU(), x_normal, 1e-6),
            (SneakyReLU(), x_normal, 1e-6),
            (Tanh(), x_normal, 1e-4),
            (Sigmoid(), x_normal, 1e-5),
            (Logit(), x_uniform, 1e-6),
            (Softplus(), x_normal, 1e-6),
            (SoftplusInverse(), F.softplus(x_normal), 1e-6),
        ]


        for bijection, x, eps in bijections:
            with self.subTest(bijection=bijection):
                self.eps = eps
                self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

    def test_pairs(self):
        batch_size = 10
        shape = [2, 3, 4]
        x_normal = torch.randn(batch_size, *shape)
        x_uniform = torch.rand(batch_size, *shape)
        bijections = [
            (Sigmoid(), Logit(), x_normal, 1e-5),
            (Softplus(), SoftplusInverse(), x_normal, 1e-5),
        ]


        for bijection, bijection_inv, x, eps in bijections:
            with self.subTest(bijection=bijection):
                self.eps = eps
                z, ldj = bijection(x)
                xr, ldjr = bijection_inv(z)
                self.assertEqual(x, xr)
                self.assertEqual(ldj, -ldjr)



if __name__ == '__main__':
    unittest.main()
