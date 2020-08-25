import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import ActNormBijection, ActNormBijection1d, ActNormBijection2d
from survae.tests.transforms.bijections import BijectionTest


class ActNormBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 50

        self.eps = 1e-6
        for shape in [(32,),
                      (32,8),
                      (32,8,8)]:
            with self.subTest(shape=shape):
                x = torch.randn(batch_size, *shape)
                if len(shape) == 1:   bijection = ActNormBijection(shape[0])
                elif len(shape) == 2: bijection = ActNormBijection1d(shape[0])
                elif len(shape) == 3: bijection = ActNormBijection2d(shape[0])
                self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

    def test_data_dep_init(self):
        batch_size = 50
        shape = (32,)
        x = torch.randn(batch_size, *shape)
        bijection = ActNormBijection(shape[0])
        self.assertEqual(bijection.initialized, torch.zeros(1))
        z, ldj = bijection(x)
        self.assertEqual(bijection.initialized, torch.ones(1))

        self.eps = 1e-5
        self.assertEqual(z.mean(0), torch.zeros(z.shape[1]))
        self.assertEqual(z.std(0), torch.ones(z.shape[1]))




if __name__ == '__main__':
    unittest.main()
