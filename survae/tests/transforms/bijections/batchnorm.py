import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import BatchNormBijection, BatchNormBijection1d, BatchNormBijection2d
from survae.tests.transforms.bijections import BijectionTest


class BatchNormBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 50

        self.eps = 1e-6
        for shape in [(32,),
                      (32,8),
                      (32,8,8)]:
            with self.subTest(shape=shape):
                x = torch.randn(batch_size, *shape)
                if len(shape) == 1:   bijection = BatchNormBijection(shape[0])
                elif len(shape) == 2: bijection = BatchNormBijection1d(shape[0])
                elif len(shape) == 3: bijection = BatchNormBijection2d(shape[0])
                bijection = bijection.eval()
                self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

    def test_normalization(self):
        batch_size = 50
        shape = (32,)
        x = torch.randn(batch_size, *shape)
        bijection = BatchNormBijection(shape[0])
        z, ldj = bijection(x)
        self.eps = 1e-5
        self.assertEqual(z.mean(0), torch.zeros(z.shape[1]))
        self.assertEqual(z.std(0), torch.ones(z.shape[1]))




if __name__ == '__main__':
    unittest.main()
