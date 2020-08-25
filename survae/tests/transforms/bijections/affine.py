import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import ScalarAffineBijection
from survae.tests.transforms.bijections import BijectionTest


class ScalarAffineBijectionTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        shape = [2, 3, 4]
        x = torch.randn(batch_size, *shape)

        def test_case(scale, shift):
            bijection = ScalarAffineBijection(scale=scale, shift=shift)
            self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *shape))

        self.eps = 1e-6
        test_case(None, 2.)
        test_case(2., None)
        test_case(2., 2.)

    def test_forward(self):
        batch_size = 10
        shape = [2, 3, 4]
        x = torch.randn(batch_size, *shape)

        def test_case(scale, shift, true_z, true_ldj):
            with self.subTest(scale=scale, shift=shift):
                bijection = ScalarAffineBijection(scale=scale, shift=shift)
                z, ldj = bijection.forward(x)
                self.assertEqual(z, true_z)
                self.assertEqual(ldj, torch.full([batch_size], true_ldj * np.prod(shape), dtype=torch.float))

        self.eps = 1e-6
        test_case(None, 2., x + 2., 0)
        test_case(2., None, x * 2., np.log(2.))
        test_case(2., 2., x * 2. + 2., np.log(2.))

    def test_inverse(self):
        batch_size = 10
        shape = [2, 3, 4]
        z = torch.randn(batch_size, *shape)

        def test_case(scale, shift, true_x):
            with self.subTest(scale=scale, shift=shift):
                bijection = ScalarAffineBijection(scale=scale, shift=shift)
                x = bijection.inverse(z)
                self.assertEqual(x, true_x)

        self.eps = 1e-6
        test_case(None, 2., z - 2.)
        test_case(2., None, z / 2.)
        test_case(2., 2., (z - 2.) / 2.)


if __name__ == '__main__':
    unittest.main()
