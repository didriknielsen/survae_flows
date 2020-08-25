import torch
import torchtestcase
import unittest
from survae.transforms import UniformDequantization
from survae.tests.transforms.surjections import SurjectionTest


class UniformDequantizationTest(SurjectionTest):

    def test_surjection_is_well_behaved(self):
        batch_size = 10
        shape = [8, 4, 4]
        num_bits_list = [2, 5, 8]

        for num_bits in num_bits_list:
            with self.subTest(num_bits=num_bits):
                x = torch.randint(0, 2**num_bits, (batch_size,) + torch.Size(shape))
                surjection = UniformDequantization(num_bits=num_bits)
                self.assert_surjection_is_well_behaved(surjection, x, z_shape=(batch_size, *shape), z_dtype=torch.float)

    def test_range(self):

        z = torch.randn((100,))
        surjection = UniformDequantization(num_bits=8)
        x = surjection.inverse(z)
        self.assertTrue(x.min()>=0)
        self.assertTrue(x.max()<=255)



if __name__ == '__main__':
    unittest.main()
