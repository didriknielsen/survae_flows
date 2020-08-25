import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.transforms import VariationalDequantization
from survae.nn.layers import LambdaLayer
from survae.flows import ConditionalInverseFlow
from survae.distributions import DiagonalNormal
from survae.transforms import ConditionalAffineBijection, Sigmoid
from survae.tests.transforms.surjections import SurjectionTest


class VariationalDequantizationTest(SurjectionTest):

    def test_surjection_is_well_behaved(self):
        batch_size = 10
        shape = [8, 4, 4]
        num_bits_list = [2, 5, 8]

        for num_bits in num_bits_list:
            with self.subTest(num_bits=num_bits):
                x = torch.randint(0, 2**num_bits, (batch_size,) + torch.Size(shape))
                encoder = ConditionalInverseFlow(base_dist=DiagonalNormal(shape),
                                                 transforms=[
                                                    ConditionalAffineBijection(nn.Sequential(LambdaLayer(lambda x: 2*x.float()/(2**num_bits-1)-1),
                                                                                             nn.Conv2d(shape[0], 2*shape[0], kernel_size=3, padding=1))),
                                                    Sigmoid()
                                                 ])
                surjection = VariationalDequantization(encoder, num_bits=num_bits)
                self.assert_surjection_is_well_behaved(surjection, x, z_shape=(batch_size, *shape), z_dtype=torch.float)

    def test_range(self):
        batch_size = 10
        shape = [8]

        z = torch.randn(batch_size, *shape)
        encoder = ConditionalInverseFlow(base_dist=DiagonalNormal(shape),
                                         transforms=[
                                            ConditionalAffineBijection(nn.Sequential(LambdaLayer(lambda x: 2*x.float()/255-1),
                                                                                     nn.Linear(shape[0], 2*shape[0]))),
                                            Sigmoid()
                                         ])
        surjection = VariationalDequantization(encoder, num_bits=8)
        x = surjection.inverse(z)
        self.assertTrue(x.min()>=0)
        self.assertTrue(x.max()<=255)



if __name__ == '__main__':
    unittest.main()
