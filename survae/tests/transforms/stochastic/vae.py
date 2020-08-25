import torch
import torchtestcase
import unittest
import torch.nn as nn
from survae.transforms import VAE
from survae.distributions import ConditionalNormal
from survae.tests.transforms.stochastic import StochasticTransformTest


class VAETest(StochasticTransformTest):

    def test_stochastic_transform_is_well_behaved(self):
        batch_size = 8
        data_size = 10
        latent_size = 2

        x = torch.randn(batch_size, data_size)

        encoder = ConditionalNormal(nn.Linear(data_size,2*latent_size))
        decoder = ConditionalNormal(nn.Linear(latent_size,2*data_size))

        transform = VAE(decoder=decoder, encoder=encoder)
        self.assert_stochastic_transform_is_well_behaved(transform, x, z_shape=(batch_size, latent_size), z_dtype=torch.float)


if __name__ == '__main__':
    unittest.main()
