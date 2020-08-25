import torch
import torchtestcase
import unittest
from survae.tests.distributions import DistributionTest
from survae.distributions import StandardUniform


class StandardUniformTest(DistributionTest):

    def test_distribution_is_well_behaved(self):
        batch_size = 10
        shape = [2, 3, 4]
        x = torch.randn(batch_size, *shape)
        distribution = StandardUniform(shape)

        self.assert_distribution_is_well_behaved(distribution, x, expected_shape=(batch_size, *shape))


if __name__ == '__main__':
    unittest.main()
