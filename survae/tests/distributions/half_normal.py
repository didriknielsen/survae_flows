import torch
import torchtestcase
import unittest
import math
import torch
from survae.tests.distributions import DistributionTest
from survae.distributions import StandardHalfNormal


class StandardHalfNormalTest(DistributionTest):

    def test_distribution_is_well_behaved(self):
        batch_size = 10
        shape = [3, 2, 2]
        x = torch.randn(batch_size, *shape).abs()
        distribution = StandardHalfNormal(shape)

        self.assert_distribution_is_well_behaved(distribution, x, expected_shape=(batch_size, *shape))

    def test_matching_torchdist(self):
        batch_size = 10
        shape = [3, 2, 2]
        x = torch.randn(batch_size, *shape).abs()

        distribution = StandardHalfNormal(shape)
        torchdist = torch.distributions.HalfNormal(torch.ones(shape))

        log_prob = distribution.log_prob(x)
        log_prob2 = torchdist.log_prob(x).reshape(batch_size, -1).sum(1)

        self.eps = 1e-5
        self.assertEqual(log_prob, log_prob2)


if __name__ == '__main__':
    unittest.main()
