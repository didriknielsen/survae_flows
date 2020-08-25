import torch
import torchtestcase
import unittest
import math
import torch
from survae.tests.distributions import DistributionTest
from survae.distributions import StandardNormal, DiagonalNormal, ConvNormal2d


class StandardNormalTest(DistributionTest):

    def test_distribution_is_well_behaved(self):
        batch_size = 10
        shape = [3, 2, 2]
        x = torch.randn(batch_size, *shape)
        distribution = StandardNormal(shape)

        self.assert_distribution_is_well_behaved(distribution, x, expected_shape=(batch_size, *shape))

    def test_matching_torchdist(self):
        batch_size = 10
        shape = [3, 2, 2]
        x = torch.randn(batch_size, *shape)

        distribution = StandardNormal(shape)
        torchdist = torch.distributions.Normal(torch.zeros(shape),torch.ones(shape))

        log_prob = distribution.log_prob(x)
        log_prob2 = torchdist.log_prob(x).reshape(batch_size, -1).sum(1)

        self.assertEqual(log_prob, log_prob2)


class DiagonalNormalTest(DistributionTest):

    def test_distribution_is_well_behaved(self):
        batch_size = 10
        shape = [2, 3, 4]
        x = torch.randn(batch_size, *shape)
        distribution = DiagonalNormal(shape)

        self.assert_distribution_is_well_behaved(distribution, x, expected_shape=(batch_size, *shape))

    def test_matching_torchdist(self):
        batch_size = 10
        shape = [3, 2, 2]
        x = torch.randn(batch_size, *shape)

        for (loc, log_scale) in [(2.0, 1.0), (-5.0, 5.0), (-0.7, -0.3)]:
            with self.subTest(loc=loc,
                              log_scale=log_scale):

                distribution = DiagonalNormal(shape)
                distribution.loc.data.fill_(loc)
                distribution.log_scale.data.fill_(log_scale)
                torchdist = torch.distributions.Normal(loc * torch.ones(shape), math.exp(log_scale) * torch.ones(shape))

                log_prob = distribution.log_prob(x)
                log_prob2 = torchdist.log_prob(x).reshape(batch_size, -1).sum(1)
                # Differences include:
                # 1: `* 0.5 * exp(-2*log_scale)` vs `/ (2 * log_scale.exp()**2)`
                # 2: `0.5 * math.log(2 * math.pi)` vs `math.log(math.sqrt(2 * math.pi))`
                # 3: `log_scale` vs `log_scale.exp().log()`

                self.eps = 1e-5
                self.assertEqual(log_prob, log_prob2)


class ConvNormal2dTest(DistributionTest):

    def test_distribution_is_well_behaved(self):
        batch_size = 10
        shape = [3, 2, 2]
        x = torch.randn(batch_size, *shape)
        distribution = ConvNormal2d(shape)

        self.assert_distribution_is_well_behaved(distribution, x, expected_shape=(batch_size, *shape))


if __name__ == '__main__':
    unittest.main()
