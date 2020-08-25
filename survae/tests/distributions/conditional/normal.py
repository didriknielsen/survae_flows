import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.tests.distributions.conditional import ConditionalDistributionTest
from survae.distributions import ConditionalMeanNormal, ConditionalMeanStdNormal, ConditionalNormal


class ConditionalMeanNormalTest(ConditionalDistributionTest):

    def test_distribution_is_well_behaved(self):
        batch_size = 16
        size = 10
        x = torch.randn(batch_size, size)

        # Basic check
        context = torch.randn(batch_size, 20)
        distribution = ConditionalMeanNormal(net=nn.Linear(20,size))
        self.assert_distribution_is_well_behaved(distribution, x, context, expected_shape=(batch_size, size))

        # Check mean
        mean = distribution.mean(context)
        self.assert_tensor_is_good(mean, x.shape)


class ConditionalMeanStdNormalTest(ConditionalDistributionTest):

    def test_distribution_is_well_behaved(self):
        batch_size = 16
        size = 10
        x = torch.randn(batch_size, size)

        # Basic check
        context = torch.randn(batch_size, 20)
        distribution = ConditionalMeanStdNormal(net=nn.Linear(20,size), scale_shape=(1,))
        self.assert_distribution_is_well_behaved(distribution, x, context, expected_shape=(batch_size, size))

        # Check mean
        mean = distribution.mean(context)
        self.assert_tensor_is_good(mean, x.shape)


class ConditionalNormalTest(ConditionalDistributionTest):

    def test_distribution_is_well_behaved(self):
        batch_size = 16
        size = 10
        x = torch.randn(batch_size, size)

        # Basic check
        context = torch.randn(batch_size, 20)
        distribution = ConditionalNormal(net=nn.Linear(20,2*size))
        self.assert_distribution_is_well_behaved(distribution, x, context, expected_shape=(batch_size, size))

        # Check mean
        mean = distribution.mean(context)
        self.assert_tensor_is_good(mean, x.shape)

        # Check mean_stddev
        mean, stddev = distribution.mean_stddev(context)
        self.assert_tensor_is_good(mean, x.shape)
        self.assert_tensor_is_good(stddev, x.shape)


if __name__ == '__main__':
    unittest.main()
