import torch
import torch.nn as nn
import torchtestcase
import unittest
from survae.tests.distributions.conditional import ConditionalDistributionTest
from survae.distributions import ConditionalBernoulli


class ConditionalBernoulliTest(ConditionalDistributionTest):

    def setUp(self):
        self.batch_size = 16
        self.size = 10
        self.x = torch.randint(2, [self.batch_size, self.size])
        self.context = torch.randn(self.batch_size, 20)
        self.distribution = ConditionalBernoulli(net=nn.Linear(20,self.size))

    def test_distribution_is_well_behaved(self):
        self.assert_distribution_is_well_behaved(self.distribution, self.x, self.context, expected_shape=(self.batch_size, self.size))

    def test_logits(self):
        logits = self.distribution.logits(self.context)
        self.assert_tensor_is_good(logits, self.x.shape)

    def test_probs(self):
        probs = self.distribution.probs(self.context)
        self.assert_tensor_is_good(probs, self.x.shape)
        self.assertTrue(probs.min() >= 0.0)
        self.assertTrue(probs.max() <= 1.0)

    def test_mode(self):
        mode = self.distribution.mode(self.context)
        self.assert_tensor_is_good(mode, self.x.shape)

    def test_mean(self):
        mean = self.distribution.mean(self.context)
        self.assert_tensor_is_good(mean, self.x.shape)


if __name__ == '__main__':
    unittest.main()
