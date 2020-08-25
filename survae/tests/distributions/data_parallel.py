import torch
import torchtestcase
import unittest
from survae.distributions import DataParallelDistribution, StandardNormal


class DataParallelDistributionTest(torchtestcase.TorchTestCase):

    def test_data_parallel(self):
        batch_size = 12
        shape = [2, 3, 4]
        x = torch.rand([batch_size] + shape)
        distribution = DataParallelDistribution(StandardNormal(shape))

        log_prob = distribution.log_prob(x)
        samples = distribution.sample(batch_size)
        samples2, log_prob2 = distribution.sample_with_log_prob(batch_size)

        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob2, torch.Tensor)
        self.assertIsInstance(samples2, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
