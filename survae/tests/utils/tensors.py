import torch
import torchtestcase
import unittest
from survae.utils import sum_except_batch, mean_except_batch
from survae.utils import split_leading_dim, merge_leading_dims, repeat_rows


class TensorUtilsTest(torchtestcase.TorchTestCase):

    def test_sum_except_batch(self):
        x1 = torch.ones(10)
        x1_sum = sum_except_batch(x1)
        self.assertEqual(x1_sum, torch.ones(10))

        x2 = torch.ones(10, 5)
        x2_sum = sum_except_batch(x2)
        self.assertEqual(x2_sum, torch.ones(10) * 5)

    def test_mean_except_batch(self):
        x1 = torch.ones(10)
        x1_sum = mean_except_batch(x1)
        self.assertEqual(x1_sum, torch.ones(10))

        x2 = torch.ones(10, 5)
        x2_sum = mean_except_batch(x2)
        self.assertEqual(x2_sum, torch.ones(10))

    def test_split_leading_dim(self):
        x = torch.randn(24, 5)
        self.assertEqual(split_leading_dim(x, [-1]), x)
        self.assertEqual(split_leading_dim(x, [2, -1]), x.view(2, 12, 5))
        self.assertEqual(split_leading_dim(x, [2, 3, -1]), x.view(2, 3, 4, 5))

    def test_merge_leading_dims(self):
        x = torch.randn(2, 3, 4, 5)
        self.assertEqual(merge_leading_dims(x, 1), x)
        self.assertEqual(merge_leading_dims(x, 2), x.view(6, 4, 5))
        self.assertEqual(merge_leading_dims(x, 3), x.view(24, 5))
        self.assertEqual(merge_leading_dims(x, 4), x.view(120))

    def test_split_merge_leading_dims_are_consistent(self):
        x = torch.randn(2, 3, 4, 5)
        y = split_leading_dim(merge_leading_dims(x, 1), [2])
        self.assertEqual(y, x)
        y = split_leading_dim(merge_leading_dims(x, 2), [2, 3])
        self.assertEqual(y, x)
        y = split_leading_dim(merge_leading_dims(x, 3), [2, 3, 4])
        self.assertEqual(y, x)
        y = split_leading_dim(merge_leading_dims(x, 4), [2, 3, 4, 5])
        self.assertEqual(y, x)

    def test_repeat_rows(self):
        x = torch.randn(2, 3, 4, 5)
        self.assertEqual(repeat_rows(x, 1), x)
        y = repeat_rows(x, 2)
        self.assertEqual(y.shape, torch.Size([4, 3, 4, 5]))
        self.assertEqual(x[0], y[0])
        self.assertEqual(x[0], y[1])
        self.assertEqual(x[1], y[2])
        self.assertEqual(x[1], y[3])


if __name__ == '__main__':
    unittest.main()
