import numpy as np
import torch
import torchtestcase
import unittest
from survae.transforms import Squeeze2d
from survae.tests.transforms.bijections import BijectionTest


class Squeeze2dTest(BijectionTest):

    def test_bijection_is_well_behaved(self):
        batch_size = 10
        setups = [
            (2, [1, 2, 2], [4, 1, 1]),
            (2, [1, 4, 4], [4, 2, 2]),
            (2, [2, 4, 4], [8, 2, 2]),
            (2, [3, 4, 4], [12, 2, 2]),
            (2, [1, 2, 2], [4, 1, 1]),
            (2, [1, 4, 4], [4, 2, 2]),
            (2, [2, 4, 4], [8, 2, 2]),
            (2, [3, 4, 4], [12, 2, 2]),
            (3, [1, 3, 3], [9, 1, 1]),
            (3, [1, 9, 9], [9, 3, 3]),
            (3, [2, 9, 9], [18, 3, 3]),
            (3, [3, 9, 9], [27, 3, 3]),
            (3, [1, 3, 3], [9, 1, 1]),
            (3, [1, 9, 9], [9, 3, 3]),
            (3, [2, 9, 9], [18, 3, 3]),
            (3, [3, 9, 9], [27, 3, 3]),
        ]

        for ordered in (False, True):
            for factor, x_shape, expected_z_shape in setups:
                with self.subTest(factor=factor, ordered=ordered, x_shape=x_shape, expected_z_shape=expected_z_shape):
                    x = torch.randn(batch_size, *x_shape)
                    bijection = Squeeze2d(factor, ordered=ordered)
                    self.assert_bijection_is_well_behaved(bijection, x, z_shape=(batch_size, *expected_z_shape))

    def test_forward(self):
        bijection = Squeeze2d(2)
        x = torch.LongTensor([[[[11,  12,  21,  22],
                                [13,  14,  23,  24],
                                [31,  32,  41,  42],
                                [33,  34,  43,  44]],
                               [[110, 120, 210, 220],
                                [130, 140, 230, 240],
                                [310, 320, 410, 420],
                                [330, 340, 430, 440]]]])
        z, _ = bijection.forward(x)

        def assert_channel_equal(channel, values):
            self.assertEqual(z[0, channel, ...], torch.LongTensor(values))

        assert_channel_equal(0, [[11,21],
                                 [31,41]])
        assert_channel_equal(1, [[12,22],
                                 [32,42]])
        assert_channel_equal(2, [[13,23],
                                 [33,43]])
        assert_channel_equal(3, [[14,24],
                                 [34,44]])
        assert_channel_equal(4, [[110,210],
                                 [310,410]])
        assert_channel_equal(5, [[120,220],
                                 [320,420]])
        assert_channel_equal(6, [[130,230],
                                 [330,430]])
        assert_channel_equal(7, [[140,240],
                                 [340,440]])

    def test_forward_ordered(self):
        bijection = Squeeze2d(2, ordered=True)
        x = torch.LongTensor([[[[11,  12,  21,  22],
                                [13,  14,  23,  24],
                                [31,  32,  41,  42],
                                [33,  34,  43,  44]],
                               [[110, 120, 210, 220],
                                [130, 140, 230, 240],
                                [310, 320, 410, 420],
                                [330, 340, 430, 440]]]])
        z, _ = bijection.forward(x)

        def assert_channel_equal(channel, values):
            self.assertEqual(z[0, channel, ...], torch.LongTensor(values))

        assert_channel_equal(0, [[11,21],
                                 [31,41]])
        assert_channel_equal(1, [[110,210],
                                 [310,410]])
        assert_channel_equal(2, [[12,22],
                                 [32,42]])
        assert_channel_equal(3, [[120,220],
                                 [320,420]])
        assert_channel_equal(4, [[13,23],
                                 [33,43]])
        assert_channel_equal(5, [[130,230],
                                 [330,430]])
        assert_channel_equal(6, [[14,24],
                                 [34,44]])
        assert_channel_equal(7, [[140,240],
                                 [340,440]])

    def test_forward_wrong_shape(self):
        batch_size = 10
        bijection = Squeeze2d(2)
        for shape in [[32, 3, 3],
                      [32, 5, 5],
                      [32, 4]]:
            with self.subTest(shape=shape):
                x = torch.randn(batch_size, *shape)
                with self.assertRaises(AssertionError):
                    bijection.forward(x)

    def test_inverse_wrong_shape(self):
        batch_size = 10
        bijection = Squeeze2d(2)
        for shape in [[3, 4, 4],
                      [33, 4, 4],
                      [32, 4]]:
            with self.subTest(shape=shape):
                z = torch.randn(batch_size, *shape)
                with self.assertRaises(AssertionError):
                    bijection.inverse(z)


if __name__ == '__main__':
    unittest.main()
