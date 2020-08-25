import torch
import torchtestcase
import unittest
import copy
from survae.tests.nn import ModuleTest
from survae.nn.layers.autoregressive import AutoregressiveShift


class AutoregressiveShiftTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        seq_len = 7
        batch_size = 10
        features = 6
        x = torch.randn(seq_len, batch_size, features)

        module = AutoregressiveShift(embed_dim=features)
        self.assert_layer_is_well_behaved(module, x)

    def test_autoregressive_shift(self):
        seq_len = 7
        batch_size = 10
        features = 6
        x = torch.randn(seq_len, batch_size, features)

        module = AutoregressiveShift(embed_dim=features)
        y = module(x)

        # Assert input x at O positions [O O O X] equals output y at O positions [Y O O O]
        self.assertEqual(x[:-1], y[1:])


if __name__ == '__main__':
    unittest.main()
