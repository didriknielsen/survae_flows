import torch
import torchtestcase
import unittest
from survae.tests.nn import ModuleTest
from survae.nn.blocks import DenseBlock


class DenseBlockTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        for gated_conv in [False, True]:
            with self.subTest(gated_conv=gated_conv):
                x = torch.randn(10, 3, 8, 8)
                module = DenseBlock(in_channels=3, out_channels=6, depth=2, growth=4,
                                    dropout=0.0, gated_conv=gated_conv, zero_init=False)
                self.assert_layer_is_well_behaved(module, x)

    def test_zero_init(self):
        x = torch.randn(10, 3, 8, 8)
        module = DenseBlock(in_channels=3, out_channels=6, depth=2, growth=4,
                            dropout=0.0, gated_conv=False, zero_init=True)
        y = module(x)
        self.assertEqual(y, torch.zeros(10, 6, 8, 8))


if __name__ == '__main__':
    unittest.main()
