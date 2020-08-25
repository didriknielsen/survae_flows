import torch
import torchtestcase
import unittest
from survae.tests.nn import ModuleTest
from survae.nn.layers.encoding import PositionalEncodingImage


class PositionalEncodingImageTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        batch_size = 10
        shape = (3,8,8)
        x = torch.rand(batch_size, *shape, 16)

        module = PositionalEncodingImage(shape, 16)
        self.assert_layer_is_well_behaved(module, x)

    def test_output(self):
        batch_size = 10
        shape = (3,8,8)
        x = torch.zeros(batch_size, *shape, 16)

        module = PositionalEncodingImage(shape, 16)
        y = module(x)

        self.assertEqual(y.shape, torch.Size([10, 3, 8, 8, 16]))

        upper_left_channel0 = module.encode_c[0,0,0,0]+module.encode_h[0,0,0,0]+module.encode_w[0,0,0,0]
        self.assertEqual(y[0,0,0,0], upper_left_channel0)
        self.assertEqual(y[1,0,0,0], upper_left_channel0)
        # ...
        self.assertEqual(y[9,0,0,0], upper_left_channel0)


if __name__ == '__main__':
    unittest.main()
