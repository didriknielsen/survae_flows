import torch
import torchtestcase
import unittest
from survae.tests.nn import ModuleTest
from survae.nn.layers import LambdaLayer


class LambdaLayerTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        batch_size = 10
        shape = (6,)
        x = torch.randn(batch_size, *shape)

        module = LambdaLayer(lambda x: 2 * x - 1)
        self.assert_layer_is_well_behaved(module, x)

        y = module(x)
        self.assertEqual(y, 2 * x - 1)


if __name__ == '__main__':
    unittest.main()
