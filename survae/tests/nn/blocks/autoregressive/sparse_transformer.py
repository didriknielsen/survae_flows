import torch
import torchtestcase
import unittest
import copy
from survae.tests.nn import ModuleTest
from survae.nn.blocks.autoregressive import DenseTransformer


class DenseTransformerTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        seq_len = 7
        batch_size = 10
        features = 6
        x = torch.randn(seq_len, batch_size, features)

        module = DenseTransformer(d_model=features, dim_feedforward=4*features, nhead=2)
        self.assert_layer_is_well_behaved(module, x)

        module = DenseTransformer(d_model=features, dim_feedforward=4*features, nhead=2, checkpoint_blocks=True)
        self.assert_layer_is_well_behaved(module, x)

    def test_autoregressive_type_B(self):
        seq_len = 7
        batch_size = 10
        features = 6
        x = torch.randn(seq_len, batch_size, features)
        x_altered = copy.deepcopy(x)
        x_altered[4,:,1] += 100.0 # Alter features 1 in position 4

        module = DenseTransformer(d_model=features, dim_feedforward=4*features, nhead=2, dropout=0.0)
        module.eval()
        y = module(x)
        y_altered = module(x_altered)

        # Assert every element up to (but not including) 4 is unaltered
        self.assertEqual(y[:4], y_altered[:4])

        # Assert element 4 is altered
        self.assertFalse((y[4] == y_altered[4]).view(-1).any())


if __name__ == '__main__':
    unittest.main()
