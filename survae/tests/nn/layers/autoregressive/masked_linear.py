import torch
import torchtestcase
import unittest
import copy
from survae.tests.nn import ModuleTest
from survae.nn.layers.autoregressive import MaskedLinear


class MaskedLinearTest(ModuleTest):

    def test_layer_is_well_behaved(self):
        batch_size = 16
        features = 10
        x = torch.randn(batch_size, features)

        data_degrees = MaskedLinear.get_data_degrees(features)

        module = MaskedLinear(data_degrees, 5, features)
        self.assert_layer_is_well_behaved(module, x)

        module = MaskedLinear(data_degrees, 5, features, random_mask=True)
        self.assert_layer_is_well_behaved(module, x)

    def test_autoregressive_type_B(self):
        batch_size = 16
        features = 10
        x = torch.randn(batch_size, features)
        x_altered = copy.deepcopy(x)
        x_altered[:,2] += 100.0 # Alter feature number 2

        data_degrees = MaskedLinear.get_data_degrees(features)

        module = MaskedLinear(data_degrees, features, features, random_mask=False)
        y = module(x)
        y_altered = module(x_altered)

        # Assert all elements up to (and NOT including) 2 are unaltered
        self.assertEqual(y[:,:2], y_altered[:,:2])

        # Assert all elements from 2 are altered
        self.assertFalse((y[:,2:] == y_altered[:,2:]).view(-1).all())


if __name__ == '__main__':
    unittest.main()
