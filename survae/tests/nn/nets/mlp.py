import torch
import torchtestcase
import unittest
import copy
from survae.nn.nets import MLP
from survae.tests.nn import ModuleTest


class MLPTest(ModuleTest):

    def test_forward(self):
        batch_size = 10

        shape = (12,)
        x = torch.randn(batch_size, *shape)

        module = MLP(12, 6, hidden_units=[12], activation='relu')
        y = module(x)
        self.assert_tensor_is_good(y, shape=(batch_size, 6))

        shape = (3,2,2)
        x = torch.randn(batch_size, *shape)
        module = MLP(12, 6, hidden_units=[12], activation='relu',
                     in_lambda=lambda x: x.view(x.shape[0], -1),
                     out_lambda=lambda x: x.view(x.shape[0], 6, 1, 1))
        y = module(x)
        self.assert_tensor_is_good(y, shape=(batch_size, 6, 1, 1))





if __name__ == '__main__':
    unittest.main()
