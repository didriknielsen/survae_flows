import torch
import torchtestcase
import unittest
import copy
from survae.nn.nets.autoregressive import PixelCNN


class PixelCNNTest(torchtestcase.TorchTestCase):

    def test_autoregressive_type_A(self):
        batch_size = 10
        shape = (3,8,8)
        x = torch.randn(batch_size, *shape)
        x_altered = copy.deepcopy(x)
        x_altered[:,1,4,2] += 100.0 # Alter channel G in position (4,2)

        module = PixelCNN(3, num_params=2, kernel_size=3, filters=8, num_blocks=1, output_filters=16)
        y = module(x)
        y_altered = module(x_altered)

        # Assert all pixels up to (4,2) are unaltered
        self.assertEqual(y[:,:,:4], y_altered[:,:,:4])
        self.assertEqual(y[:,:,4,:2], y_altered[:,:,4,:2])

        # Assert channel R is unaltered
        self.assertEqual(y[:,0,4,2], y_altered[:,0,4,2])

        # Assert channel G is unaltered
        self.assertEqual(y[:,1,4,2], y_altered[:,1,4,2])

        # Assert channel B is altered
        self.assertFalse((y[:,2,4,2] == y_altered[:,2,4,2]).view(-1).any())

        # Assert all elements in next pixel are altered
        self.assertFalse((y[:,:,4,3] == y_altered[:,:,4,3]).view(-1).any())


if __name__ == '__main__':
    unittest.main()
