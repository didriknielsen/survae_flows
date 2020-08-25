import torch
import torchtestcase
import unittest
import copy
from survae.nn.nets.autoregressive import DecoderOnlyTransformer2d


class DecoderOnlyTransformer2dTest(torchtestcase.TorchTestCase):

    def test_autoregressive_type_A_raster_cwh(self):
        batch_size = 10
        shape = (3,8,8)
        num_bits = 5
        x = torch.randint(0, 2**num_bits, (batch_size,)+shape)
        x_altered = copy.deepcopy(x)
        x_altered[:,1,4,2] = torch.randint(0, 2**num_bits, (batch_size,)) # Alter channel G in position (4,2)

        module = DecoderOnlyTransformer2d(shape,
                                          output_dim=2,
                                          num_bits=num_bits,
                                          autoregressive_order='cwh',
                                          d_model=12,
                                          nhead=2,
                                          num_layers=2,
                                          dim_feedforward=24,
                                          dropout=0.0,
                                          checkpoint_blocks=False)
        module.eval()
        y = module(x)
        y_altered = module(x_altered)

        # Assert shape is correct
        self.assertEqual(y.shape, torch.Size([10,3,8,8,2]))

        # Assert all pixels up to (4,2) are unaltered
        self.assertEqual(y[:,:,:4], y_altered[:,:,:4])
        self.assertEqual(y[:,:,4,:2], y_altered[:,:,4,:2])

        # Assert channel R is unaltered
        self.assertEqual(y[:,0,4,2], y_altered[:,0,4,2])

        # Assert channel G is unaltered
        self.assertEqual(y[:,1,4,2], y_altered[:,1,4,2])

        # Assert channel B is altered
        self.assertFalse((y[:,2,4,2] == y_altered[:,2,4,2]).reshape(-1).all())

        # Assert some elements in next pixel are altered
        self.assertFalse((y[:,:,4,3] == y_altered[:,:,4,3]).reshape(-1).all())

    def test_autoregressive_type_A_raster_whc(self):
        batch_size = 10
        shape = (3,8,8)
        num_bits = 5
        x = torch.randint(0, 2**num_bits, (batch_size,)+shape)
        x_altered = copy.deepcopy(x)
        x_altered[:,1,4,2] = torch.randint(0, 2**num_bits, (batch_size,)) # Alter channel G in position (4,2)

        module = DecoderOnlyTransformer2d(shape,
                                          output_dim=2,
                                          num_bits=num_bits,
                                          autoregressive_order='whc',
                                          d_model=12,
                                          nhead=2,
                                          num_layers=2,
                                          dim_feedforward=24,
                                          dropout=0.0,
                                          checkpoint_blocks=False)
        module.eval()
        y = module(x)
        y_altered = module(x_altered)

        # Assert shape is correct
        self.assertEqual(y.shape, torch.Size([10,3,8,8,2]))

        # Assert channel R is unaltered
        self.assertEqual(y[:,0], y_altered[:,0])

        # Assert all pixels up to (4,2) of channel G are unaltered
        self.assertEqual(y[:,1,:4], y_altered[:,1,:4])
        self.assertEqual(y[:,1,4,:3], y_altered[:,1,4,:3])

        # Assert all pixel (4,3) of channel G is altered
        self.assertFalse((y[:,1,4,3] == y_altered[:,1,4,3]).reshape(-1).all())

        # Assert channel B is altered
        self.assertFalse((y[:,2,4,2] == y_altered[:,2,4,2]).reshape(-1).all())


if __name__ == '__main__':
    unittest.main()
