import torch
import torchtestcase
import unittest
import copy
from survae.tests.nn import ModuleTest
from survae.nn.layers.autoregressive import Image2Seq, Seq2Image


class Image2Seq2ImageTest(ModuleTest):

    ar_orders = {'cwh','whc','zigzag_cs'}

    def test_layer_is_well_behaved(self):
        seq_len = 3*4*4
        image_shape = (3,4,4)
        batch_size = 10
        features = 6
        x_seq = torch.randn(seq_len, batch_size, features)
        x_image = torch.randn(batch_size, *image_shape, features)

        for ar_order in self.ar_orders:
            self.assert_layer_is_well_behaved(Image2Seq(ar_order, image_shape), x_image)
            self.assert_layer_is_well_behaved(Seq2Image(ar_order, image_shape), x_seq)

    def test_cwh(self):
        x_image = torch.arange(12).float().reshape([1,3,2,2,1])

        im2seq = Image2Seq('cwh', (3,2,2))
        x_seq = im2seq(x_image)

        self.assertEqual(x_seq[0,:,:], x_image[:,0,0,0,:])
        self.assertEqual(x_seq[1,:,:], x_image[:,1,0,0,:])
        self.assertEqual(x_seq[2,:,:], x_image[:,2,0,0,:])
        self.assertEqual(x_seq[3,:,:], x_image[:,0,0,1,:])
        self.assertEqual(x_seq[4,:,:], x_image[:,1,0,1,:])
        self.assertEqual(x_seq[5,:,:], x_image[:,2,0,1,:])
        self.assertEqual(x_seq[6,:,:], x_image[:,0,1,0,:])

    def test_whc(self):
        x_image = torch.arange(12).float().reshape([1,3,2,2,1])

        im2seq = Image2Seq('whc', (3,2,2))
        x_seq = im2seq(x_image)

        self.assertEqual(x_seq[0,:,:], x_image[:,0,0,0,:])
        self.assertEqual(x_seq[1,:,:], x_image[:,0,0,1,:])
        self.assertEqual(x_seq[2,:,:], x_image[:,0,1,0,:])
        self.assertEqual(x_seq[3,:,:], x_image[:,0,1,1,:])
        self.assertEqual(x_seq[4,:,:], x_image[:,1,0,0,:])

    def test_zigzag_cs(self):
        x_image = torch.arange(3*4*4).float().reshape([1,3,4,4,1])

        im2seq = Image2Seq('zigzag_cs', (2,4,4))
        x_seq = im2seq(x_image)

        self.assertEqual(x_seq[0,:,:], x_image[:,0,0,0,:])
        self.assertEqual(x_seq[1,:,:], x_image[:,1,0,0,:])
        self.assertEqual(x_seq[2,:,:], x_image[:,0,0,1,:])
        self.assertEqual(x_seq[3,:,:], x_image[:,1,0,1,:])
        self.assertEqual(x_seq[4,:,:], x_image[:,0,1,0,:])
        self.assertEqual(x_seq[5,:,:], x_image[:,1,1,0,:])
        self.assertEqual(x_seq[6,:,:], x_image[:,0,2,0,:])
        self.assertEqual(x_seq[7,:,:], x_image[:,1,2,0,:])
        self.assertEqual(x_seq[8,:,:], x_image[:,0,1,1,:])
        self.assertEqual(x_seq[9,:,:], x_image[:,1,1,1,:])
        self.assertEqual(x_seq[10,:,:], x_image[:,0,0,2,:])
        self.assertEqual(x_seq[11,:,:], x_image[:,1,0,2,:])
        self.assertEqual(x_seq[12,:,:], x_image[:,0,0,3,:])
        self.assertEqual(x_seq[13,:,:], x_image[:,1,0,3,:])

    def test_im2seq2im(self):
        seq_len = 3*4*4
        image_shape = (3,4,4)
        batch_size = 10
        features = 6
        x_image = torch.randn(batch_size, *image_shape, features)

        for ar_order in self.ar_orders:
            im2seq = Image2Seq(ar_order, image_shape)
            seq2im = Seq2Image(ar_order, image_shape)
            x_seq = im2seq(x_image)
            x_image2 = seq2im(x_seq)

            self.assertEqual(x_seq.shape, torch.Size([seq_len, batch_size, features]))
            self.assertEqual(x_image2.shape, torch.Size([batch_size, *image_shape, features]))
            self.assertEqual(x_image, x_image2)

    def test_seq2im2seq(self):
        seq_len = 3*4*4
        image_shape = (3,4,4)
        batch_size = 10
        features = 6
        x_seq = torch.randn(seq_len, batch_size, features)

        for ar_order in self.ar_orders:
            im2seq = Image2Seq(ar_order, image_shape)
            seq2im = Seq2Image(ar_order, image_shape)
            x_image = seq2im(x_seq)
            x_seq2 = im2seq(x_image)

            self.assertEqual(x_image.shape, torch.Size([batch_size, *image_shape, features]))
            self.assertEqual(x_seq2.shape, torch.Size([seq_len, batch_size, features]))
            self.assertEqual(x_seq, x_seq2)


if __name__ == '__main__':
    unittest.main()
