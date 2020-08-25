import torch
import torchtestcase
import unittest
from survae.tests.nn import ModuleTest
from survae.nn.layers import ElementwiseParams, ElementwiseParams1d, ElementwiseParams2d


class ElementwiseParamsTest(ModuleTest):

    def setUp(self):
        batch_size = 10
        shape = (6,)
        self.x = torch.randn(batch_size, *shape)

    def test_layer_is_well_behaved(self):
        module = ElementwiseParams(3)
        self.assert_layer_is_well_behaved(module, self.x)

    def test_shape(self):
        module = ElementwiseParams(3)
        y = module(self.x)
        expected_shape = (10, 2, 3)
        self.assertEqual(y.shape, expected_shape)

        module = ElementwiseParams(2)
        y = module(self.x)
        expected_shape = (10, 3, 2)
        self.assertEqual(y.shape, expected_shape)

    def test_order(self):
        module = ElementwiseParams(2, mode='interleaved')
        y = module(self.x)
        self.assertEqual(y[:,0], torch.stack([self.x[:,0], self.x[:,3]], dim=-1))
        self.assertEqual(y[:,1], torch.stack([self.x[:,1], self.x[:,4]], dim=-1))
        self.assertEqual(y[:,2], torch.stack([self.x[:,2], self.x[:,5]], dim=-1))

        module = ElementwiseParams(2, mode='sequential')
        y = module(self.x)
        self.assertEqual(y[:,0],self.x[:,0:2])
        self.assertEqual(y[:,1],self.x[:,2:4])
        self.assertEqual(y[:,2],self.x[:,4:6])


class ElementwiseParams1dTest(ModuleTest):

    def setUp(self):
        batch_size = 10
        shape = (6,4)
        self.x = torch.randn(batch_size, *shape)

    def test_layer_is_well_behaved(self):
        module = ElementwiseParams1d(3)
        self.assert_layer_is_well_behaved(module, self.x)

    def test_shape(self):
        module = ElementwiseParams1d(3)
        y = module(self.x)
        expected_shape = (10, 2, 4, 3)
        self.assertEqual(y.shape, expected_shape)

        module = ElementwiseParams1d(2)
        y = module(self.x)
        expected_shape = (10, 3, 4, 2)
        self.assertEqual(y.shape, expected_shape)

    def test_mode(self):
        module = ElementwiseParams1d(2, mode='interleaved')
        y = module(self.x)
        self.assertEqual(y[:,0], torch.stack([self.x[:,0], self.x[:,3]], dim=-1))
        self.assertEqual(y[:,1], torch.stack([self.x[:,1], self.x[:,4]], dim=-1))
        self.assertEqual(y[:,2], torch.stack([self.x[:,2], self.x[:,5]], dim=-1))

        module = ElementwiseParams1d(2, mode='sequential')
        y = module(self.x)
        self.assertEqual(y[:,0], torch.stack([self.x[:,0], self.x[:,1]], dim=-1))
        self.assertEqual(y[:,1], torch.stack([self.x[:,2], self.x[:,3]], dim=-1))
        self.assertEqual(y[:,2], torch.stack([self.x[:,4], self.x[:,5]], dim=-1))


class ElementwiseParams2dTest(ModuleTest):

    def setUp(self):
        batch_size = 10
        shape = (6,4,4)
        self.x = torch.randn(batch_size, *shape)

    def test_layer_is_well_behaved(self):
        module = ElementwiseParams2d(3)
        self.assert_layer_is_well_behaved(module, self.x)

    def test_shape(self):
        module = ElementwiseParams2d(3)
        y = module(self.x)
        expected_shape = (10, 2, 4, 4, 3)
        self.assertEqual(y.shape, expected_shape)

        module = ElementwiseParams2d(2)
        y = module(self.x)
        expected_shape = (10, 3, 4, 4, 2)
        self.assertEqual(y.shape, expected_shape)

    def test_mode(self):
        module = ElementwiseParams2d(2, mode='interleaved')
        y = module(self.x)
        self.assertEqual(y[:,0], torch.stack([self.x[:,0], self.x[:,3]], dim=-1))
        self.assertEqual(y[:,1], torch.stack([self.x[:,1], self.x[:,4]], dim=-1))
        self.assertEqual(y[:,2], torch.stack([self.x[:,2], self.x[:,5]], dim=-1))

        module = ElementwiseParams2d(2, mode='sequential')
        y = module(self.x)
        self.assertEqual(y[:,0], torch.stack([self.x[:,0], self.x[:,1]], dim=-1))
        self.assertEqual(y[:,1], torch.stack([self.x[:,2], self.x[:,3]], dim=-1))
        self.assertEqual(y[:,2], torch.stack([self.x[:,4], self.x[:,5]], dim=-1))


if __name__ == '__main__':
    unittest.main()
