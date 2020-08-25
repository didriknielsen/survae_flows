import torch
import torchtestcase
import unittest
import copy
from survae.nn.layers.autoregressive import MaskedLinear
from survae.nn.nets.autoregressive import MADE, AgnosticMADE


class MADETest(torchtestcase.TorchTestCase):

    def test_shape(self):
        batch_size = 16
        features = 10
        hidden_features = 5*[50]
        num_params = 3

        inputs = torch.randn(batch_size, features)

        for random_order, random_mask in [(False, False),
                                          (False, True),
                                          (True, False),
                                          (True, True)]:
            with self.subTest(random_order=random_order,
                              random_mask=random_mask):
                model = MADE(
                    features=features,
                    num_params=num_params,
                    hidden_features=hidden_features,
                    random_order=random_order,
                    random_mask=random_mask,
                )
                outputs = model(inputs)
                self.assertEqual(outputs.dim(), 3)
                self.assertEqual(outputs.shape[0], batch_size)
                self.assertEqual(outputs.shape[1], features)
                self.assertEqual(outputs.shape[2], num_params)

    def test_total_mask_sequential(self):
        features = 10
        hidden_features = 5*[50]
        num_params = 1

        model = MADE(
            features=features,
            num_params=num_params,
            hidden_features=hidden_features,
            random_order=False,
            random_mask=False,
        )
        total_mask = None
        for module in model.modules():
            if isinstance(module, MaskedLinear):
                if total_mask is None:
                    total_mask = module.mask
                else:
                    total_mask = module.mask @ total_mask
        total_mask = (total_mask > 0).float()
        reference = torch.tril(torch.ones([features, features]), -1)
        self.assertEqual(total_mask, reference)

    def test_total_mask_random(self):
        features = 10
        hidden_features = 5*[50]
        num_params = 1

        model = MADE(
            features=features,
            num_params=num_params,
            hidden_features=hidden_features,
            random_order=False,
            random_mask=True,
        )
        total_mask = None
        for module in model.modules():
            if isinstance(module, MaskedLinear):
                if total_mask is None:
                    total_mask = module.mask
                else:
                    total_mask = module.mask @ total_mask
        total_mask = (total_mask > 0).float()
        self.assertEqual(torch.triu(total_mask), torch.zeros([features, features]))

    def test_autoregressive_type_A(self):
        batch_size = 16
        features = 10
        hidden_features = 2*[50]
        num_params = 3
        x = torch.randn(batch_size, features)
        x_altered = copy.deepcopy(x)
        x_altered[:,2] += 100.0 # Alter feature number 2

        for random_mask in [True, False]:
            with self.subTest(random_mask=random_mask):
                module = MADE(
                    features=features,
                    num_params=num_params,
                    hidden_features=hidden_features,
                    random_order=False,
                    random_mask=random_mask,
                )
                y = module(x)
                y_altered = module(x_altered)

                # Assert all elements up to (and including) 2 are unaltered
                self.assertEqual(y[:,:3], y_altered[:,:3])

                # Assert all elements from 2 are altered
                self.assertFalse((y[:,3:] == y_altered[:,3:]).view(-1).all())


class AgnosticMADETest(torchtestcase.TorchTestCase):

    def test_shape(self):
        batch_size = 16
        features = 10
        hidden_features = 5*[50]
        num_params = 3

        inputs = torch.randn(batch_size, features)

        for order_agnostic, connect_agnostic in [(False, False),
                                                 (False, True),
                                                 (True, False),
                                                 (True, True)]:
            with self.subTest(order_agnostic=order_agnostic,
                              connect_agnostic=connect_agnostic):
                model = AgnosticMADE(
                    features=features,
                    num_params=num_params,
                    hidden_features=hidden_features,
                    order_agnostic=order_agnostic,
                    connect_agnostic=connect_agnostic,
                    num_masks=16,
                )
                outputs = model(inputs)
                self.assertEqual(outputs.dim(), 3)
                self.assertEqual(outputs.shape[0], batch_size)
                self.assertEqual(outputs.shape[1], features)
                self.assertEqual(outputs.shape[2], num_params)

    def test_autoregressive_type_A(self):
        batch_size = 16
        features = 10
        hidden_features = 2*[50]
        num_params = 3
        x = torch.randn(batch_size, features)
        x_altered = copy.deepcopy(x)
        x_altered[:,2] += 100.0 # Alter feature number 2

        for connect_agnostic in [True, False]:
            with self.subTest(connect_agnostic=connect_agnostic):
                module = AgnosticMADE(
                    features=features,
                    num_params=num_params,
                    hidden_features=hidden_features,
                    order_agnostic=False,
                    connect_agnostic=connect_agnostic,
                    num_masks=2,
                )
                y = module(x) # Call with mask 0, mask updated to 1
                _ = module(x) # Call with mask 1, mask updated to 0
                y_altered = module(x_altered) # Call with mask 0, mask updated to 1

                # Assert all elements up to (and including) 2 are unaltered
                self.assertEqual(y[:,:3], y_altered[:,:3])

                # Assert all elements from 2 are altered
                self.assertFalse((y[:,3:] == y_altered[:,3:]).view(-1).all())

    def test_connect_agnostic(self):
        batch_size = 16
        features = 10
        hidden_features = 2*[50]
        num_params = 3
        x = torch.randn(batch_size, features)
        x_altered = copy.deepcopy(x)
        x_altered[:,2] += 100.0 # Alter feature number 2


        for order_agnostic, connect_agnostic in [(False, True),
                                                 (True, False),
                                                 (True, True)]:
            with self.subTest(order_agnostic=order_agnostic,
                              connect_agnostic=connect_agnostic):

                module = AgnosticMADE(
                    features=features,
                    num_params=num_params,
                    hidden_features=hidden_features,
                    order_agnostic=order_agnostic,
                    connect_agnostic=connect_agnostic,
                    num_masks=2,
                )
                y = module(x) # Call with mask 0, mask updated to 1
                y_mask1 = module(x) # Call with mask 1, mask updated to 0
                y_mask0 = module(x) # Call with mask 0, mask updated to 1

                # Assert elements same for same mask
                self.assertTrue((y == y_mask0).view(-1).all())

                # Assert some elements different for different mask
                self.assertTrue((y != y_mask1).view(-1).any())


if __name__ == '__main__':
    unittest.main()
