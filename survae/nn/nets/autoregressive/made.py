import warnings
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.nn.layers import act_module, ElementwiseParams
from survae.nn.layers.autoregressive import MaskedLinear


# Adapted from https://github.com/bayesiains/nsf/blob/master/nde/transforms/made.py
# and https://github.com/karpathy/pytorch-made/blob/master/made.py

class MADE(nn.Sequential):
    """Implementation of MADE."""

    def __init__(self,
                 features,
                 num_params,
                 hidden_features,
                 random_order=False,
                 random_mask=False,
                 random_seed=None,
                 activation='relu',
                 dropout_prob=0.,
                 batch_norm=False):

        layers = []

        # Build layers
        data_degrees = MaskedLinear.get_data_degrees(features, random_order=random_order, random_seed=random_seed)
        in_degrees = copy.deepcopy(data_degrees)
        for i, out_features in enumerate(hidden_features):
            layers.append(MaskedLinear(in_degrees=in_degrees,
                                       out_features=out_features,
                                       data_features=features,
                                       random_mask=random_mask,
                                       random_seed=random_seed+i if random_seed else None, # Change random seed to get different masks
                                       is_output=False))
            in_degrees = layers[-1].degrees
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(act_module(activation))
            if dropout_prob>0.0:
                layers.append(nn.Dropout(dropout_prob))

        # Build output layer
        layers.append(MaskedLinear(in_degrees=in_degrees,
                                   out_features=features * num_params,
                                   data_features=features,
                                   random_mask=random_mask,
                                   random_seed=random_seed,
                                   is_output=True,
                                   data_degrees=data_degrees))
        layers.append(ElementwiseParams(num_params, mode='sequential'))

        super(MADE, self).__init__(*layers)


class AgnosticMADE(MADE):
    """Implementation of order/connectivity-agnostic MADE."""

    def __init__(self,
                 features,
                 num_params,
                 hidden_features,
                 order_agnostic=True,
                 connect_agnostic=True,
                 num_masks=16,
                 activation='relu',
                 dropout_prob=0.,
                 batch_norm=False):

        self.features = features
        self.order_agnostic = order_agnostic
        self.connect_agnostic = connect_agnostic
        self.num_masks = num_masks
        self.current_mask = 0
        super(AgnosticMADE, self).__init__(features=features,
                                           num_params=num_params,
                                           hidden_features=hidden_features,
                                           random_order=order_agnostic,
                                           random_mask=connect_agnostic,
                                           random_seed=self.current_mask,
                                           activation=activation,
                                           dropout_prob=dropout_prob,
                                           batch_norm=batch_norm)

    def update_masks(self):
        self.current_mask = (self.current_mask + 1) % self.num_masks

        # Get autoregressive order
        data_degrees = MaskedLinear.get_data_degrees(self.features, random_order=self.order_agnostic, random_seed=self.current_mask)

        # Get connectivity patterns
        in_degrees = copy.deepcopy(data_degrees)
        for module in self.modules():
            if isinstance(module, MaskedLinear):
                module.update_mask_and_degrees(in_degrees=in_degrees,
                                               data_degrees=data_degrees,
                                               random_mask=self.connect_agnostic,
                                               random_seed=self.current_mask)
                in_degrees = module.degrees

    def forward(self, x):
        if self.num_masks>1: self.update_masks()
        return super(AgnosticMADE, self).forward(x)
