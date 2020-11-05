import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.nn.layers import LambdaLayer
from survae.nn.layers import act_module


class MLP(nn.Sequential):
    def __init__(self, input_size, output_size, hidden_units, activation='relu', in_lambda=None, out_lambda=None):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            input_size: (int): write your description
            output_size: (int): write your description
            hidden_units: (int): write your description
            activation: (str): write your description
            in_lambda: (todo): write your description
            out_lambda: (float): write your description
        """
        layers = []
        if in_lambda: layers.append(LambdaLayer(in_lambda))
        for in_size, out_size in zip([input_size] + hidden_units[:-1], hidden_units):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(act_module(activation))
        layers.append(nn.Linear(hidden_units[-1], output_size))
        if out_lambda: layers.append(LambdaLayer(out_lambda))

        super(MLP, self).__init__(*layers)
