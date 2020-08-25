import torch
import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        if lambd is None: lambd = lambda x: x
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
