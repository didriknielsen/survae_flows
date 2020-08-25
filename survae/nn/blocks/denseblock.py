import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth, dropout):
        super(DenseLayer, self).__init__()

        layers = []

        layers.extend([
            nn.Conv2d(in_channels, in_channels, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        ])

        if dropout > 0.:
            layers.append(nn.Dropout(p=dropout))

        layers.extend([
            nn.Conv2d(in_channels, growth, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ])

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        h = self.nn(x)
        h = torch.cat([x, h], dim=1)
        return h


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(GatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels * 3,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        h = self.conv(x)
        a, b, c = torch.chunk(h, chunks=3, dim=1)
        return a + b * torch.sigmoid(c)


class DenseBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, depth, growth,
                 dropout=0.0, gated_conv=False, zero_init=False):

        layers = [DenseLayer(in_channels+i*growth, growth, dropout) for i in range(depth)]

        if gated_conv:
            layers.append(GatedConv2d(in_channels+depth*growth, out_channels, kernel_size=1, padding=0))
        else:
            layers.append(nn.Conv2d(in_channels+depth*growth, out_channels, kernel_size=1, padding=0))

        if zero_init:
            nn.init.zeros_(layers[-1].weight)
            if hasattr(layers[-1], 'bias'):
                nn.init.zeros_(layers[-1].bias)

        super(DenseBlock, self).__init__(*layers)


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, growth,
                 dropout=0.0, gated_conv=False, zero_init=False):
        super(ResidualDenseBlock, self).__init__()

        self.dense = DenseBlock(in_channels=in_channels,
                                out_channels=out_channels,
                                depth=depth,
                                growth=growth,
                                dropout=dropout,
                                gated_conv=gated_conv,
                                zero_init=zero_init)

    def forward(self, x):
        return x + self.dense(x)
