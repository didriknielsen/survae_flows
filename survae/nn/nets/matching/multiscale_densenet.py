import torch
import torch.nn as nn
from survae.nn.nets.matching import DenseNet


class MultiscaleDenseNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales,
                 num_blocks, mid_channels, depth, growth, dropout,
                 gated_conv=False, zero_init=False):
        super(MultiscaleDenseNet, self).__init__()
        assert num_scales > 1
        self.num_scales = num_scales

        def get_densenet(cin, cout, zinit=False):
            return DenseNet(in_channels=cin,
                            out_channels=cout,
                            num_blocks=num_blocks,
                            mid_channels=mid_channels,
                            depth=depth,
                            growth=growth,
                            dropout=dropout,
                            gated_conv=gated_conv,
                            zero_init=zinit)

        # Down in
        self.down_in = get_densenet(in_channels, mid_channels)

        # Down
        down = []
        for i in range(num_scales - 1):
            down.append(nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=2, padding=0, stride=2),
                                      get_densenet(mid_channels, mid_channels)))
        self.down = nn.ModuleList(down)

        # Up
        up = []
        for i in range(num_scales - 1):
            up.append(nn.Sequential(get_densenet(mid_channels, mid_channels),
                                    nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, padding=0, stride=2)))
        self.up = nn.ModuleList(up)

        # Up out
        self.up_out = get_densenet(mid_channels, out_channels, zinit=zero_init)

    def forward(self, x):

        # Down in
        d = [self.down_in(x)]

        # Down
        for down_layer in self.down:
            d.append(down_layer(d[-1]))

        # Up
        u = [d[-1]]
        for i, up_layer in enumerate(self.up):
            u.append(up_layer(u[-1]) + d[self.num_scales - 2 - i]) #

        # Up out
        return self.up_out(u[-1])
