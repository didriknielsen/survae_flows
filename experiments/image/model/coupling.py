import torch
import torch.nn as nn
from survae.utils import sum_except_batch
from survae.transforms import AffineCouplingBijection, ConditionalAffineCouplingBijection
from survae.nn.nets import DenseNet
from survae.nn.layers import ElementwiseParams2d


class Coupling(AffineCouplingBijection):

    def __init__(self, in_channels, num_blocks, mid_channels, depth, growth, dropout, gated_conv):

        assert in_channels % 2 == 0

        net = nn.Sequential(DenseNet(in_channels=in_channels//2,
                                     out_channels=in_channels,
                                     num_blocks=num_blocks,
                                     mid_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=True),
                            ElementwiseParams2d(2, mode='sequential'))
        super(Coupling, self).__init__(coupling_net=net)

    def _elementwise_forward(self, x, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        z = shift + torch.exp(log_scale) * x
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        x = (z - shift) * torch.exp(-log_scale)
        return x


class ConditionalCoupling(ConditionalAffineCouplingBijection):

    def __init__(self, in_channels, num_context, num_blocks, mid_channels, depth, growth, dropout, gated_conv):

        assert in_channels % 2 == 0

        net = nn.Sequential(DenseNet(in_channels=in_channels//2+num_context,
                                     out_channels=in_channels,
                                     num_blocks=num_blocks,
                                     mid_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=True),
                            ElementwiseParams2d(2, mode='sequential'))
        super(ConditionalCoupling, self).__init__(coupling_net=net)

    def _elementwise_forward(self, x, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        z = shift + torch.exp(log_scale) * x
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        x = (z - shift) * torch.exp(-log_scale)
        return x
