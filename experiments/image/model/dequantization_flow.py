import torch
import torch.nn as nn
from survae.flows import ConditionalInverseFlow
from survae.distributions import ConvNormal2d
from survae.transforms import Unsqueeze2d, Sigmoid, Conv1x1
from survae.nn.layers import LambdaLayer
from survae.nn.blocks import DenseBlock
from .coupling import ConditionalCoupling


class DequantizationFlow(ConditionalInverseFlow):

    def __init__(self, data_shape, num_bits, num_steps, num_context,
                 num_blocks, mid_channels, depth, growth, dropout, gated_conv):

        context_net = nn.Sequential(LambdaLayer(lambda x: 2*x.float()/(2**num_bits-1)-1),
                                    DenseBlock(in_channels=data_shape[0],
                                               out_channels=mid_channels,
                                               depth=4,
                                               growth=16,
                                               dropout=dropout,
                                               gated_conv=gated_conv,
                                               zero_init=False),
                                    nn.Conv2d(mid_channels, mid_channels, kernel_size=2, stride=2, padding=0),
                                    DenseBlock(in_channels=mid_channels,
                                               out_channels=num_context,
                                               depth=4,
                                               growth=16,
                                               dropout=dropout,
                                               gated_conv=gated_conv,
                                               zero_init=False))



        transforms = []
        sample_shape = (data_shape[0] * 4, data_shape[1] // 2, data_shape[2] // 2)
        for i in range(num_steps):
            transforms.extend([
                Conv1x1(sample_shape[0]),
                ConditionalCoupling(in_channels=sample_shape[0],
                                    num_context=num_context,
                                    num_blocks=num_blocks,
                                    mid_channels=mid_channels,
                                    depth=depth,
                                    growth=growth,
                                    dropout=dropout,
                                    gated_conv=gated_conv)
            ])

        # Final shuffle of channels, squeeze and sigmoid
        transforms.extend([Conv1x1(sample_shape[0]),
                           Unsqueeze2d(),
                           Sigmoid()
                          ])

        super(DequantizationFlow, self).__init__(base_dist=ConvNormal2d(sample_shape),
                                                 transforms=transforms,
                                                 context_init=context_net)
