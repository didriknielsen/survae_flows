import torch
import torch.nn as nn
from survae.flows import Flow
from survae.distributions import ConvNormal2d, StandardNormal, StandardHalfNormal
from survae.transforms import UniformDequantization, VariationalDequantization, ScalarAffineBijection, Squeeze2d, Conv1x1, Slice, SimpleMaxPoolSurjection2d, ActNormBijection2d
from .coupling import Coupling
from .dequantization_flow import DequantizationFlow


class PoolFlow(Flow):

    def __init__(self, data_shape, num_bits, num_scales, num_steps, actnorm, pooling,
                 dequant, dequant_steps, dequant_context,
                 densenet_blocks, densenet_channels, densenet_depth,
                 densenet_growth, dropout, gated_conv):

        transforms = []
        current_shape = data_shape
        if dequant == 'uniform':
            transforms.append(UniformDequantization(num_bits=num_bits))
        elif dequant == 'flow':
            dequantize_flow = DequantizationFlow(data_shape=data_shape,
                                                 num_bits=num_bits,
                                                 num_steps=dequant_steps,
                                                 num_context=dequant_context,
                                                 num_blocks=densenet_blocks,
                                                 mid_channels=densenet_channels,
                                                 depth=densenet_depth,
                                                 growth=densenet_growth,
                                                 dropout=dropout,
                                                 gated_conv=gated_conv)
            transforms.append(VariationalDequantization(encoder=dequantize_flow, num_bits=num_bits))

        # Change range from [0,1]^D to [-0.5, 0.5]^D
        transforms.append(ScalarAffineBijection(shift=-0.5))

        # Initial squeeze
        transforms.append(Squeeze2d())
        current_shape = (current_shape[0] * 4,
                         current_shape[1] // 2,
                         current_shape[2] // 2)

        # Pooling flows
        for scale in range(num_scales):
            for step in range(num_steps):
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))
                transforms.extend([
                    Conv1x1(current_shape[0]),
                    Coupling(in_channels=current_shape[0],
                             num_blocks=densenet_blocks,
                             mid_channels=densenet_channels,
                             depth=densenet_depth,
                             growth=densenet_growth,
                             dropout=dropout,
                             gated_conv=gated_conv)
                ])

            if scale < num_scales-1:
                noise_shape = (current_shape[0] * 3,
                               current_shape[1] // 2,
                               current_shape[2] // 2)
                if pooling=='none':
                    transforms.append(Squeeze2d())
                    transforms.append(Slice(StandardNormal(noise_shape), num_keep=current_shape[0], dim=1))
                elif pooling=='max':
                    decoder = StandardHalfNormal(noise_shape)
                    transforms.append(SimpleMaxPoolSurjection2d(decoder=decoder))
                current_shape = (current_shape[0],
                                 current_shape[1] // 2,
                                 current_shape[2] // 2)
            else:
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))

        super(PoolFlow, self).__init__(base_dist=ConvNormal2d(current_shape),
                                       transforms=transforms)
