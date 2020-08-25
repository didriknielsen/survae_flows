from .pool_flow import PoolFlow


def add_model_args(parser):

    # Flow params
    parser.add_argument('--num_scales', type=int, default=3)
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--actnorm', type=eval, default=False)
    parser.add_argument('--pooling', type=str, default='max', choices={'none', 'max'})

    # Dequant params
    parser.add_argument('--dequant', type=str, default='uniform', choices={'uniform', 'flow'})
    parser.add_argument('--dequant_steps', type=int, default=4)
    parser.add_argument('--dequant_context', type=int, default=32)

    # Net params
    parser.add_argument('--densenet_blocks', type=int, default=1)
    parser.add_argument('--densenet_channels', type=int, default=64)
    parser.add_argument('--densenet_depth', type=int, default=10)
    parser.add_argument('--densenet_growth', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gated_conv', type=eval, default=True)


def get_model_id(args):
    return 'pool_flow'


def get_model(args, data_shape):

    return PoolFlow(data_shape=data_shape,
                    num_bits=args.num_bits,
                    num_scales=args.num_scales,
                    num_steps=args.num_steps,
                    actnorm=args.actnorm,
                    pooling=args.pooling,
                    dequant=args.dequant,
                    dequant_steps=args.dequant_steps,
                    dequant_context=args.dequant_context,
                    densenet_blocks=args.densenet_blocks,
                    densenet_channels=args.densenet_channels,
                    densenet_depth=args.densenet_depth,
                    densenet_growth=args.densenet_growth,
                    dropout=args.dropout,
                    gated_conv=args.gated_conv)
