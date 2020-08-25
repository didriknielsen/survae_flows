import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from survae.optim.schedulers import LinearWarmupScheduler

optim_choices = {'sgd', 'adam', 'adamax'}


def add_optim_args(parser):

    # Model params
    parser.add_argument('--optimizer', type=str, default='adam', choices=optim_choices)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--momentum_sqr', type=float, default=0.999)
    parser.add_argument('--gamma', type=float, default=0.995)


def get_optim_id(args):
    return 'expdecay'


def get_optim(args, model):
    assert args.optimizer in optim_choices

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr))
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr))

    if args.warmup is not None:
        scheduler_iter = LinearWarmupScheduler(optimizer, total_epoch=args.warmup)
    else:
        scheduler_iter = None

    scheduler_epoch = ExponentialLR(optimizer, gamma=args.gamma)


    return optimizer, scheduler_iter, scheduler_epoch
