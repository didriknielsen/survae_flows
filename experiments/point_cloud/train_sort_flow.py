import os
import torch
import argparse

# Plot
import matplotlib.pyplot as plt
from utils import get_args_table
from torch.utils.tensorboard import SummaryWriter

# Data
from data import get_data, dataset_choices

# Model
import torch.nn as nn
from survae.flows import Flow
from survae.distributions import StandardNormal
from survae.transforms import AdditiveCouplingBijection, AffineCouplingBijection, ActNormBijection, ActNormBijection1d, Reverse, Shuffle, PermuteAxes, Reshape, ScalarAffineBijection, SimpleSortSurjection
from survae.nn.layers import ElementwiseParams1d, scale_fn
from nets import DenseTransformer, PositionalDenseTransformer

# Optim
from torch.optim import Adam, Adamax
from torch.optim.lr_scheduler import ExponentialLR
from survae.optim.schedulers import LinearWarmupScheduler

###########
## Setup ##
###########

parser = argparse.ArgumentParser()

# Data params
parser.add_argument('--dataset', type=str, default='spatial_mnist', choices=dataset_choices)

# Model params
parser.add_argument('--name', type=str, default='')
parser.add_argument('--dimwise', type=eval, default=True)
parser.add_argument('--lenwise', type=eval, default=True)
parser.add_argument('--num_flows', type=int, default=16)
parser.add_argument('--actnorm', type=eval, default=True)
parser.add_argument('--affine', type=eval, default=True)
parser.add_argument('--scale_fn', type=str, default='softplus', choices={'exp', 'softplus', 'sigmoid', 'tanh_exp'})

parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--activation', type=str, default='gelu', choices={'relu', 'elu', 'gelu'})
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--checkpoint_blocks', type=eval, default=False)

# Train params
parser.add_argument('--train', type=eval, default=True)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='adam', choices={'adam', 'adamax'})
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--valid_every', type=int, default=10)
parser.add_argument('--warmup', type=int, default=None)
parser.add_argument('--gamma', type=float, default=None)

# Plot params
parser.add_argument('--rowcol', type=int, default=8)
parser.add_argument('--pixels', type=int, default=1000)
parser.add_argument('--dpi', type=int, default=96)

args = parser.parse_args()
assert args.epochs % args.valid_every == 0
run_name = '{}_sort_flow_{}'.format(args.dataset, args.name)
if args.train:
    tb_writer_path = os.path.join('tb', run_name)
    assert not os.path.exists(tb_writer_path), 'Run by name "{}" already stored'.format(args.name)

torch.manual_seed(0)
if not os.path.exists('models'): os.makedirs('models')
if not os.path.exists('results'): os.makedirs('results')
if not os.path.exists('figures'): os.makedirs('figures')
if not os.path.exists('tb'): os.makedirs('tb')

##################
## Specify data ##
##################

train_loader, valid_loader, test_loader = get_data(args)

###################
## Specify model ##
###################

transforms=[
    PermuteAxes((0,2,1)), # (B, 50, 2) -> (B, 2, 50)
    ScalarAffineBijection(scale=1/28, shift=-0.5),
    SimpleSortSurjection(dim=2, lambd=lambda x: x[:,0,:]),
]

D = 2 # Number of data dimensions
L = 50 # Number of points
P = 2 if args.affine else 1 # Number of elementwise parameters

def dimwise(transforms):
    net = nn.Sequential(DenseTransformer(d_input=D//2, d_output=P*D//2, d_model=args.d_model, nhead=args.nhead,
                                         num_layers=args.num_layers, dim_feedforward=4*args.d_model,
                                         dropout=args.dropout, activation=args.activation,
                                         checkpoint_blocks=args.checkpoint_blocks),
                        ElementwiseParams1d(P))
    if args.affine: transforms.append(AffineCouplingBijection(net, split_dim=1, scale_fn=scale_fn(args.scale_fn)))
    else:           transforms.append(AdditiveCouplingBijection(net, split_dim=1))
    transforms.append(Reverse(D, dim=1))
    return transforms

def lenwise(transforms):
    net = nn.Sequential(PositionalDenseTransformer(l_input=L//2, d_input=D, d_output=P*D, d_model=args.d_model, nhead=args.nhead,
                                                   num_layers=args.num_layers, dim_feedforward=4*args.d_model,
                                                   dropout=args.dropout, activation=args.activation,
                                                   checkpoint_blocks=args.checkpoint_blocks),
                        ElementwiseParams1d(P))
    if args.affine: transforms.append(AffineCouplingBijection(net, split_dim=2, scale_fn=scale_fn(args.scale_fn)))
    else:           transforms.append(AdditiveCouplingBijection(net, split_dim=2))
    transforms.append(Shuffle(L, dim=2))
    return transforms

for _ in range(args.num_flows):
    if args.dimwise: transforms = dimwise(transforms)
    if args.lenwise: transforms = lenwise(transforms)
    if args.actnorm: transforms.append(ActNormBijection1d(2))



model = Flow(base_dist=StandardNormal((D,L)),
             transforms=transforms).to(args.device)
if not args.train:
    state_dict = torch.load('models/{}.pt'.format(run_name))
    model.load_state_dict(state_dict)


#######################
## Specify optimizer ##
#######################

if args.optimizer == 'adam':
    optimizer = Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'adamax':
    optimizer = Adamax(model.parameters(), lr=args.lr)

if args.warmup is not None:
    scheduler_iter = LinearWarmupScheduler(optimizer, total_epoch=args.warmup)
else:
    scheduler_iter = None

if args.gamma is not None:
    scheduler_epoch = ExponentialLR(optimizer, gamma=args.gamma)
else:
    scheduler_epoch = None

#####################
## Define training ##
#####################

def train(model, train_loader, epoch):
    model = model.train()
    loss_sum = 0.0
    for i, x in enumerate(train_loader):
        optimizer.zero_grad()
        loss = -model.log_prob(x.to(args.device)).mean()
        loss.backward()
        optimizer.step()
        if scheduler_iter: scheduler_iter.step()
        loss_sum += -loss.detach().cpu().item() / 50
        print('Epoch: {}/{}, Iter: {}/{}, PPLL: {:.3f}'.format(epoch+1, args.epochs, i+1, len(train_loader), loss_sum/(i+1)), end='\r')
    print('')
    if scheduler_epoch: scheduler_epoch.step()
    return loss_sum / len(train_loader)

def evaluate(model, eval_loader):
    model = model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        for i, x in enumerate(eval_loader):
            loss = -model.log_prob(x.to(args.device)).mean()
            loss_sum += -loss.detach().cpu().item() / 50
            print('Eval, Iter: {}/{}, PPLL: {:.3f}'.format(i+1, len(eval_loader), loss_sum/(i+1)), end='\r')
        print('')
    return loss_sum / len(eval_loader)

##############
## Training ##
##############

if args.train:
    # Train
    print('Training...')
    writer = SummaryWriter(tb_writer_path)
    for epoch in range(args.epochs):
        train_ppll = train(model, train_loader, epoch=epoch)
        writer.add_scalar('train_ppll', train_ppll, global_step=epoch+1)
        if (epoch+1) % args.valid_every == 0:
            valid_ppll = evaluate(model, valid_loader)
            writer.add_scalar('valid_ppll', valid_ppll, global_step=epoch+1)

    # Save log-likelihood
    with open('results/{}_valid_loglik.txt'.format(run_name), 'w') as f:
        f.write(str(valid_ppll))

    # Save args
    args_table = get_args_table(vars(args))
    with open('results/{}_args.txt'.format(run_name), 'w') as f:
        f.write(str(args_table))

    # Save model
    state_dict = model.state_dict()
    torch.save(state_dict, 'models/{}.pt'.format(run_name))

##########
## Test ##
##########

# Test
test_ppll = evaluate(model, test_loader)
if args.train:
    writer.add_scalar('test_ppll', test_ppll, global_step=epoch+1)

# Save log-likelihood
with open('results/{}_test_loglik.txt'.format(run_name), 'w') as f:
    f.write(str(test_ppll))

##############
## Sampling ##
##############

if args.dataset in {'spatial_mnist'}:
    bounds = [[0, 28], [0, 28]]
else:
    raise NotImplementedError()

model = model.eval()
samples = model.sample(args.rowcol**2)
samples = samples.cpu().numpy()
fig, ax = plt.subplots(args.rowcol, args.rowcol, figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
for i in range(args.rowcol):
    for j in range(args.rowcol):
        idx = i+args.rowcol*j
        ax[i][j].scatter(samples[idx,:,0], samples[idx,:,1])
        ax[i][j].set_xlim(bounds[0])
        ax[i][j].set_ylim(bounds[1])
        ax[i][j].axis('off')
plt.savefig('figures/{}.png'.format(run_name), bbox_inches = 'tight', pad_inches = 0)
plt.show()
