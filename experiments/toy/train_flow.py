import os
import torch
import argparse

# Plot
import matplotlib.pyplot as plt
from utils import get_args_table

# Data
from data import get_data, dataset_choices

# Model
import torch.nn as nn
from survae.flows import Flow
from survae.distributions import StandardNormal
from survae.transforms import AdditiveCouplingBijection, AffineCouplingBijection, ActNormBijection, Reverse
from survae.nn.nets import MLP
from survae.nn.layers import ElementwiseParams, scale_fn

# Optim
from torch.optim import Adam, Adamax

###########
## Setup ##
###########

parser = argparse.ArgumentParser()

# Data params
parser.add_argument('--dataset', type=str, default='checkerboard', choices=dataset_choices)
parser.add_argument('--train_samples', type=int, default=128*1000)
parser.add_argument('--test_samples', type=int, default=128*1000)

# Model params
parser.add_argument('--num_flows', type=int, default=4)
parser.add_argument('--actnorm', type=eval, default=False)
parser.add_argument('--affine', type=eval, default=True)
parser.add_argument('--scale_fn', type=str, default='exp', choices={'exp', 'softplus', 'sigmoid', 'tanh_exp'})
parser.add_argument('--hidden_units', type=eval, default=[50])
parser.add_argument('--activation', type=str, default='relu', choices={'relu', 'elu', 'gelu'})

# Train params
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='adam', choices={'adam', 'adamax'})
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cpu')

# Plot params
parser.add_argument('--num_samples', type=int, default=128*1000)
parser.add_argument('--grid_size', type=int, default=500)
parser.add_argument('--pixels', type=int, default=1000)
parser.add_argument('--dpi', type=int, default=96)
parser.add_argument('--clim', type=float, default=0.05)
# Check the DPI of your monitor at: https://www.infobyip.com/detectmonitordpi.php

args = parser.parse_args()

torch.manual_seed(0)
if not os.path.exists('figures'): os.mkdir('figures')

##################
## Specify data ##
##################

train_loader, test_loader = get_data(args)

###################
## Specify model ##
###################

D = 2 # Number of data dimensions
P = 2 if args.affine else 1 # Number of elementwise parameters

transforms = []
for _ in range(args.num_flows):
    net = nn.Sequential(MLP(1, P,
                            hidden_units=args.hidden_units,
                            activation=args.activation),
                        ElementwiseParams(P))
    if args.affine: transforms.append(AffineCouplingBijection(net, scale_fn=scale_fn(args.scale_fn)))
    else:           transforms.append(AdditiveCouplingBijection(net))
    if args.actnorm: transforms.append(ActNormBijection(D))
    transforms.append(Reverse(D))
transforms.pop()


model = Flow(base_dist=StandardNormal((2,)),
             transforms=transforms).to(args.device)

#######################
## Specify optimizer ##
#######################

if args.optimizer == 'adam':
    optimizer = Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'adamax':
    optimizer = Adamax(model.parameters(), lr=args.lr)

##############
## Training ##
##############

print('Training...')
for epoch in range(args.epochs):
    loss_sum = 0.0
    for i, x in enumerate(train_loader):
        optimizer.zero_grad()
        loss = -model.log_prob(x.to(args.device)).mean()
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().cpu().item()
        print('Epoch: {}/{}, Iter: {}/{}, Nats: {:.3f}'.format(epoch+1, args.epochs, i+1, len(train_loader), loss_sum/(i+1)), end='\r')
    print('')
final_train_bpd = loss_sum / len(train_loader)

#############
## Testing ##
#############

print('Testing...')
with torch.no_grad():
    loss_sum = 0.0
    for i, x in enumerate(test_loader):
        loss = -model.log_prob(x.to(args.device)).mean()
        loss_sum += loss.detach().cpu().item()
        print('Iter: {}/{}, Nats: {:.3f}'.format(i+1, len(test_loader), loss_sum/(i+1)), end='\r')
    print('')
final_test_nats = loss_sum / len(test_loader)

##############
## Sampling ##
##############

print('Sampling...')
if args.dataset == 'face_einstein':
    bounds = [[0, 1], [0, 1]]
else:
    bounds = [[-4, 4], [-4, 4]]

# Plot samples
samples = model.sample(args.num_samples)
samples = samples.cpu().numpy()
plt.figure(figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
plt.hist2d(samples[...,0], samples[...,1], bins=256, range=bounds)
plt.xlim(bounds[0])
plt.ylim(bounds[1])
plt.axis('off')
plt.savefig('figures/{}_flow_samples.png'.format(args.dataset), bbox_inches = 'tight', pad_inches = 0)

# Plot density
xv, yv = torch.meshgrid([torch.linspace(bounds[0][0], bounds[0][1], args.grid_size), torch.linspace(bounds[1][0], bounds[1][1], args.grid_size)])
x = torch.cat([xv.reshape(-1,1), yv.reshape(-1,1)], dim=-1)
with torch.no_grad():
    logprobs = model.log_prob(x)
plt.figure(figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
plt.pcolormesh(xv, yv, logprobs.exp().reshape(xv.shape))
plt.xlim(bounds[0])
plt.ylim(bounds[1])
plt.axis('off')
print('Range:', logprobs.exp().min().numpy(), logprobs.exp().max().numpy())
print('Limits:', 0.0, args.clim)
plt.clim(0,args.clim)
plt.savefig('figures/{}_flow_density.png'.format(args.dataset), bbox_inches = 'tight', pad_inches = 0)

# Save log-likelihood
with open('figures/{}_flow_loglik.txt'.format(args.dataset), 'w') as f:
    f.write(str(final_test_nats))

# Save args
args_table = get_args_table(vars(args))
with open('figures/{}_flow_args.txt'.format(args.dataset), 'w') as f:
    f.write(str(args_table))

# Display plots
plt.show()
