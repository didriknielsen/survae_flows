import os
import torch
import argparse

# Plot
import matplotlib.pyplot as plt

# Data
from data import get_data, dataset_choices

# Model
import torch.nn as nn
from survae.flows import Flow
from survae.distributions import StandardUniform
from survae.nn.nets import MLP
from layers import ElementAbsSurjection, ShiftBijection, ScaleBijection

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

classifier = MLP(2, 1,
                 hidden_units=args.hidden_units,
                 activation=args.activation,
                 out_lambda=lambda x: x.view(-1))

model = Flow(base_dist=StandardUniform((2,)),
             transforms=[
                ElementAbsSurjection(classifier=classifier),
                ShiftBijection(shift=torch.tensor([[0.0, 4.0]])),
                ScaleBijection(scale=torch.tensor([[1/4, 1/8]]))
                        ]).to(args.device)

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
with torch.no_grad():
    samples = model.sample(args.num_samples)
    samples = samples.cpu().numpy()
plt.figure(figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
plt.hist2d(samples[...,0], samples[...,1], bins=256, range=bounds)
plt.xlim(bounds[0])
plt.ylim(bounds[1])
plt.axis('off')
plt.savefig('figures/{}_abs_unif_samples.png'.format(args.dataset), bbox_inches = 'tight', pad_inches = 0)

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
print('Range:', logprobs.exp().min().item(), logprobs.exp().max().item())
plt.clim(0,0.05)
plt.savefig('figures/{}_abs_unif_density.png'.format(args.dataset), bbox_inches = 'tight', pad_inches = 0)

# Save log-likelihood
with open('figures/{}_abs_unif_loglik.txt'.format(args.dataset), 'w') as f:
    f.write(str(final_test_nats))

# Display plots
plt.show()
