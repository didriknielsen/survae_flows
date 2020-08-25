import torch
import argparse

# Plot
import matplotlib.pyplot as plt

# Data
from data import get_data, dataset_choices

###########
## Setup ##
###########

parser = argparse.ArgumentParser()

# Data params
parser.add_argument('--dataset', type=str, default='spatial_mnist', choices=dataset_choices)

# Plotting params
parser.add_argument('--rowcol', type=int, default=8)
parser.add_argument('--pixels', type=int, default=1000)
parser.add_argument('--dpi', type=int, default=96)
# Check the DPI of your monitor at: https://www.infobyip.com/detectmonitordpi.php

args = parser.parse_args()
args.batch_size = args.rowcol**2

torch.manual_seed(0)

##################
## Specify data ##
##################

train_loader, valid_loader, test_loader = get_data(args)
x = next(iter(test_loader)).numpy()

##############
## Sampling ##
##############

if args.dataset in {'spatial_mnist'}:
    bounds = [[0, 28], [0, 28]]
else:
    raise NotImplementedError()

fig, ax = plt.subplots(args.rowcol,args.rowcol, figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
for i in range(args.rowcol):
    for j in range(args.rowcol):
        idx = i+args.rowcol*j
        ax[i][j].scatter(x[idx,:,0], x[idx,:,1])
        ax[i][j].set_xlim(bounds[0])
        ax[i][j].set_ylim(bounds[1])
        ax[i][j].axis('off')
plt.savefig('figures/{}.png'.format(args.dataset), bbox_inches = 'tight', pad_inches = 0)
plt.show()
