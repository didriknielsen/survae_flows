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
parser.add_argument('--dataset', type=str, default='checkerboard', choices=dataset_choices)
parser.add_argument('--samples', type=int, default=128*1000)

# Plotting params
parser.add_argument('--pixels', type=int, default=1000)
parser.add_argument('--dpi', type=int, default=96)
# Check the DPI of your monitor at: https://www.infobyip.com/detectmonitordpi.php

args = parser.parse_args()
args.train_samples = args.samples
args.test_samples = args.samples
args.batch_size = 128

torch.manual_seed(0)

##################
## Specify data ##
##################

_, test_loader = get_data(args)

##############
## Sampling ##
##############

test_data = test_loader.dataset.data.numpy()
if args.dataset == 'face_einstein':
    bounds = [[0, 1], [0, 1]]
else:
    bounds = [[-4, 4], [-4, 4]]

plt.figure(figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
plt.hist2d(test_data[...,0], test_data[...,1], bins=256, range=bounds)
plt.xlim(bounds[0])
plt.ylim(bounds[1])
plt.axis('off')
plt.savefig('figures/{}.png'.format(args.dataset), bbox_inches = 'tight', pad_inches = 0)
plt.show()
