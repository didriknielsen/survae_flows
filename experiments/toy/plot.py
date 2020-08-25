import argparse

import matplotlib.pyplot as plt
from data import get_data, dataset_choices

###########
## Setup ##
###########

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='checkerboard', choices=dataset_choices)
parser.add_argument('--train_samples', type=int, default=128000)
parser.add_argument('--test_samples', type=int, default=128000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--bounds', type=float, default=4.0)

args = parser.parse_args()

###############
## Load data ##
###############

print('Loading data...')
train_loader = get_data(args)[0]
samples = train_loader.dataset.data.cpu().numpy()
print('samples.shape:', samples.shape)

###############
## Plot data ##
###############

print('Plotting data...')
if args.dataset == 'face_einstein':
    bounds = [[0, 1], [0, 1]]
else:
    bounds = [[-args.bounds, args.bounds], [-args.bounds, args.bounds]]
plt.hist2d(samples[...,0], samples[...,1], bins=256, range=bounds)
plt.xlim(bounds[0])
plt.ylim(bounds[1])
plt.show()
