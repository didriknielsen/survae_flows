import argparse

from data import get_data, add_data_args
import torchvision.utils as vutils
import matplotlib.pyplot as plt

###########
## Setup ##
###########

parser = argparse.ArgumentParser()

add_data_args(parser)
parser.add_argument('--num_cols', type=int, default=8)

args = parser.parse_args()

###############
## Load data ##
###############

print('Loading data...')
train_loader = get_data(args)[0]
images = next(iter(train_loader))

###############
## Plot data ##
###############

print('images.shape = {}'.format(images.shape))
print('Plotting data...')

images = images.float()/(2**args.num_bits - 1)
image_grid = vutils.make_grid(images, nrow=args.num_cols, padding=2)
image_grid = image_grid.permute([1,2,0]).detach().cpu().numpy()

plt.figure()
plt.imshow(image_grid)
plt.show()
