import torch
import time
import pickle
import argparse
from utils import set_seeds

# Exp
from experiment.flow import FlowExperiment, add_exp_args

# Data
from data.data import get_data, get_data_id, add_data_args

# Model
from model.model import get_model, get_model_id, add_model_args

# Optim
from optim.base import get_optim, get_optim_id, add_optim_args

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--new_epochs', type=int)
parser.add_argument('--new_lr', type=float)
more_args = parser.parse_args()

path_args = '{}/args.pickle'.format(more_args.model)
path_check = '{}/check/checkpoint.pt'.format(more_args.model)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

# Adjust args
args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
args.epochs = more_args.new_epochs
args.lr = more_args.new_lr
args.resume = None

# Store more_args
args.start_model = more_args.model
args.new_epochs = more_args.new_epochs
args.new_lr = more_args.new_lr

##################
## Specify data ##
##################

train_loader, eval_loader, data_shape = get_data(args)
data_id = get_data_id(args)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)
model_id = get_model_id(args)

#######################
## Specify optimizer ##
#######################

optimizer, _, _ = get_optim(args, model)
optim_id = 'more'

##############
## Training ##
##############

exp = FlowExperiment(args=args,
                     data_id=data_id,
                     model_id=model_id,
                     optim_id=optim_id,
                     train_loader=train_loader,
                     eval_loader=eval_loader,
                     model=model,
                     optimizer=optimizer,
                     scheduler_iter=None,
                     scheduler_epoch=None)

# Load checkpoint
exp.checkpoint_load('{}/check/'.format(more_args.model))

# Adjust lr
for param_group in exp.optimizer.param_groups:
    param_group['lr'] = more_args.new_lr

exp.run()
