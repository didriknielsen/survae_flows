# Data
from survae.data.datasets.toy import CheckerboardDataset
from torch.utils.data import DataLoader

# Model
import torch.nn as nn
from survae.flows import Flow
from survae.distributions import StandardNormal
from survae.transforms import AffineCouplingBijection, ActNormBijection, Reverse
from survae.nn.layers import ElementwiseParams

# Optim
from torch.optim import Adam

# Plot
import matplotlib.pyplot as plt

##########
## Data ##
##########

train = CheckerboardDataset(num_points=128*1000)
test = CheckerboardDataset(num_points=128*1000)
train_loader = DataLoader(train, batch_size=128, shuffle=False)
test_loader = DataLoader(test, batch_size=128, shuffle=True)

###########
## Model ##
###########

def net():
  return nn.Sequential(nn.Linear(1, 200), nn.ReLU(),
                       nn.Linear(200, 100), nn.ReLU(),
                       nn.Linear(100, 2), ElementwiseParams(2))

model = Flow(base_dist=StandardNormal((2,)),
             transforms=[
               AffineCouplingBijection(net()), ActNormBijection(2), Reverse(2),
               AffineCouplingBijection(net()), ActNormBijection(2), Reverse(2),
               AffineCouplingBijection(net()), ActNormBijection(2), Reverse(2),
               AffineCouplingBijection(net()), ActNormBijection(2),
             ])

###########
## Optim ##
###########

optimizer = Adam(model.parameters(), lr=1e-3)

###########
## Train ##
###########

print('Training...')
for epoch in range(10):
    l = 0.0
    for i, x in enumerate(train_loader):
        optimizer.zero_grad()
        loss = -model.log_prob(x).mean()
        loss.backward()
        optimizer.step()
        l += loss.detach().cpu().item()
        print('Epoch: {}/{}, Loglik: {:.3f}'.format(epoch+1, 10, l/(i+1)), end='\r')
    print('')

############
## Sample ##
############

print('Sampling...')
data = test.data.numpy()
samples = model.sample(100000).numpy()

fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].set_title('Data')
ax[0].hist2d(data[...,0], data[...,1], bins=256, range=[[-4, 4], [-4, 4]])
ax[0].set_xlim([-4, 4]); ax[0].set_ylim([-4, 4]); ax[0].axis('off')
ax[1].set_title('Samples')
ax[1].hist2d(samples[...,0], samples[...,1], bins=256, range=[[-4, 4], [-4, 4]])
ax[1].set_xlim([-4, 4]); ax[1].set_ylim([-4, 4]); ax[1].axis('off')
plt.tight_layout()
plt.show()
