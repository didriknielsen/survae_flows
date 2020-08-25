import torch
from survae.utils import sum_except_batch
from survae.distributions import ConditionalDistribution
from survae.transforms.surjections import Surjection


class Augment(Surjection):
    '''
    A simple augmentation layer which augments the input with additional elements.
    This is useful for constructing augmented normalizing flows [1, 2].

    References:
        [1] Augmented Normalizing Flows: Bridging the Gap Between Generative Flows and Latent Variable Models,
            Huang et al., 2020, https://arxiv.org/abs/2002.07101
        [2] VFlow: More Expressive Generative Flows with Variational Data Augmentation,
            Chen et al., 2020, https://arxiv.org/abs/2002.09741
    '''
    stochastic_forward = True

    def __init__(self, encoder, x_size, split_dim=1):
        super(Augment, self).__init__()
        assert split_dim >= 1
        self.encoder = encoder
        self.split_dim = split_dim
        self.x_size = x_size
        self.cond = isinstance(self.encoder, ConditionalDistribution)

    def split_z(self, z):
        split_proportions = (self.x_size, z.shape[self.split_dim] - self.x_size)
        return torch.split(z, split_proportions, dim=self.split_dim)

    def forward(self, x):
        if self.cond: z2, logqz2 = self.encoder.sample_with_log_prob(context=x)
        else:         z2, logqz2 = self.encoder.sample_with_log_prob(num_samples=x.shape[0])
        z = torch.cat([x, z2], dim=self.split_dim)
        ldj = -logqz2
        return z, ldj

    def inverse(self, z):
        x, z2 = self.split_z(z)
        return x
