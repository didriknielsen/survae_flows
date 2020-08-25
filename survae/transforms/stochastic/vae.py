import torch
from survae.transforms.stochastic import StochasticTransform


class VAE(StochasticTransform):
    '''
    A variational autoencoder [1, 2] layer.

    Args:
        decoder: ConditionalDistribution, the decoder p(x|z).
        encoder: ConditionalDistribution, the encoder q(z|x).

    References:
        [1] Auto-Encoding Variational Bayes,
            Kingma & Welling, 2013, https://arxiv.org/abs/1312.6114
        [2] Stochastic Backpropagation and Approximate Inference in Deep Generative Models,
            Rezende et al., 2014, https://arxiv.org/abs/1401.4082
    '''

    def __init__(self, decoder, encoder):
        super(VAE, self).__init__()
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x):
        z, log_qz = self.encoder.sample_with_log_prob(context=x)
        log_px = self.decoder.log_prob(x, context=z)
        return z, log_px - log_qz

    def inverse(self, z):
        return self.decoder.sample(context=z)
