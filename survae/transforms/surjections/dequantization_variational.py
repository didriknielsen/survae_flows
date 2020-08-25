import torch
from survae.distributions import ConditionalDistribution
from survae.transforms.surjections import Surjection


class VariationalDequantization(Surjection):
    '''
    A variational dequantization layer.
    This is useful for converting discrete variables to continuous [1, 2].

    Forward:
        `z = (x+u)/K, u~encoder(x)`
        where `x` is discrete, `x \in {0,1,2,...,K-1}^D`
        and `encoder` is a conditional distribution.
    Inverse:
        `x = Quantize(z, K)`

    Args:
        encoder: ConditionalDistribution, a conditional distribution/flow which
            outputs samples in `[0,1]^D` conditioned on `x`.
        num_bits: int, number of bits in quantization,
            i.e. 8 for `x \in {0,1,2,...,255}^D`
            or 5 for `x \in {0,1,2,...,31}^D`.

    References:
        [1] RNADE: The real-valued neural autoregressive density-estimator,
            Uria et al., 2013, https://arxiv.org/abs/1306.0186
        [2] Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design,
            Ho et al., 2019, https://arxiv.org/abs/1902.00275
    '''

    stochastic_forward = True

    def __init__(self, encoder, num_bits=8):
        super(VariationalDequantization, self).__init__()
        assert isinstance(encoder, ConditionalDistribution)
        self.num_bits = num_bits
        self.quantization_bins = 2**num_bits
        self.register_buffer('ldj_per_dim', -torch.log(torch.tensor(self.quantization_bins, dtype=torch.float)))
        self.encoder = encoder

    def _ldj(self, shape):
        batch_size = shape[0]
        num_dims = shape[1:].numel()
        ldj = self.ldj_per_dim * num_dims
        return ldj.repeat(batch_size)

    def forward(self, x):
        u, qu = self.encoder.sample_with_log_prob(context=x)
        z = (x.type(u.dtype) + u) / self.quantization_bins
        ldj = self._ldj(z.shape) - qu
        return z, ldj

    def inverse(self, z):
        z = self.quantization_bins * z
        return z.floor().clamp(min=0, max=self.quantization_bins-1).long()
