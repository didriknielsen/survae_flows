import torch
from survae.transforms.bijections import Squeeze2d


class Unsqueeze2d(Squeeze2d):
    """
    A bijection defined for image data that trades channel dimensions for spatial
    dimensions, i.e. "unsqueezes" the inputs along the channel dimensions.
    Introduced in the RealNVP paper [1].

    Args:
        factor: int, the factor to squeeze by (default=2).
        ordered: bool, if True, squeezing happens imagewise.
                       if False, squeezing happens channelwise.
                       For more details, see example (default=False).

    Source implementation:
        Based on `squeeze_nxn`, `squeeze_2x2`, `squeeze_2x2_ordered`, `unsqueeze_2x2` in:
        https://github.com/laurent-dinh/models/blob/master/real_nvp/real_nvp_utils.py

    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    """

    def __init__(self, factor=2, ordered=False):
        super(Unsqueeze2d, self).__init__(factor=factor, ordered=ordered)

    def forward(self, x):
        z = self._unsqueeze(x)
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        x = self._squeeze(z)
        return x
