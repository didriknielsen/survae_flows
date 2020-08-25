import torch
from survae.transforms.bijections import Bijection


class Squeeze2d(Bijection):
    """
    A bijection defined for image data that trades spatial dimensions for channel
    dimensions, i.e. "squeezes" the inputs along the channel dimensions.
    Introduced in the RealNVP paper [1].

    Args:
        factor: int, the factor to squeeze by (default=2).
        ordered: bool, if True, squeezing happens imagewise.
                       if False, squeezing happens channelwise.
                       For more details, see example (default=False).

    Source implementation:
        Based on `squeeze_nxn`, `squeeze_2x2`, `squeeze_2x2_ordered`, `unsqueeze_2x2` in:
        https://github.com/laurent-dinh/models/blob/master/real_nvp/real_nvp_utils.py

    Example:
        Input x of shape (1, 2, 4, 4):

        [[[[ 1  2  1  2]
           [ 3  4  3  4]
           [ 1  2  1  2]
           [ 3  4  3  4]]

          [[10 20 10 20]
           [30 40 30 40]
           [10 20 10 20]
           [30 40 30 40]]]]

        Standard output z of shape (1, 8, 2, 2):

        [[[[ 1  1]
           [ 1  1]]

          [[ 2  2]
           [ 2  2]]

          [[ 3  3]
           [ 3  3]]

          [[ 4  4]
           [ 4  4]]

          [[10 10]
           [10 10]]

          [[20 20]
           [20 20]]

          [[30 30]
           [30 30]]

          [[40 40]
           [40 40]]]]

        Ordered output z of shape (1, 8, 2, 2):

        [[[[ 1  1]
           [ 1  1]]

          [[10 10]
           [10 10]]

          [[ 4  4]
           [ 4  4]]

          [[40 40]
           [40 40]]

          [[ 2  2]
           [ 2  2]]

          [[20 20]
           [20 20]]

          [[ 3  3]
           [ 3  3]]

          [[30 30]
           [30 30]]]]

    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    """

    def __init__(self, factor=2, ordered=False):
        super(Squeeze2d, self).__init__()
        assert isinstance(factor, int)
        assert factor > 1
        self.factor = factor
        self.ordered = ordered

    def _squeeze(self, x):
        assert len(x.shape) == 4, 'Dimension should be 4, but was {}'.format(len(x.shape))
        batch_size, c, h, w = x.shape
        assert h % self.factor == 0, 'h = {} not multiplicative of {}'.format(h, self.factor)
        assert w % self.factor == 0, 'w = {} not multiplicative of {}'.format(w, self.factor)
        t = x.view(batch_size, c, h // self.factor, self.factor, w // self.factor, self.factor)
        if not self.ordered:
            t = t.permute(0, 1, 3, 5, 2, 4).contiguous()
        else:
            t = t.permute(0, 3, 5, 1, 2, 4).contiguous()
        z = t.view(batch_size, c * self.factor ** 2, h // self.factor, w // self.factor)
        return z

    def _unsqueeze(self, z):
        assert len(z.shape) == 4, 'Dimension should be 4, but was {}'.format(len(z.shape))
        batch_size, c, h, w = z.shape
        assert c % (self.factor ** 2) == 0, 'c = {} not multiplicative of {}'.format(c, self.factor ** 2)
        if not self.ordered:
            t = z.view(batch_size, c // self.factor ** 2, self.factor, self.factor, h, w)
            t = t.permute(0, 1, 4, 2, 5, 3).contiguous()
        else:
            t = z.view(batch_size, self.factor, self.factor, c // self.factor ** 2, h, w)
            t = t.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = t.view(batch_size, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x

    def forward(self, x):
        z = self._squeeze(x)
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        x = self._unsqueeze(z)
        return x
