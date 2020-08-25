import math
import torch
from survae.distributions import Distribution
from survae.transforms.surjections import Surjection


class SimpleMaxPoolSurjection2d(Surjection):
    '''
    An max pooling layer.

    Args:
        decoder: Distribution, a distribution of shape (3*c, h//2, w//2) with non-negative elements.
    '''

    stochastic_forward = False

    def __init__(self, decoder):
        super(SimpleMaxPoolSurjection2d, self).__init__()
        assert isinstance(decoder, Distribution)
        self.decoder = decoder

    def _squeeze(self, x):
        b,c,h,w = x.shape
        t = x.view(b, c, h // 2, 2, w // 2, 2)
        t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
        xr = t.view(b, c, h // 2, w // 2, 4)
        return xr

    def _unsqueeze(self, xr):
        b,c,h,w,_ = xr.shape
        t = xr.view(b, c, h, w, 2, 2)
        t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = t.view(b, c, h * 2, w * 2)
        return x

    def _k_mask(self, k):
        idx_all = torch.arange(4).view(1,1,4).expand(k.shape+(4,)).to(k.device)
        mask=k.unsqueeze(-1).expand_as(idx_all)==idx_all
        return mask

    def _deconstruct_x(self, x):
        xs = self._squeeze(x)
        z, k = xs.max(-1)
        mask = self._k_mask(k)
        xr = xs[~mask].view(k.shape+(3,))
        xds = z.unsqueeze(-1)-xr
        b,c,h,w,_ = xds.shape
        xd = xds.permute(0,1,4,2,3).reshape(b,3*c,h,w) # (B,C,H,W,3)->(B,3*C,H,W)
        return z, xd, k

    def _construct_x(self, z, xd, k):
        b,c,h,w = xd.shape
        xds = xd.reshape(b,c//3,3,h,w).permute(0,1,3,4,2) # (B,3*C,H,W)->(B,C,H,W,3)
        xr = z.unsqueeze(-1)-xds
        mask = self._k_mask(k)
        xs = z.new_zeros(z.shape+(4,))
        xs.masked_scatter_(mask, z)
        xs.masked_scatter_(~mask, xr)
        x = self._unsqueeze(xs)
        return x

    def forward(self, x):
        z, xd, k = self._deconstruct_x(x)
        ldj_k = - math.log(4) * z.shape[1:].numel()
        ldj = self.decoder.log_prob(xd) + ldj_k
        return z, ldj

    def inverse(self, z):
        k = torch.randint(0, 4, z.shape, device=z.device)
        xd = self.decoder.sample(z.shape[0])
        x = self._construct_x(z, xd, k)
        return x
