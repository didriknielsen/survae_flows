import torch
from survae.transforms.bijections import Bijection


class ScalarAffineBijection(Bijection):
    """
    Computes `z = shift + scale * x`, where `scale` and `shift` are scalars, and `scale` is non-zero.
    """

    def __init__(self, shift=None, scale=None):
        super(ScalarAffineBijection, self).__init__()
        assert isinstance(shift, float) or shift is None, 'shift must be a float or None'
        assert isinstance(scale, float) or scale is None, 'scale must be a float or None'

        if shift is None and scale is None:
            raise ValueError('At least one of scale and shift must be provided.')
        if scale == 0.:
            raise ValueError('Scale` cannot be zero.')

        self.register_buffer('_shift', torch.tensor(shift if (shift is not None) else 0.))
        self.register_buffer('_scale', torch.tensor(scale if (scale is not None) else 1.))

    @property
    def _log_scale(self):
        return torch.log(torch.abs(self._scale))

    def forward(self, x):
        batch_size = x.shape[0]
        num_dims = x.shape[1:].numel()
        z = x * self._scale + self._shift
        ldj = torch.full([batch_size], self._log_scale * num_dims, device=x.device, dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        batch_size = z.shape[0]
        num_dims = z.shape[1:].numel()
        x = (z - self._shift) / self._scale
        return x
