import torch
from survae.transforms.bijections import Bijection


class Rotate(Bijection):
    """
    Rotates inputs 90, 180 or 270 degrees around given dimensions dim1 and dim2.
    For input with shape (B,C,H,W), dim1=2, dim2=3 will rotate on (H,W).

    Args:
        degrees: int, shape (dim_size)
        dim1: int, dimension 1 to permute
        dim2: int, dimension 2 to permute
    """

    def __init__(self, degrees, dim1, dim2):
        super(Rotate, self).__init__()
        assert isinstance(degrees, int), 'degrees must be an integer'
        assert isinstance(dim1, int), 'dim1 must be an integer'
        assert isinstance(dim2, int), 'dim2 must be an integer'
        assert degrees in {90,180,270}
        assert dim1 != 0
        assert dim2 != 0
        assert dim1 != dim2

        self.degrees = degrees
        self.dim1 = dim1
        self.dim2 = dim2

    def _rotate90(self, x):
        return x.transpose(self.dim1, self.dim2).flip(self.dim1)

    def _rotate90_inv(self, z):
        return z.flip(self.dim1).transpose(self.dim1, self.dim2)

    def _rotate180(self, x):
        return x.flip(self.dim1).flip(self.dim2)

    def _rotate180_inv(self, z):
        return z.flip(self.dim2).flip(self.dim1)

    def _rotate270(self, x):
        return x.transpose(self.dim1, self.dim2).flip(self.dim2)

    def _rotate270_inv(self, z):
        return z.flip(self.dim2).transpose(self.dim1, self.dim2)

    def forward(self, x):
        if self.degrees == 90: x = self._rotate90(x)
        elif self.degrees == 180: x = self._rotate180(x)
        elif self.degrees == 270: x = self._rotate270(x)
        return x, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def inverse(self, z):
        if self.degrees == 90: z = self._rotate90_inv(z)
        elif self.degrees == 180: z = self._rotate180_inv(z)
        elif self.degrees == 270: z = self._rotate270_inv(z)
        return z
