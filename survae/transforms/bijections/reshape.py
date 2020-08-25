import torch
from survae.transforms.bijections import Bijection


class Reshape(Bijection):

    def __init__(self, input_shape, output_shape):
        super(Reshape, self).__init__()
        self.input_shape = torch.Size(input_shape)
        self.output_shape = torch.Size(output_shape)
        assert self.input_shape.numel() == self.output_shape.numel()

    def forward(self, x):
        batch_size = (x.shape[0],)
        z = x.reshape(batch_size + self.output_shape)
        ldj = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        batch_size = (z.shape[0],)
        x = z.reshape(batch_size + self.input_shape)
        return x
