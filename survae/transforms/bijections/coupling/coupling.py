import torch
from survae.transforms.bijections import Bijection


class CouplingBijection(Bijection):
    """Transforms each input variable with an invertible elementwise bijection.

    This input variables are split in two parts. The second part is transformed conditioned on the first part.
    The coupling network takes the first part as input and outputs trasnformations for the second part.

    Args:
        coupling_net: nn.Module, a coupling network such that for x = [x1,x2]
            elementwise_params = coupling_net(x1)
        split_dim: int, dimension to split the input (default=1).
        num_condition: int or None, number of parameters to condition on.
            If None, the first half is conditioned on:
            - For even inputs (1,2,3,4), (1,2) will be conditioned on.
            - For odd inputs (1,2,3,4,5), (1,2,3) will be conditioned on.
    """

    def __init__(self, coupling_net, split_dim=1, num_condition=None):
        super(CouplingBijection, self).__init__()
        assert split_dim >= 1
        self.coupling_net = coupling_net
        self.split_dim = split_dim
        self.num_condition = num_condition

    def split_input(self, input):
        if self.num_condition:
            split_proportions = (self.num_condition, input.shape[self.split_dim] - self.num_condition)
            return torch.split(input, split_proportions, dim=self.split_dim)
        else:
            return torch.chunk(input, 2, dim=self.split_dim)

    def forward(self, x):
        id, x2 = self.split_input(x)
        elementwise_params = self.coupling_net(id)
        z2, ldj = self._elementwise_forward(x2, elementwise_params)
        z = torch.cat([id, z2], dim=self.split_dim)
        return z, ldj

    def inverse(self, z):
        with torch.no_grad():
            id, z2 = self.split_input(z)
            elementwise_params = self.coupling_net(id)
            x2 = self._elementwise_inverse(z2, elementwise_params)
            x = torch.cat([id, x2], dim=self.split_dim)
        return x

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, x, elementwise_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, z, elementwise_params):
        raise NotImplementedError()
