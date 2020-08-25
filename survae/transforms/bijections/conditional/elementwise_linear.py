import torch
from survae.transforms.bijections.conditional import ConditionalBijection
from survae.utils import sum_except_batch


class ConditionalAdditiveBijection(ConditionalBijection):
    """
    Computes `z = shift + x`, where `shift = net(context)`.
    """

    def __init__(self, context_net):
        super(ConditionalAdditiveBijection, self).__init__()
        self.context_net = context_net

    def forward(self, x, context):
        z = x + self.context_net(context)
        ldj = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        return z, ldj

    def inverse(self, z, context):
        x = z - self.context_net(context)
        return x


class ConditionalAffineBijection(ConditionalBijection):
    """
    Computes `z = shift + scale * x`, where `shift, log_scale = net(context)`.
    """

    def __init__(self, context_net, param_dim=1):
        super(ConditionalAffineBijection, self).__init__()
        self.context_net = context_net
        self.param_dim = param_dim

    def forward(self, x, context):
        params = self.context_net(context)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.param_dim)
        z = mean + log_std.exp() * x
        ldj = sum_except_batch(log_std)
        return z, ldj

    def inverse(self, z, context):
        params = self.context_net(context)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.param_dim)
        x = (z - mean) * torch.exp(-log_std)
        return x
