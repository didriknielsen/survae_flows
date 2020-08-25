import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.transforms.bijections import Bijection


class LinearLowRank(Bijection):
    """
    Linear bijection z = (D + UV)x, where D is diagonal,
    U.shape = (num_features, rank) and V.shape = (rank, num_features).

    Args:
        num_features: int, Number of features in the input and output.
        rank: int, the rank of the low-rank matrix.
        bias: bool, if True a bias is included (default=False).
    """
    def __init__(self, num_features, rank, bias=False):
        super(LinearLowRank, self).__init__()
        assert rank >= 1 and rank <= num_features, 'rank should be 1 <= rank <= num_features, but got rank {}'.format(rank)
        self.num_features = num_features
        self.rank = rank
        self.d = nn.Parameter(torch.Tensor(num_features))
        self.U = nn.Parameter(torch.Tensor(num_features, rank))
        self.V = nn.Parameter(torch.Tensor(rank, num_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.d, 1-0.001, 1+0.001)
        nn.init.uniform_(self.U, -0.001, 0.001)
        nn.init.uniform_(self.V, -0.001, 0.001)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def K(self):
        '''K = I + VD^{-1}U'''
        I  = torch.eye(self.rank, dtype=self.d.dtype, device=self.d.device)
        VDinvU = torch.einsum('vd,d,du->vu', self.V, 1/self.d, self.U)
        return I + VDinvU

    def forward(self, x):
        '''
        z = Dx + UV^Tx
        ldj = sum(log(abs(D))) + log(abs(det(K)))
        '''
        z = self.d * x + torch.einsum('dr,br->bd', self.U, torch.einsum('rd,bd->br', self.V, x))
        if self.bias is not None: z = z + self.bias
        ldj = self.d.abs().log().sum() + torch.slogdet(self.K)[1]
        ldj = ldj.expand([x.shape[0]])
        return z, ldj

    def inverse(self, z):
        '''x = D^{-1}z - D^{-1}UK^{-1}VD^{-1}z'''
        if self.bias is not None: z = z - self.bias
        VDiz = torch.einsum('rd,bd->br', self.V, z / self.d)
        KiVDiz = torch.solve(VDiz.t(), self.K)[0].t()
        UKiVDiz = torch.einsum('dr,br->bd', self.U, KiVDiz)
        x = (z - UKiVDiz) / self.d
        return x
