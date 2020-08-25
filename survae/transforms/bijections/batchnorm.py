import math
import torch
import torch.nn as nn
from survae.transforms.bijections import Bijection


class _BatchNormBijection(Bijection):
    '''
    Base class for invertible batch normalization [1].

    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    '''

    def __init__(self, num_features, momentum=0.1, epsilon=1e-5):
        super(_BatchNormBijection, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        self.register_params()

    @property
    def weight(self):
        return self.log_weight.exp()

    def forward(self, x):
        if self.training:
            mean, var = self.compute_stats(x)
            self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)
        else:
            mean, var = self.running_mean, self.running_var

        z = self.weight * ((x - mean) / torch.sqrt(var + self.epsilon)) + self.bias

        ldj = self.log_weight - 0.5 * torch.log(var + self.epsilon)
        ldj = ldj.sum().expand([x.shape[0]]) * self.ldj_multiplier(x)
        return z, ldj

    def inverse(self, z):
        if self.training:
            raise RuntimeError('BatchNorm inverse is only available in eval mode, not in training mode.')
        return torch.sqrt(self.running_var + self.epsilon) * ((z - self.bias) / self.weight) + self.running_mean

    def register_params(self):
        '''Register parameters'''
        raise NotImplementedError()

    def compute_stats(self, x):
        '''Compute mean and var'''
        raise NotImplementedError()

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        raise NotImplementedError()


class BatchNormBijection(_BatchNormBijection):
    '''
    Batch normalization [1] for inputs on the form (B,D).

    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    '''

    def register_params(self):
        '''Register parameters'''
        self.register_buffer('running_mean', torch.zeros(1, self.num_features))
        self.register_buffer('running_var', torch.ones(1, self.num_features))
        self.register_parameter('bias', nn.Parameter(torch.zeros(1, self.num_features)))
        self.register_parameter('log_weight', nn.Parameter(torch.zeros(1, self.num_features)))

    def compute_stats(self, x):
        '''Compute mean and var'''
        mean = torch.mean(x, dim=0, keepdim=True)
        var = torch.var(x, dim=0, keepdim=True)
        return mean, var

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        return 1


class BatchNormBijection1d(_BatchNormBijection):
    '''
    Batch normalization [1] for inputs on the form (B,C,L).

    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    '''

    def register_params(self):
        '''Register parameters'''
        self.register_buffer('running_mean', torch.zeros(1, self.num_features, 1))
        self.register_buffer('running_var', torch.ones(1, self.num_features, 1))
        self.register_parameter('bias', nn.Parameter(torch.zeros(1, self.num_features, 1)))
        self.register_parameter('log_weight', nn.Parameter(torch.zeros(1, self.num_features, 1)))

    def compute_stats(self, x):
        '''Compute mean and var'''
        mean = torch.mean(x, dim=[0, 2], keepdim=True)
        var = torch.var(x, dim=[0, 2], keepdim=True)
        return mean, var

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        return x.shape[2]


class BatchNormBijection2d(_BatchNormBijection):
    '''
    Batch normalization [1] for inputs on the form (B,C,H,W).

    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    '''

    def register_params(self):
        '''Register parameters'''
        self.register_buffer('running_mean', torch.zeros(1, self.num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, self.num_features, 1, 1))
        self.register_parameter('bias', nn.Parameter(torch.zeros(1, self.num_features, 1, 1)))
        self.register_parameter('log_weight', nn.Parameter(torch.zeros(1, self.num_features, 1, 1)))

    def compute_stats(self, x):
        '''Compute mean and var'''
        mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
        var = torch.var(x, dim=[0, 2, 3], keepdim=True)
        return mean, var

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        return x.shape[2:4].numel()
