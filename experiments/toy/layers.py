import math
import torch
import torch.nn.functional as F
from survae.transforms import Surjection, Bijection, StochasticTransform
from survae.utils import sum_except_batch


class ElementAbsSurjection(Surjection):
    stochastic_forward = False

    def __init__(self, classifier, element=0):
        super(ElementAbsSurjection, self).__init__()
        self.classifier = classifier
        self.element = element

    def forward(self, x):
        s = (x[:, self.element].sign()+1)/2
        z = x
        z[:, self.element] = x[:, self.element].abs()
        logit_pi = self.classifier(z)
        ldj = sum_except_batch(-F.binary_cross_entropy_with_logits(logit_pi, s, reduction='none'))
        return z, ldj

    def inverse(self, z):
        logit_pi = self.classifier(z)
        s = torch.bernoulli(torch.sigmoid(logit_pi))
        x = z
        x[:, self.element] = (2*s-1)*x[:, self.element]
        return x


class ScaleBijection(Bijection):

    def __init__(self, scale):
        super(ScaleBijection, self).__init__()
        self.register_buffer('scale', scale)

    @property
    def log_scale(self):
        return torch.log(torch.abs(self.scale)).sum()

    def forward(self, x):
        z = x * self.scale
        ldj = x.new_ones(x.shape[0]) * self.log_scale
        return z, ldj

    def inverse(self, z):
        x = z / self.scale
        return x


class ShiftBijection(Bijection):

    def __init__(self, shift):
        super(ShiftBijection, self).__init__()
        self.register_buffer('shift', shift)

    def forward(self, x):
        z = x + self.shift
        ldj = x.new_zeros(x.shape[0])
        return z, ldj

    def inverse(self, z):
        x = z - self.shift
        return x
