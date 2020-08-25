import torch
from torch.distributions import Categorical
from survae.distributions.conditional import ConditionalDistribution
from survae.utils import sum_except_batch


class ConditionalCategorical(ConditionalDistribution):
    """A Categorical distribution with conditional logits."""

    def __init__(self, net):
        super(ConditionalCategorical, self).__init__()
        self.net = net

    def cond_dist(self, context):
        logits = self.net(context)
        return Categorical(logits=logits)

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.sample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.sample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def logits(self, context):
        return self.cond_dist(context).logits

    def probs(self, context):
        return self.cond_dist(context).probs

    def mode(self, context):
        return self.cond_dist(context).logits.argmax(-1)
