from survae.transforms import ConditionalTransform


class ConditionalBijection(ConditionalTransform):
    """Base class for ConditionalBijection"""

    bijective = True
    stochastic_forward = False
    stochastic_inverse = False
    lower_bound = False
