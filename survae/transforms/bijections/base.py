from survae.transforms import Transform


class Bijection(Transform):
    """Base class for Bijection"""

    bijective = True
    stochastic_forward = False
    stochastic_inverse = False
    lower_bound = False
