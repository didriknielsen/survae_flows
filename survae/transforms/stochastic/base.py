from survae.transforms import Transform


class StochasticTransform(Transform):
    """Base class for StochasticTransform"""

    has_inverse = True
    bijective = False
    stochastic_forward = True
    stochastic_inverse = True
