from survae.transforms import Transform


class Surjection(Transform):
    """Base class for Surjection"""

    bijective = False

    @property
    def stochastic_forward(self):
        raise NotImplementedError()

    @property
    def stochastic_inverse(self):
        return not self.stochastic_forward
