import numpy as np

from itreg.operators import LinearOperator


class PointwiseRealPart(LinearOperator):
    """The pointwise real part operator

    Parameters
    ----------
    domain : :class:`~itreg.spaces.Space`
        The domain on which the operator is defined.
    """

    def __init__(self, domain):
        codomain = domain.real_space()
        super().__init__(domain, codomain)

    def _eval(self, x):
        return x.real

    def _adjoint(self, y):
        return y
