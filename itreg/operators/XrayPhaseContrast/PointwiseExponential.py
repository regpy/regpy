import numpy as np

from itreg.operators import NonlinearOperator


class PointwiseExponential(NonlinearOperator):
    """The pointwise exponential operator exp.

    Parameters
    ----------
    domain : :class:`~itreg.spaces.Space`
        The domain on which the operator is defined.

    Notes
    -----
    The pointwise exponential operator :math:`exp` is defined as

    .. math:: exp(f)(x) = exp(f(x)).

    Its discrete form is 

    .. math:: exp(x)_i = exp(x_i).
    """

    def __init__(self, domain):
        super().__init__(domain, domain)

    def _eval(self, x, differentiate=False):
        if differentiate:
            self._exponential_factor = np.exp(x)
            return self._exponential_factor
        return np.exp(x)

    def _derivative(self, x):
        return self._exponential_factor * x

    def _adjoint(self, y):
        return self._exponential_factor.conj() * y
