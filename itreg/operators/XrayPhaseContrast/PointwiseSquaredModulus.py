import numpy as np

from itreg.operators import NonlinearOperator


class PointwiseSquaredModulus(NonlinearOperator):
    """The pointwise squared modulus operator.

    Parameters
    ----------
    domain : :class:`~itreg.spaces.Space`
        The domain on which the operator is defined.

    Notes
    -----
    The pointwise exponential operator :math:`exp` is defined as

    .. math:: (|f|^2)(x) = |f(x)|^2 = Re(f(x))^2 + Im(f(x))^2

    where :math:`Re` and :math:`Im` denote the real- and imaginary parts. 
    Its discrete form is 

    .. math:: (|x|^2)_i = Re(x_i)^2 + Im(x_i)^2.
    """

    def __init__(self, domain):
        codomain = domain.real_space()
        super().__init__(domain, codomain)

    def _eval(self, x, differentiate=False):
        if differentiate:
            self._factor = 2 * x
        return x.real**2 + x.imag**2

    def _derivative(self, x):
        return (self._factor.conj() * x).real

    def _adjoint(self, y):
        return self._factor * y
