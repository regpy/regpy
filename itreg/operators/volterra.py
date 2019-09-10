import numpy as np

from . import LinearOperator, NonlinearOperator
from .. import spaces


class Volterra(LinearOperator):
    """The discrete Volterra operator.

    The discrete Volterra operator is essentially a cumulative sum as in
    :func:`numpy.cumsum`. See Notes below.

    Parameters
    ----------
    domain : :class:`~itreg.spaces.Space`
        The domain on which the operator is defined.
    codomain : :class:`~itreg.spaces.Space`, optional
        The operator's codomain. Defaults to `domain`.

    Notes
    -----
    The Volterra operator :math:`V` is defined as

    .. math:: (Vf)(x) = \\int_0^x f(t) dt.

    Its discrete form, using a Riemann sum, is simply

    .. math:: (Vx)_i = h \\sum_{j \\leq i} x_j,

    where :math:`h` is the grid spacing.
    """

    def __init__(self, domain):
        assert isinstance(domain, spaces.UniformGrid)
        assert domain.ndim == 1
        super().__init__(domain, domain)

    def _eval(self, x):
        return self.domain.volume_elem * np.cumsum(x)

    def _adjoint(self, y):
        return self.domain.volume_elem * np.flipud(np.cumsum(np.flipud(y)))


class NonlinearVolterra(NonlinearOperator):
    """The non-linear discrete Volterra operator.

    This is like the linear :class:`~itreg.operators.Volterra` operator with an
    additional exponent:

    .. math:: (Vx)_i = h \\sum_{j \\leq i} x_j^n,

    Parameters
    ----------
    domain : :class:`~itreg.spaces.Space`
        The domain on which the operator is defined.
    exponent : float
        The exponent.
    codomain : :class:`~itreg.spaces.Space`, optional
        The operator's codomain. Defaults to `domain`.
    """

    def __init__(self, domain, exponent):
        assert isinstance(domain, spaces.UniformGrid)
        assert domain.ndim == 1
        self.exponent = exponent
        super().__init__(domain, domain)

    def _eval(self, x, differentiate=False):
        if differentiate:
            self._factor = self.exponent * x**(self.exponent - 1)
        return self.domain.volume_elem * np.cumsum(x**self.exponent)

    def _derivative(self, x):
        return self.domain.volume_elem * np.cumsum(self._factor * x)

    def _adjoint(self, y):
        return self.domain.volume_elem * self._factor * np.flipud(np.cumsum(np.flipud(y)))
