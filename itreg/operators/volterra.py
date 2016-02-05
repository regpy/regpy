import logging
import numpy as np

from . import LinearOperator

__all__ = ['Volterra']

log = logging.getLogger(__name__)


class Volterra(LinearOperator):
    """The (discrete) Volterra operator.

    The discrete Volterra operator is essentially a cumulative sum as in
    :func:`numpy.cumsum`. See Notes below.

    Parameters
    ----------
    domx : :class:`Space <itreg.spaces.Space>`
        The domain on which the operator is defined.
    domy : :class:`Space <itreg.spaces.Space>`, optional
        The operator's codomain. Defaults to `domx`.
    spacing : float, optional
        The grid spacing. Defaults to 1.

    Notes
    -----

    The Volterra operator :math:`V` is defined as

    .. math:: (Vf)(x) = \int_0^x f(t) dt.

    Its discrete form, using a Riemann sum, is simply

    .. math:: (Vx)_i = h \sum_{j \leq i} x_j,

    where :math:`h` is the grid spacing.

    """
    def __init__(self, domx, domy=None, spacing=1):
        super().__init__(domx, domy, log)
        assert(len(self.domx.shape) == 1)
        assert(self.domx.shape == self.domy.shape)
        self.spacing = spacing

    def __call__(self, x):
        return self.spacing * np.cumsum(x)

    def adjoint(self, x):
        return self.spacing * np.flipud(np.cumsum(np.flipud(x)))
