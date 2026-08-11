r"""Hilbert spaces for spherical-harmonic surface parameterizations."""

import numpy as np

from regpy.operators import PtwMultiplication
from regpy.util import Errors, memoized_property
from regpy.vecsps.surface import StarSHSurfaceSpc

from .base import HilbertSpace


__all__ = ["SobolevSphereSHBasis"]


class SobolevSphereSHBasis(HilbertSpace):
    r"""Sobolev space on a spherical-harmonic coefficient vector space.

    If

    .. math::

        f = \sum_{\ell=0}^{L}\sum_{m=-\ell}^{\ell}
            f_{\ell m}Y_{\ell m},

    then the discrete :math:`H^s(S^2)` inner product is

    .. math::

        \langle f,g\rangle_{H^s(S^2)}
        =
        \sum_{\ell=0}^{L}
        (1+\ell(\ell+1))^s
        \sum_{m=-\ell}^{\ell}
        f_{\ell m}\overline{g_{\ell m}}.

    Hence the Gram operator is diagonal in the spherical-harmonic basis.

    Parameters
    ----------
    vecsp : regpy.vecsps.surface.StarSHSurfaceSpc
        Spherical-harmonic coefficient space.
    index : real, optional
        Sobolev index :math:`s`. Any finite real value, including negative
        and fractional indices, is supported. Default: 1.
    """

    def __init__(self, vecsp: StarSHSurfaceSpc, index: float = 1.0):
        if not isinstance(vecsp, StarSHSurfaceSpc):
            raise TypeError(
                Errors.not_instance(
                    vecsp,
                    StarSHSurfaceSpc,
                    "SobolevSphereSHBasis requires a StarSHSurfaceSpc.",
                )
            )
        if not isinstance(index, (int, float, np.integer, np.floating)):
            raise TypeError(
                Errors.type_error(
                    f"index must be a finite real scalar. Given: {index!r}"
                )
            )
        if not np.isfinite(index):
            raise ValueError(
                Errors.value_error(
                    f"index must be finite. Given: {index!r}"
                )
            )

        self.index = float(index)
        super().__init__(vecsp)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.vecsp == other.vecsp and self.index == other.index
        return NotImplemented

    @memoized_property
    def weights(self):
        """Diagonal weights in ``l*l + l + m`` ordering."""
        ell = np.arange(self.vecsp.sh_deg + 1, dtype=float)
        degree_weights = (1.0 + ell * (ell + 1.0)) ** self.index
        return np.repeat(
            degree_weights,
            2 * np.arange(self.vecsp.sh_deg + 1) + 1,
        )

    @memoized_property
    def gram(self):
        """Diagonal Gram operator in the spherical-harmonic basis."""
        return PtwMultiplication(self.vecsp, self.weights)
