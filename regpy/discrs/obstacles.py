import numpy as np

from regpy.discrs import UniformGrid
import regpy.util as util


# The coefficients of this curve are actually equidistant samples of the radial function,
# so we can simply inherit from UniformGrid to get the Sobolev implementation for free.
class StarTrigDiscr(UniformGrid):
    """A discretization representing star-shaped obstacles parametrized in a trigonometric basis.
    Will always be 1d an complex.

    Parameters
    ----------
    n : int
        The number of coefficients.
    """
    def __init__(self, n):
        assert isinstance(n, int)
        super().__init__(np.linspace(0, 2 * np.pi, n, endpoint=False))

    def eval_curve(self, coeffs, nvals=None, nderivs=0):
        """Compute a curve for the given coefficients. All parameters will be passed to the
        constructor of `StarTrigCurve`, which see.
        """
        return StarTrigCurve(self, coeffs, nvals, nderivs)

    def sample(self, f):
        return np.asarray(
            np.broadcast_to(f(np.linspace(0, 2*np.pi, self.size, endpoint=False)), self.shape),
            dtype=self.dtype
        )


class StarTrigCurve:
    # TODO Rename attributes. `q`, `z`, `zpabs`, etc are not good names.

    """A class representing star shaped 2d curves with radial function parametrized in a
    trigonometric basis. Should usually be instantiated via `StarTrigDiscr.eval_curve`.

    Parameters
    ----------
    discr : StarTrigDiscr
        The underlying discretization.
    coeffs : array-like
        The coefficient array of the radial function.
    nvals : int, optional
        How many points on the curve to compute. The points will be at equispaced angles in
        `[0, 2pi)`. If omitted, the number of points will match the number of `coeffs`.
    nderivs : int, optional
        How many derivatives to compute. At most 3 derivatives are implemented.
    """

    def __init__(self, discr, coeffs, nvals=None, nderivs=0):
        assert isinstance(nderivs, int) and 0 <= nderivs <= 3
        self.discr = discr
        """The discretization."""
        self.coeffs = coeffs
        """The coefficients."""
        self.nvals = nvals or self.discr.size
        """The number of computed values."""
        self.nderivs = nderivs
        """The number of computed derivatives."""

        self._frqs = 1j * np.arange(self.discr.size // 2 + 1)
        # (nvals / nx) * irfft(rfft(x), nvals) can be used for trig interpolation
        self.q = (self.nvals / self.discr.size) * np.fft.irfft(
            (self._frqs ** np.arange(self.nderivs + 1)[:, np.newaxis]) * np.fft.rfft(coeffs),
            self.nvals,
            axis=1
        )
        """The computed radii of the curve and its derivatives, shaped `(nderivs, nvals)`."""
        q = self.q

        t = np.linspace(0, 2 * np.pi, self.nvals, endpoint=False)
        cost = np.cos(t)
        sint = np.sin(t)

        self.z = np.stack([
            q[0, :] * cost,
            q[0, :] * sint
        ])
        """The points on the curve as `(2, nvals)` array."""

        if nderivs < 1:
            return

        self.zp = np.stack([
            q[1, :] * cost - q[0, :] * sint,
            q[1, :] * sint + q[0, :] * cost
        ])
        """The points on the first derivative as `(2, nvals)` array."""
        self.zpabs = np.sqrt(self.zp[0, :]**2 + self.zp[1, :]**2)
        """The absolute values of the first derivative as `(nvals,)` array."""
        self.normal = np.stack([
            self.zp[1, :], -self.zp[0, :]
        ])
        """The outer normal vector as `(2, nvals)` array."""

        if nderivs < 2:
            return

        self.zpp = np.stack([
            q[2, :] * cost - 2 * q[1, :] * sint - q[0, :] * cost,
            q[2, :] * sint + 2 * q[1, :] * cost - q[0, :] * sint
        ])
        """The points on the second derivative as `(2, nvals)` array."""

        if nderivs < 3:
            return

        self.zppp = np.stack([
            q[3, :] * cost - 3 * q[2, :] * sint - 3 * q[1, :] * cost + q[0, :] * sint,
            q[3, :] * sint + 3 * q[2, :] * cost - 3 * q[1, :] * sint - q[0, :] * cost
        ])
        """The points on the third derivative as `(2, nvals)` array."""

    # TODO Should these be turned into an operator?

    def der_normal(self, h):
        """Computes the normal part of the perturbation of the curve caused by
        perturbing the coefficient vector curve.coeff in direction `h`."""
        return (self.q[0, :] / self.zpabs) * (self.nvals / self.discr.size) * np.fft.irfft(
            np.fft.rfft(h), self.nvals
        )

    def adjoint_der_normal(self, g):
        """Computes the adjoint of `der_normal`."""
        return util.adjoint_rfft(
            util.adjoint_irfft(
                (self.q[0, :] / self.zpabs) * (self.nvals / self.discr.size) * g,
                self.discr.size // 2 + 1
            ),
            self.discr.size
        )

    def arc_length_der(self, h):
        """Computes the derivative of `h` with respect to arclength."""
        return (self.nvals / self.discr.size) * np.fft.irfft(
            self._frqs * np.fft.rfft(h), self.nvals
        ) / self.zpabs
