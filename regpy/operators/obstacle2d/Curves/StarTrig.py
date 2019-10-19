import numpy as np

from regpy.discrs import Discretization
from regpy.util import trig_interpolate


class StarTrigDiscr(Discretization):
    def __init__(self, n):
        assert isinstance(n, int)
        # TODO Curves should be real. Use rfft?
        super().__init__(n, dtype=complex)

    def eval_curve(self, coeffs, nvals=None, nderivs=0):
        return StarTrigCurve(self, coeffs, nvals, nderivs)


class StarTrigCurve:
    # TODO Rename attributes. `q`, `z`, `zpabs`, etc are not good names.

    def __init__(self, discr, coeffs, nvals=None, nderivs=0):
        assert isinstance(nderivs, int) and 0 <= nderivs <= 3
        self.discr = discr
        self.coeffs = coeffs
        self.nvals = nvals or self.discr.size
        self.nderivs = nderivs

        coeffhat = trig_interpolate(self.coeffs, self.nvals)
        self.q = np.zeros((nderivs + 1, self.nvals))
        self._frqs = 1j * np.linspace(-self.nvals / 2, self.nvals / 2 - 1, self.nvals)
        for d in range(0, nderivs + 1):
            self.q[d, :] = np.real(np.fft.ifft(np.fft.fftshift(self._frqs**d * coeffhat)))

        q = self.q
        t = 2 * np.pi * np.linspace(0, self.nvals - 1, self.nvals) / self.nvals
        cost = np.cos(t)
        sint = np.sin(t)

        self.z = np.append(q[0, :] * cost, q[0, :] * sint).reshape(2, q[0, :].shape[0])

        if nderivs < 1:
            return

        self.zp = np.append(
            q[1, :] * cost - q[0, :] * sint,
            q[1, :] * sint + q[0, :] * cost
        ).reshape(2, q[0, :].shape[0])
        self.zpabs = np.sqrt(self.zp[0, :]**2 + self.zp[1, :]**2)
        # outer normal vector
        self.normal = np.append(
            self.zp[1, :], -self.zp[0, :]
        ).reshape(2, self.zp[0, :].shape[0])

        if nderivs < 2:
            return

        self.zpp = np.append(
            q[2, :] * cost - 2 * q[1, :] * sint - q[0, :] * cost,
            q[2, :] * sint + 2 * q[1, :] * cost - q[0, :] * sint
        ).reshape(2, self.nvals)

        if nderivs < 3:
            return

        self.zppp = np.append(
            q[3, :] * cost - 3 * q[2, :] * sint - 3 * q[1, :] * cost + q[0, :] * sint,
            q[3, :] * sint + 3 * q[2, :] * cost - 3 * q[1, :] * sint - q[0, :] * cost
        ).reshape(2, self.nvals)

    # TODO Should these be turned into an operator?

    def der_normal(self, h):
        """Computes the normal part of the perturbation of the curve caused by
        perturbing the coefficient vector curve.coeff in direction `h`."""
        h_long = np.fft.ifft(np.fft.fftshift(trig_interpolate(h, self.nvals)))
        return self.q[0, :] * h_long / self.zpabs

    def adjoint_der_normal(self, g):
        """Computes the adjoint of `der_normal`."""
        adj_long = g * self.q[0, :] / self.zpabs
        adj = (len(g) / self.nvals) * np.fft.ifft(np.fft.fftshift(
            trig_interpolate(adj_long, self.discr.size))
        )
        # TODO Why real part? Do we use only real perturbations to the coefficients?
        return adj.real

    def arc_length_der(self, h):
        """Computes the derivative of `h` with respect to arclength"""
        return np.fft.ifft(np.fft.fftshift(
            self._frqs * trig_interpolate(h, self.nvals)
        )) / self.zpabs
