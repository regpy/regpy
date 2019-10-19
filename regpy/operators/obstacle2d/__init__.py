import numpy as np

from regpy.discrs import UniformGrid
from regpy.operators import Operator
from regpy.discrs.obstacles import StarTrigDiscr


class Potential(Operator):
    """ identification of the shape of a heat source from measurements of the
     heat flux.
     see F. Hettlich & W. Rundell "Iterative methods for the reconstruction
     of an inverse potential problem", Inverse Problems 12 (1996) 251–266
     or
     sec. 3 in T. Hohage "Logarithmic convergence rates of the iteratively
     regularized Gauss–Newton method for an inverse potential
     and an inverse scattering problem" Inverse Problems 13 (1997) 1279–1299
    """

    def __init__(self, domain, radius, nmeas, nforward=64):
        assert isinstance(domain, StarTrigDiscr)
        self.radius = radius
        self.nforward = nforward

        super().__init__(
            domain=domain,
            codomain=UniformGrid(np.linspace(0, 2 * np.pi, nmeas, endpoint=False))
        )

        k = 1 + np.arange(self.nforward)
        k_t = np.outer(k, np.linspace(0, 2 * np.pi, self.nforward, endpoint=False))
        k_tfl = np.outer(k, self.codomain.coords[0])
        self.cosin = np.cos(k_t)
        self.sinus = np.sin(k_t)
        self.cos_fl = np.cos(k_tfl)
        self.sin_fl = np.sin(k_tfl)

    def _eval(self, coeff, differentiate=False):
        nfwd = self.nforward
        self._bd = self.domain.eval_curve(coeff, nvals=nfwd, nderivs=1)
        q = self._bd.q[0, :]
        if q.max() >= self.radius:
            raise ValueError('object penetrates measurement circle')
        if q.min() <= 0:
            raise ValueError('radial function negative')

        qq = q**2
        flux = 1 / (2 * self.radius * nfwd) * np.sum(qq) * self.codomain.ones()
        fac = 2 / (nfwd * self.radius)
        for j in range(0, (nfwd - 1) // 2):
            fac /= self.radius
            qq *= q
            flux += (
                (fac / (j + 3)) * self.cos_fl[:, j] * np.sum(qq * self.cosin[j, :]) +
                (fac / (j + 3)) * self.sin_fl[:, j] * np.sum(qq * self.sinus[j, :])
            )

        if nfwd % 2 == 0:
            fac /= self.radius
            qq *= q
            flux += fac * self.cos_fl[:, nfwd // 2] * np.sum(qq * self.cosin[nfwd // 2, :])
        return flux

    def _derivative(self, h_coeff):
        nfwd = self.nforward
        h = self._bd.der_normal(h_coeff)
        q = self._bd.q[0, :]
        qq = self._bd.zpabs

        der = 1 / (self.radius * nfwd) * np.sum(qq * h) * self.codomain.ones()
        fac = 2 / (nfwd * self.radius)
        for j in range((nfwd - 1) // 2):
            fac /= self.radius
            qq = qq * q
            der += fac * (
                self.cos_fl[:, j] * np.sum(qq * h * self.cosin[j, :]) +
                self.sin_fl[:, j] * np.sum(qq * h * self.sinus[j, :])
            )

        if nfwd % 2 == 0:
            fac /= self.radius
            qq = qq * q
            der += fac * self.cos_fl[:, nfwd // 2] * np.sum(qq * h * self.cosin[nfwd // 2, :])
        return der.real

    def _adjoint(self, g):
        nfwd = self.nforward
        q = self._bd.q[0, :]
        qq = self._bd.zpabs

        adj = 1 / (self.radius * nfwd) * np.sum(g) * qq
        fac = 2 / (nfwd * self.radius)
        for j in range((nfwd - 1) // 2):
            fac /= self.radius
            qq = qq * q
            adj += fac * (
                np.sum(g * self.cos_fl[:, j]) * (self.cosin[j, :] * qq) +
                np.sum(g * self.sin_fl[:, j]) * (self.sinus[j, :] * qq)
            )

        if nfwd % 2 == 0:
            fac /= self.radius
            qq = qq * q
            adj += fac * np.sum(g * self.cos_fl[:, nfwd // 2]) * (self.cosin[nfwd // 2, :] * qq)

        adj = self._bd.adjoint_der_normal(adj).real
        return adj
