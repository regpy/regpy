import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from regpy.discrs import Discretization, UniformGrid
from regpy.operators import Operator
from regpy.operators.obstacle2d.Curves.StarTrig import StarTrig


# TODO This module needs to be cleaned up.


class PotentialOp(Operator):
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
        self.radius = radius
        self.nforward = nforward

        self.bd = StarTrig(64)

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
        self.bd.bd_eval(coeff, nfwd, 1)
        q = self.bd.q[0, :]
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
        h = self.bd.der_normal(h_coeff)
        q = self.bd.q[0, :]
        qq = self.bd.zpabs

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
        q = self.bd.q[0, :]
        qq = self.bd.zpabs

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

        adj = self.bd.adjoint_der_normal(adj).real
        return adj

    # def accept_proposed(self, positions):
    #     self.bd.bd_eval(positions, N, 1)
    #     q = self.bd.q[0, :]
    #     if q.max() >= self.radius:
    #         return False
    #
    #     if q.min() <= 0:
    #         return False
    #     return True


# def create_synthetic_data(self, noiselevel):
#     N = self.op.Nfwd_synth
#     t = 2 * np.pi / N * np.arange(0, N, 1)
#     t_fl = self.op.meas_angles
#     q = self.op.obstacle.bd_ex.radial(self.op.obstacle.bd_ex, N)
#     qq = q**2
#
#     flux = 1 / (2 * self.op.radius * N) * sum(qq) * np.ones(len(t_fl))
#     fac = 2 / (N * self.op.radius)
#     for j in range(0, int((N - 1) / 2)):
#         fac = fac / self.op.radius
#         qq = qq * q
#         flux = flux + (fac / (j + 3)) * np.cos((j + 1) * t_fl) * np.sum(qq * np.cos((j + 1) * t)) \
#                + (fac / (j + 3)) * np.sin((j + 1) * t_fl) * np.sum(qq * np.sin((j + 1) * t))
#
#     if N % 2 == 0:
#         fac = fac / self.op.radius
#         qq = qq * q
#         flux = flux + fac * np.cos(N / 2 * t_fl) * np.sum(qq * np.cos(N / 2 * t))
#     noise = np.random.randn(len(flux))
#     data = flux + noiselevel * noise / self.Hcodomain.norm(noise)
#     return data
#
#
# def create_impulsive_noise(n, eta, var=None):
#     """Create Mc such that |Mc|<eta
#     """
#     Mc = set('')
#     while (len(Mc) < int(eta * n)):
#         s = np.ceil(np.random.rand() * n)
#         t = np.ceil(np.random.rand() * n)
#         st = np.arange(s, t)
#         if s < t and len(Mc) + len(st) <= int(eta * n):
#             Mc = Mc.union(st)
#         if (len(Mc) == int(eta * n) - 1):
#             break
#
#     if var is None:
#         r"""Now create random noise on Mc such that noise = \pm 1/\eta with equal
#         probability"""
#         res = np.zeros((n))
#         res[np.asarry(list(Mc)).astype(int)] = (2 * np.random.uniform(1, 2, len(Mc)) - 3) / eta
#     else:
#         """Create Gaussian noise on Mc with variance var^2"""
#         res = np.zeros(n)
#         res[np.asarray(list(Mc)).astype(int)] = var * np.random.randn(len(Mc))
#     return res


def plots(op, reco, reco_data, data, exact_data, exact_solution, figsize=(8, 8), n=64):
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize)
    axs[0].set_title('Domain')
    axs[1].set_title('Heat source')
    axs[1].plot(exact_data)
    axs[1].plot(data)
    axs[1].plot(reco_data)
    ymin = 0.7 * min(reco_data.min(), data.min(), exact_data.min())
    ymax = 1.3 * max(reco_data.max(), data.max(), exact_data.max())
    axs[1].set_ylim((ymin, ymax))
    bd = op.bd
    pts = bd.coeff2Curve(reco, n)
    pts_2 = bd.coeff2Curve(exact_solution, n)
    poly = Polygon(np.column_stack([pts[0, :], pts[1, :]]), animated=True, fill=False)
    poly_2 = Polygon(np.column_stack([pts_2[0, :], pts_2[1, :]]), animated=True, fill=False)
    axs[0].add_patch(poly)
    axs[0].add_patch(poly_2)
    xmin = 1.5 * min(pts[0, :].min(), pts_2[0, :].min())
    xmax = 1.5 * max(pts[0, :].max(), pts_2[0, :].max())
    ymin = 1.5 * min(pts[1, :].min(), pts_2[1, :].min())
    ymax = 1.5 * max(pts[1, :].max(), pts_2[1, :].max())
    axs[0].set_xlim((xmin, xmax))
    axs[0].set_ylim((ymin, ymax))
    plt.show()
