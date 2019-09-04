from . import Solver

import logging
import numpy as np

from ..util import eps


class TikhonovCG(Solver):
    def __init__(self, setting, rhs, regpar, xref=None, tol=eps, reltolx=None, reltoly=None):
        assert setting.op.linear

        super().__init__()
        self.setting = setting
        self.rhs = rhs
        self.regpar = regpar
        self.tol = tol
        self.reltolx = reltolx
        self.reltoly = reltoly

        self.x = self.setting.op.domain.zeros()
        if self.reltolx is not None:
            self.norm_x = 0
        self.y = self.setting.op.codomain.zeros()
        if self.reltoly is not None:
            self.ytilde = self.setting.op.codomain.zeros()
            self.norm_y = 0

        ztilde = self.setting.Hcodomain.gram(self.rhs)
        self.stilde = self.setting.op.adjoint(ztilde)
        if xref is not None:
            self.stilde += self.regpar * self.setting.Hdomain.gram(xref)
        s = self.setting.Hdomain.gram_inv(self.stilde)
        self.norm_s = np.real(np.vdot(self.stilde, s))
        self.norm_s0 = self.norm_s
        self.d = s
        self.dtilde = np.copy(self.stilde)
        self.kappa = 1

    def _next(self):
        z = self.setting.op(self.d)
        ztilde = self.setting.Hcodomain.gram(z)
        # gamma = self.norm_s / np.real(
        #     np.vdot(ztilde, z) + self.regpar * np.vdot(self.dtilde, self.d)
        # )
        aux = np.real(
            np.vdot(ztilde, z) + self.regpar * np.vdot(self.dtilde, self.d)
        )
        gamma = self.norm_s / aux

        self.x += gamma * self.d
        if self.reltolx is not None:
            self.norm_x = np.real(np.vdot(self.x, self.setting.Hdomain.gram(self.x)))

        self.y += gamma * z
        if self.reltoly is not None:
            self.ytilde += gamma * ztilde
            self.norm_y = np.real(np.vdot(self.ytilde, self.y))

        self.stilde -= gamma * np.real(self.setting.op.adjoint(ztilde) + self.regpar * self.dtilde)
        s = self.setting.Hdomain.gram_inv(self.stilde)
        norm_s_old = self.norm_s
        self.norm_s = np.real(np.vdot(self.stilde, s))
        beta = self.norm_s / norm_s_old
        self.kappa = 1 + beta * self.kappa

        if (
            self.reltolx is not None and
            np.sqrt(self.norm_s / self.norm_x / self.kappa) / self.regpar
            < self.reltolx / (1 + self.reltolx)
        ):
            return self.converge()

        if (
            self.reltoly is not None and
            np.sqrt(self.norm_s / self.norm_y / self.kappa / self.regpar)
            < self.reltoly / (1 + self.reltoly)
        ):
            return self.converge()

        if (
            self.tol is not None and
            np.sqrt(self.norm_s / self.norm_s0 / self.kappa) < self.tol
        ):
            return self.converge()

        self.d *= beta
        self.d += s
        self.dtilde *= beta
        self.dtilde += self.stilde
