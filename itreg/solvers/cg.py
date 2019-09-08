from . import Solver

import logging
import numpy as np

from .. import util


class TikhonovCG(Solver):
    def __init__(self, setting, data, regpar, xref=None, tol=util.eps, reltolx=None, reltoly=None):
        assert setting.op.linear

        super().__init__()
        self.setting = setting
        self.regpar = regpar
        self.tol = tol
        self.reltolx = reltolx
        self.reltoly = reltoly

        self.x = self.setting.op.domain.zeros()
        if self.reltolx is not None:
            self.norm_x = 0
        self.y = self.setting.op.codomain.zeros()
        if self.reltoly is not None:
            self.g_y = self.setting.op.codomain.zeros()
            self.norm_y = 0

        self.g_res = self.setting.op.adjoint(self.setting.Hcodomain.gram(data))
        if xref is not None:
            self.g_res += self.regpar * self.setting.Hdomain.gram(xref)
        res = self.setting.Hdomain.gram_inv(self.g_res)
        self.norm_res = np.real(np.vdot(self.g_res, res))
        self.norm_res_init = self.norm_res
        self.dir = res
        self.g_dir = np.copy(self.g_res)
        self.kappa = 1

    def _next(self):
        Tdir = self.setting.op(self.dir)
        g_Tdir = self.setting.Hcodomain.gram(Tdir)
        stepsize = self.norm_res / np.real(
            np.vdot(g_Tdir, Tdir) + self.regpar * np.vdot(self.g_dir, self.dir)
        )

        self.x += stepsize * self.dir
        if self.reltolx is not None:
            self.norm_x = np.real(np.vdot(self.x, self.setting.Hdomain.gram(self.x)))

        self.y += stepsize * Tdir
        if self.reltoly is not None:
            self.g_y += stepsize * g_Tdir
            self.norm_y = np.real(np.vdot(self.g_y, self.y))

        self.g_res -= stepsize * (self.setting.op.adjoint(g_Tdir) + self.regpar * self.g_dir)
        res = self.setting.Hdomain.gram_inv(self.g_res)

        norm_res_old = self.norm_res
        self.norm_res = np.real(np.vdot(self.g_res, res))
        beta = self.norm_res / norm_res_old

        self.kappa = 1 + beta * self.kappa

        if (
            self.reltolx is not None and
            np.sqrt(self.norm_res / self.norm_x / self.kappa) / self.regpar
                < self.reltolx / (1 + self.reltolx)
        ):
            return self.converge()

        if (
            self.reltoly is not None and
            np.sqrt(self.norm_res / self.norm_y / self.kappa / self.regpar)
                < self.reltoly / (1 + self.reltoly)
        ):
            return self.converge()

        if (
            self.tol is not None and
            np.sqrt(self.norm_res / self.norm_res_init / self.kappa) < self.tol
        ):
            return self.converge()

        self.dir *= beta
        self.dir += res
        self.g_dir *= beta
        self.g_dir += self.g_res
