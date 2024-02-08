import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util

from regpy.operators import Identity


class TikhonovCG(Solver):
    """The Tikhonov method for linear inverse problems. Minimizes

        ||T x - data||**2 + regpar * ||x - xref||**2

    using a conjugate gradient method.

    Parameters
    ----------
    setting : regpy.solvers.HilbertSpaceSetting
        The setting of the forward problem.
    data : array-like
        The measured data.
    regpar : float
        The regularization parameter. Must be positive.
    tol : float, optional
        The tolerance for the residual relative to the initial at which to stop. Default is
        the machine epsilon. Iterating beyond this point produces `NaN`s.
    reltolx, reltoly : float, optional
        Relative tolerance in domain and codomain.
    krylov_basis : Compute orthonormal basis vectors of the Krylov subspaces while running CG solver
    """
    def __init__(
        self, setting, data, regpar, xref=None, 
        tol=1e-6, reltolx=0.3, reltoly=0.3, 
        krylov_basis=None, preconditioner=None,
        logging_level = logging.INFO
        ):
        assert setting.op.linear

        super().__init__()
        self.log.setLevel(logging_level)
        self.setting = setting
        """The problem setting."""
        self.regpar = regpar
        """The regularization parameter."""
        #self.log.debug('rel. tolerances: {} in domain, {} in codomain, {} reduction residual'.format(reltolx,reltoly,tol))
        self.tol = tol
        """The tolerance."""

        # TODO Improve documentation for these two.
        self.reltolx = reltolx
        """The relative tolerance in the domain."""
        self.reltoly = reltoly
        """The relative tolerance in the codomain."""

        self.x = self.setting.op.domain.zeros()
        if self.reltolx is not None:
            self.norm_x = 0
        self.y = self.setting.op.codomain.zeros()
        if self.reltoly is not None:
            self.g_y = self.setting.op.codomain.zeros()
            self.norm_y = 0

        
        if preconditioner is None:
            self.preconditioner = Identity (self.setting.h_domain.vecsp)
            self.penalty = Identity (self.setting.h_domain.vecsp)
        else: 
            self.preconditioner = preconditioner
            self.penalty = self.preconditioner * self.setting.h_domain.gram * self.preconditioner * self.setting.h_domain.gram_inv

        self.g_res = self.preconditioner( self.setting.op.adjoint(self.setting.h_codomain.gram(data)) )
        """The gram matrix applied to the residual."""
        if xref is not None:
            self.g_res += self.regpar *self.preconditioner( self.setting.h_domain.gram(xref) )
        res = self.setting.h_domain.gram_inv(self.g_res)
        """The residual."""
        self.norm_res = np.real(np.vdot(self.g_res, res))
        """The norm of the residual."""
        self.norm_res_init = self.norm_res
        """The norm of the residual in the first iteration, for `tol`."""
        self.dir = res
        """The direction of descent."""
        self.g_dir = np.copy(self.g_res)
        """The gram matrix applied to the direction of descent."""
        # TODO Improve documentation
        self.kappa = 1
        """Auxiliary parameter for estimating the relative tolerances."""

        self.krylov_basis=krylov_basis
        if self.krylov_basis is not None: 
            self.iteration_number=0
            self.krylov_basis[self.iteration_number, :] = res / np.linalg.norm(res)
        """In every iteration step of the Tikhonov solver a new orthonormal vector is computed"""


    def _next(self):
        Tdir = self.setting.op( self.preconditioner(self.dir) )
        g_Tdir = self.setting.h_codomain.gram(Tdir)
        stepsize = self.norm_res / np.real(
            np.vdot(g_Tdir, Tdir) + self.regpar * np.vdot(self.penalty (self.g_dir), self.dir)
        )

        self.x += stepsize * self.dir
        if self.reltolx is not None:
            self.norm_x = np.real(np.vdot(self.x, self.setting.h_domain.gram(self.x)))

        self.y += stepsize * Tdir
        if self.reltoly is not None:
            self.g_y += stepsize * g_Tdir
            self.norm_y = np.real(np.vdot(self.g_y, self.y))

        self.g_res -= stepsize * (self.preconditioner( self.setting.op.adjoint(g_Tdir) )+ self.regpar * self.penalty (self.g_dir) )
        res = self.setting.h_domain.gram_inv(self.g_res)

        norm_res_old = self.norm_res
        self.norm_res = np.real(np.vdot(self.g_res, res))
        beta = self.norm_res / norm_res_old

        if self.krylov_basis is not None:
            self.iteration_number+=1
            if self.iteration_number < self.krylov_basis.shape[0]:
                self.krylov_basis[self.iteration_number, :] = res / np.linalg.norm(res)

        self.kappa = 1 + beta * self.kappa

        if self.krylov_basis is None or self.iteration_number > self.krylov_basis.shape[0]:
            """If Krylov subspace basis is computed, then stop the iteration only if the number of iterations exceeds the order of the Krylov space"""
            
            tol_report = 'it.{} err/Tol '.format(self.iteration_step_nr)
            if self.reltolx is not None:
                valx = np.sqrt(self.norm_res / self.norm_x / self.kappa) / self.regpar
                tol_report = tol_report+'X:{:1.1e}/{:1.1e} '.format(valx,self.reltolx / (1 + self.reltolx))
                if valx < self.reltolx / (1 + self.reltolx):
                    return self.converge()

            if self.reltoly is not None:
                valy = np.sqrt(self.norm_res / self.norm_y / self.kappa / self.regpar)
                tol_report = tol_report+"Y:{:1.1e}/{:1.1e} ".format(valy,self.reltoly / (1 + self.reltoly))
                if valy < self.reltoly / (1 + self.reltoly):
                    return self.converge()

            if self.tol is not None:
                val = np.sqrt(self.norm_res / self.norm_res_init / self.kappa) 
                tol_report = tol_report+"res.red: {:1.1e}/{:1.1e}".format(val,self.tol)
                if val < self.tol: 
                    return self.converge()

            self.log.debug(tol_report)

        self.dir *= beta
        self.dir += res
        self.g_dir *= beta
        self.g_dir += self.g_res