import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util

from regpy.operators import Identity
from regpy.stoprules import CountIterations

class TikhonovCG(Solver):
    r"""The Tikhonov method for linear inverse problems. Minimizes
    \[
        \Vert T x - data\Vert^2 + regpar * \Vert x - xref\Vert^2
    \]
    using a conjugate gradient method. 
    To determine a stopping index yielding guaranteed error bounds, a partial embedded minimal residual method (MR) is 
    used, which can be implemented by updating a scalar parameter in each iteration. 
    For details on the use of the embedded MR method, see the Master thesis by Andrea Dietrich 
    "Analytische und numerische Untersuchung eines Abbruchkriteriums für das CG-Verfahren zur Minimierung 
    von Tikhonov Funktionalen", Univ. Göttingen, 2017 

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
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

        """The iteration is stopped at the first iteration index for which one of the following tolerance 
        criteria is satisfied."""
        self.tol = tol
        """The absolute tolerance in the domain."""
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
        """The Gram matrix applied to the direction of descent."""
        self.kappa = 1
        """ratio of tze squared norms of the residuals of the CG method and the MR-method.
        Used for error estimation."""

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
                    self.log.info(tol_report)
                    return self.converge()

            if self.reltoly is not None:
                valy = np.sqrt(self.norm_res / self.norm_y / self.kappa / self.regpar)
                tol_report = tol_report+"Y:{:1.1e}/{:1.1e} ".format(valy,self.reltoly / (1 + self.reltoly))
                if valy < self.reltoly / (1 + self.reltoly):
                    self.log.info(tol_report)
                    return self.converge()

            if self.tol is not None:
                val = np.sqrt(self.norm_res / self.norm_res_init / self.kappa) 
                tol_report = tol_report+"res.red: {:1.1e}/{:1.1e}".format(val,self.tol)
                if val < self.tol: 
                    self.log.info(tol_report)
                    return self.converge()

            self.log.debug(tol_report)

        self.dir *= beta
        self.dir += res
        self.g_dir *= beta
        self.g_dir += self.g_res


class GeometricSequence:
    r"""Iterator generating a geometric sequence
    Parameters: alpha0, q
    Yields: Sequence defined recursively by 
        alpha_0 = alpha0
        alpha_{n+1} = q*alpha_n
    """    
    def __init__(self, alpha0,q):
        self.alpha = alpha0
        self.alpha0 = alpha0
        self.q = q

    def __iter__(self):
        self.alpha = self.alpha0
        return self

    def __next__(self):
        result = self.alpha
        self.alpha = self.alpha*self.q
        return result

class TikhonovAlphaGrid(Solver):
    r"""Class runnning Tikhonov regularization on a grid of different regularization parameters.
    This allows to choose the regularization parameter by some stopping rule. 
    Tikhonov functionals are minimized by an inner CG iteration.

    Parameters:
    setting:  regpy.solvers.RegularizationSetting
        The setting of the forward problem.
    data: array-like
        The right hand side.
    alphas: Either an iterable giving the grid of alphas or a tuple (alpha0,q)
        In the latter case the seuqence \((alpha0*q^n)_{n=0,1,2,...}\) is generated.
    xref: Initial guess

    Further keyword arguments for TikhonovCG can be given. 
    """
    def __init__(self,setting, data, alphas, max_inner_iter=1000,**kwargs):
        if isinstance(alphas,tuple) and len(alphas)==2:
            self._alphas = GeometricSequence(alphas[0],alphas[1])
        else:
            self._alphas = alphas
        self.setting = setting
        """The problem setting."""
        self.data = data
        """Right hand side of the operator equation."""
        self.max_inner_iter = max_inner_iter
        """maximum number of inner CG iterations."""
        if not 'logging_level' in kwargs:
            kwargs['logging_level']= logging.WARNING
        self.kwargs = kwargs
        """Arguments passed to TikhonovCG"""
        super().__init__()
        if 'xref' in kwargs:
            self.x = kwargs['xref']
            self.y = setting.op(self.x)
        else:
            self.x = setting.op.domain.zeros()
            self.y = setting.op.codomain.zeros()

    def _next(self):
        try:
            alpha = next(self._alphas)
        except StopIteration:
            return self.converge()
        inner_stoprule = CountIterations(max_iterations=self.max_inner_iter)
        inner_stoprule.log = self.log.getChild('CountIterations')
        inner_stoprule.log.setLevel(logging.WARNING)
        self.kwargs['xref']=self.x
        tikhcg =TikhonovCG(self.setting,self.data,alpha,**self.kwargs)
        self.x, self.y = tikhcg.run(inner_stoprule)
        self.log.info('alpha = {}, inner CG its = {}'.format(alpha,inner_stoprule.iteration))

class NonstationaryIteratedTikhonov(Solver):
    r"""Iterated Tikhonov regularization with a given (fixed) sequence of regularization parameters.
       Tikhonov functionals are minimized by an inner CG iteration.

    Parameters:
    setting:  regpy.solvers.RegularizationSetting
        The setting of the forward problem.
    data: array-like
        The right hand side.
    alphas: Either an iterable giving the grid of alphas or a tuple (alpha0,q)
        In the latter case the seuqence \((alpha0*q^n)_{n=0,1,2,...}\) is generated.
    xref: Initial guess.

    Further keyword arguments for TikhonovCG may be given
    """
    def __init__(self,setting, data, alphas, max_inner_iter=1000,**kwargs):
        if isinstance(alphas,tuple) and len(alphas)==2:
            self._alphas = GeometricSequence(alphas[0],alphas[1])
        else:
            self._alphas = alphas
        self.setting = setting
        """The problem setting."""
        self.data = data
        """Right hand side of the operator equations."""
        self.max_inner_iter = max_inner_iter
        """Maximum number of inner CG iterations."""
        self.alpha_eff = np.inf
        """effective regularization parameter. 1/alpha_eff is the sum of the reciprocals of all regularization parameters."""
        if not 'logging_level' in kwargs:
            kwargs['logging_level']= logging.WARNING
        self.kwargs = kwargs
        """Keyword arguments passed to TikhonovCG"""
        super().__init__()
        if 'href' in kwargs:
            self.x = kwargs['href']
            self.y = setting.op(self.x)
        else:
            self.x = setting.op.domain.zeros()
            self.y = setting.op.codomain.zeros()

    def _next(self):
        try:
            alpha = next(self._alphas)
        except StopIteration:
            return self.converge()
        self.alpha_eff = 1./(1./alpha + 1./self.alpha_eff)
        inner_stoprule = CountIterations(max_iterations=self.max_inner_iter)
        inner_stoprule.log = self.log.getChild('CountIterations')
        inner_stoprule.log.setLevel(logging.WARNING)
        tikhcg =TikhonovCG(self.setting,self.data,alpha,**self.kwargs)
        self.x, self.y = tikhcg.run(inner_stoprule)
        self.log.info('alpha_eff = {}, inner CG its = {}'.format(self.alpha_eff,inner_stoprule.iteration))