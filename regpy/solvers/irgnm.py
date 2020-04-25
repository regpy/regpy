import numpy as np

from regpy.solvers import HilbertSpaceSetting, Solver
from regpy.solvers.tikhonov import TikhonovCG


class IrgnmCG(Solver):
    """The Iteratively Regularized Gauss-Newton Method method. In each iteration, minimizes

        ||T(x_n) + T'[x_n] h - data||**2 + regpar_n * ||x_n + h - init||**2

    where `T` is a Frechet-differentiable operator, using `regpy.solvers.tikhonov.TikhonovCG`.
    `regpar_n` is a decreasing geometric sequence of regularization parameters.

    Parameters
    ----------
    setting : regpy.solvers.HilbertSpaceSetting
        The setting of the forward problem.
    data : array-like
        The measured data.
    regpar : float
        The initial regularization parameter. Must be positive.
    regpar_step : float, optional
        The factor by which to reduce the `regpar` in each iteration. Default: `2/3`.
    init : array-like, optional
        The initial guess. Default: the zero array.
    cgpars : dict
        Parameter dictionary passed to the inner `regpy.solvers.tikhonov.TikhonovCG` solver.
    """

    def __init__(self, setting, data, regpar, regpar_step=2 / 3, init=None, cgpars=None):
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.data = data
        """The measured data."""
        if init is None:
            init = self.setting.op.domain.zeros()
        self.init = np.asarray(init)
        """The initial guess."""
        self.x = np.copy(self.init)
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar = regpar
        """The regularizaton parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        if cgpars is None:
            cgpars = {}
        self.cgpars = cgpars
        """The additional `regpy.solvers.tikhonov.TikhonovCG` parameters."""

    def _next(self):
        self.log.info('Running Tikhonov solver.')
        step, _ = TikhonovCG(
            setting=HilbertSpaceSetting(self.deriv, self.setting.Hdomain, self.setting.Hcodomain),
            data=self.data - self.y,
            regpar=self.regpar,
            xref=self.init - self.x,
            **self.cgpars
        ).run()
        self.x += step
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar *= self.regpar_step
        
from regpy.operators import MatrixMultiplication
from regpy import util
        
class IrgnmCGLanczos(Solver):
    def __init__(self, setting, data, regpar, regpar_step=2 / 3, init=None, cgpars=None):
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.data = data
        """The measured data."""
        if init is None:
            init = self.setting.op.domain.zeros()
        self.init = np.asarray(init)
        """The initial guess."""
        self.x = np.copy(self.init)
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar = regpar
        """The regularizaton parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        if cgpars is None:
            cgpars = {}
        self.cgpars = cgpars
        """The additional `regpy.solvers.tikhonov.TikhonovCG` parameters."""
        
        self.k=0
        self.eigval_num = 5
        # orthonormalization computed in which krylov space
        self.krylov_num = 5
        self.orthonormal = np.zeros((self.krylov_num, self.data.shape[0]))
        self.need_prec_update = True
                
    def _next(self):
        self.log.info('Running Tikhonov solver.')
        
        if self.need_prec_update:
            step, _ = Tikhonov_need_update(
                setting=HilbertSpaceSetting(self.deriv, self.setting.Hdomain, self.setting.Hcodomain),
                data=self.data - self.y,
                regpar=self.regpar,
                orthonormal=self.orthonormal,
                xref=self.init - self.x,
                **self.cgpars
            ).run()
            self.need_prec_update = False
            self._lanzcos_update()

#Needs to be updated          
        else:
            preconditioner = MatrixMultiplication(self.M, domain=self.setting.Hdomain.discr, codomain=self.setting.Hdomain.discr)
            step, _ = TikhonovCG(
                setting=HilbertSpaceSetting(self.deriv * preconditioner, self.setting.Hdomain, self.setting.Hcodomain),
                data=self.data - self.y,
                regpar=self.regpar,
                xref=preconditioner(self.init - self.x),
                **self.cgpars
            ).run()
            step = self.M_inverse @ step
            
        self.x += step
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar *= self.regpar_step
        
        self.k+=1
        if (int(np.sqrt(self.k)))**2 == self.k:
            self.need_prec_update = True
            
            
    def _lanzcos_update(self):
        """perform lanzcos method to calculate the preconditioner"""
        #print(self.orthonormal)
        self.L = np.zeros((self.krylov_num, self.krylov_num))
        for i in range(0, self.krylov_num):
            self.L[i, :] = np.dot(self.orthonormal, self.setting.Hdomain.gram_inv(
                self.deriv.adjoint(
                    self.setting.Hcodomain.gram(self.deriv((self.orthonormal[i, :]))))))
        # TODO: Only compute the three biggest eigenvalues with Lanczos method
        # self.lamb, self.U=np.linalg.eig(self.L)
        # self.diag_lamb=np.zeros(self.L.shape)
        # for i in range(0, self.eigval_num):
        #    self.diag_lamb[i, i]=1/(self._regpar+self.lamb[i])-1/self._regpar
        #print(self.L)
        from scipy.sparse.linalg import eigsh
        self.lamb, self.U = eigsh(self.L, self.eigval_num, which='LM')
        
        self.diag_lamb = np.diag ( np.sqrt(self.lamb + self.regpar) - np.sqrt(self.regpar) )
        self.diag_lamb_inverse = np.diag( np.sqrt(1 / (self.lamb + self.regpar) ) - np.sqrt(1 / self.regpar) )

        self.lanczos_krylov = np.float64(self.U @ self.diag_lamb @ self.U.transpose())
        self.lanczos_krylov_inverse = np.float64(self.U @ self.diag_lamb_inverse @ self.U.transpose())

#TODO: We need M^(1/2) and M^(-1/2) instead
        self.M = self.orthonormal.transpose() @ self.lanczos_krylov_inverse @ self.orthonormal + np.sqrt(1/self.regpar) * np.identity(self.orthonormal.shape[1])
        self.M_inverse = self.orthonormal.transpose() @ self.lanczos_krylov @ self.orthonormal + np.sqrt(self.regpar) * np.identity(self.orthonormal.shape[1]) 
        

class Tikhonov_need_update(Solver):
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
    """
    def __init__(self, setting, data, regpar, orthonormal, xref=None, tol=util.eps, reltolx=None, reltoly=None):
        assert setting.op.linear

        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.regpar = regpar
        """The regularization parameter."""
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

        self.g_res = self.setting.op.adjoint(self.setting.Hcodomain.gram(data))
        """The gram matrix applied to the residual."""
        if xref is not None:
            self.g_res += self.regpar * self.setting.Hdomain.gram(xref)
        res = self.setting.Hdomain.gram_inv(self.g_res)
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
#new        
        self.orthonormal=orthonormal
        self.inner_number=0
        if self.inner_number <= self.orthonormal.shape[0]:
            self.orthonormal[self.inner_number, :] = res / np.linalg.norm(res)

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
#new        
        self.inner_number+=1
        if self.inner_number < self.orthonormal.shape[0]:
            self.orthonormal[self.inner_number, :] = res / np.linalg.norm(res)

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
        

def _lanczos(self, L, v, maxit):
    """perform lanczos method to calculate tridiagonal decomposition"""
    epsilon = np.dot(v, L @ v)
    w = L @ v - epsilon * v
    zeta = np.linalg.norm(w)
    v_old = v

    V = np.zeros((maxit, maxit))
    Epsilon = np.zeros(maxit)
    Zeta = np.zeros(maxit - 1)

    V[0, :] = v
    Epsilon[0] = epsilon

    counter = 1
    while (zeta != 0 and counter < maxit):
        v = w / zeta
        epsilon = np.dot(v, L @ v)
        w = L @ v - epsilon * v - zeta * v_old
        zeta = np.linalg.norm(w)
        v_old = v

        V[counter, :] = v
        Epsilon[counter] = epsilon
        Zeta[counter - 1] = zeta
        counter += 1
    return [V, Epsilon, Zeta]
