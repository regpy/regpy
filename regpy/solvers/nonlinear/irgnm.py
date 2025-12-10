from copy import copy
from math import sqrt,isqrt

import numpy as np

from regpy.stoprules import CountIterations
from regpy.util import Errors

from ..general import Setting, RegSolver
from ..linear.tikhonov import TikhonovCG

__all__ = ["IrgnmCG","LevenbergMarquardt","IrgnmCGPrec"]

class IrgnmCG(RegSolver):
    r"""The Iteratively Regularized Gauss-Newton Method method. In each iteration, minimizes

    .. math::
        \Vert(x_{n}) + T'[x_n] h - data\Vert^{2} + regpar_{n} \cdot \Vert x_{n} + h - init\Vert^{2}

    where :math:`T` is a Frechet-differentiable operator, using `regpy.solvers.linear.tikhonov.TikhonovCG`.
    :math:`regpar_n` is a decreasing geometric sequence of regularization parameters.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem.
    data : array-like, default None
        The measured data. If it is None it is taken from the setting.
    regpar : float, default None
        The initial regularization parameter. Must be positive. If it is None it is taken from the setting.
    regpar_step : float, optional
        The factor by which to reduce the `regpar` in each iteration. Default: :math:`2/3`.
    init : array-like, optional
        The initial guess. Default: the zero array.
    cg_pars : dict
        Parameter dictionary for stopping of inner CG iteration passed to the inner `regpy.solvers.linear.tikhonov.TikhonovCG` solver.
    cg_stop: int
        Maximum number of inner CG iterations
    simplified_op : regpy.operators.Operator
        An operator the with the same mapping properties as setting.op, which is cheaper to evaluate. 
        It is used for the derivative in the Newton equation. 
        Default: None - then the derivative of setting.op is used.
    """

    def __init__(
               self, setting, data=None, regpar=None, regpar_step=2 / 3, 
                 init=None, 
                 cg_pars={'reltolx': 1/3., 'reltoly': 1/3.,'all_tol_criteria': False}, 
                cgstop=1000, 
                inner_it_logging_level = "WARNING", 
                simplified_op = None
         ):
        super().__init__(setting)
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        if(regpar is None):
            if(not setting.is_tikhonov):
                raise ValueError(Errors.value_error("Regularization parameter has to be included in setting or given directly."))
            regpar=setting.regpar
        self.data = data
        """The measured data."""
        if init is None:
            init = self.op.domain.zeros()
        self.init = init
        """The initial guess."""
        self.x = copy(self.init)
        if simplified_op:
            self.simplified_op = simplified_op
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y, self.deriv = self.op.linearize(self.x)
        self.regpar = regpar
        """The regularization parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        self.cgstop = cgstop
        """Maximum number of iterations for inner CG solver, or None"""
        self.inner_it_logging_level = inner_it_logging_level
        self._nr_inner_steps = 0

    def _next(self):
        if self.cgstop is not None:
            stoprule = CountIterations(self.cgstop)
        else:
            stoprule = CountIterations(2**15)
        # Disable info logging, but don't override log level for all CountIterations instances.
        stoprule.log = self.log.getChild('CountIterations')
        stoprule.log.setLevel("INFO")
        # Running Tikhonov solver
        step, _ = TikhonovCG(
            setting=Setting(self.deriv, self.h_domain, self.h_codomain),
            data=self.data - self.y,
            regpar=self.regpar,
            xref=self.init - self.x,
            **self.cg_pars,
            logging_level = self.inner_it_logging_level
        ).run(stoprule=stoprule)
        self.x += step
        if hasattr(self,'simplified_op'):
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y , self.deriv = self.op.linearize(self.x)
        self.regpar *= self.regpar_step
        self._nr_inner_steps = stoprule.iteration
        self.log.info('its.{}: alpha={}, CG its:{}'.format(self.iteration_step_nr,self.regpar,self._nr_inner_steps))
    
    def nr_inner_its(self):
        return self._nr_inner_steps
        

class LevenbergMarquardt(RegSolver):
    r"""The Levenberg-Marquardt method. In each iteration, minimizes

    .. math::
        \Vert(x_{n}) + T'[x_n] h - data\Vert^{2} + regpar_{n} \cdot \Vert h\Vert^{2}

    where :math:`T` is a Frechet-differentiable operator, using `regpy.solvers.linear.tikhonov.TikhonovCG`.
    :math:`regpar_n` is a decreasing geometric sequence of regularization parameters.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem.
    data : array-like, default None
        The measured data. If it is None it is taken from the setting.
    regpar : float, default None
        The initial regularization parameter. Must be positive. If it is None it is taken from the setting.
    regpar_step : float, optional
        The factor by which to reduce the `regpar` in each iteration. Default: :math:`2/3`.
    init : array-like, optional
        The initial guess. Default: the zero array.
    cg_pars : dict
        Parameter dictionary for stopping of inner CG iteration passed to the inner `regpy.solvers.linear.tikhonov.TikhonovCG` solver.
    cg_stop: int
        Maximum number of inner CG iterations
    simplified_op : regpy.operators.Operator
        An operator the with the same mapping properties as setting.op, which is cheaper to evaluate. 
        It is used for the derivative in the Newton equation. 
        Default: None - then the derivative of setting.op is used.
    """

    def __init__(
               self, setting, data=None, regpar=None, regpar_step=2 / 3, 
                 init=None, 
                 cg_pars={'reltolx': 1/3., 'reltoly': 1/3.,'all_tol_criteria': False}, 
                cgstop=1000, 
                inner_it_logging_level = "WARNING", 
                simplified_op = None
         ):
        super().__init__(setting)
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        if(regpar is None):
            if(not setting.is_tikhonov):
                raise ValueError(Errors.value_error("Regularization parameter has to be included in setting or given directly."))
            regpar=setting.regpar
        self.data = data
        """The measured data."""
        if init is None:
            init = self.op.domain.zeros()
        self.init = init
        """The initial guess."""
        self.x = copy(self.init)
        if simplified_op:
            self.simplified_op = simplified_op
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y, self.deriv = self.op.linearize(self.x)
        self.regpar = regpar
        """The regularization parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        self.cgstop = cgstop
        """Maximum number of iterations for inner CG solver, or None"""
        self.inner_it_logging_level = inner_it_logging_level
        self._nr_inner_steps = 0

    def _next(self):
        if self.cgstop is not None:
            stoprule = CountIterations(self.cgstop)
        else:
            stoprule = CountIterations(2**15)
        # Disable info logging, but don't override log level for all CountIterations instances.
        stoprule.log = self.log.getChild('CountIterations')
        stoprule.log.setLevel("WARNING")
        # Running Tikhonov solver
        step, _ = TikhonovCG(
            setting=Setting(self.deriv, self.h_domain, self.h_codomain),
            data=self.data - self.y,
            regpar=self.regpar,
            **self.cg_pars,
            logging_level = self.inner_it_logging_level
        ).run(stoprule=stoprule)
        self.x += step
        if hasattr(self,'simplified_op'):
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y , self.deriv = self.op.linearize(self.x)
        self.regpar *= self.regpar_step
        self._nr_inner_steps = stoprule.iteration
        self.log.info('its.{}: alpha={}, CG its:{}'.format(self.iteration_step_nr,self.regpar,self._nr_inner_steps))
    
    def nr_inner_its(self):
        return self._nr_inner_steps


from regpy.operators import EinSum,PtwMultiplication
from regpy import util
from scipy.sparse.linalg import eigsh
from regpy.vecsps import UniformGridFcts
        
class IrgnmCGPrec(RegSolver):
    r"""The Iteratively Regularized Gauss-Newton Method method. In each iteration, minimizes

    .. math::
        \Vert F(x_n) + F'[x_n] h - data\Vert^2 + \text{regpar}_n  \Vert x_n + h - init\Vert^2

    where :math:`F` is a Frechet-differentiable operator, by solving in every iteration step the problem
    
    .. math::
        \underset{Mh = g}{\mathrm{minimize}}    \Vert T (M  g) - rhs\Vert^2 + \text{regpar} \Vert M  (g - x_{ref})\Vert^2

    with `regpy.solvers.linear.tikhonov.TikhonovCG' and spectral preconditioner :math:`M`.
    The spectral preconditioner :math:`M` is chosen, such that:

    .. math::
        M  A  M \approx Id

    where :math:`A = (T^t Gram_{codomain} T + \text{regpar} Id) = T^* T + \text{regpar} Id` 

    Note that the Tikhonov CG solver computes an orthonormal basis of vectors spanning the Krylov subspace of 
    the order of the number of iterations: :math:`\{v_j\}`
    We approximate A by the operator:

    .. math::
        C_k: v \mapsto \text{regpar} v +\sum_{j=1}^k \langle v, v_j\rangle \lambda_j v_j

    where lambda are the biggest eigenvalues of :math:`T*T`.
    
    We choose: :math:`M = C_k^{-1/2} and M^{-1} = C_k^{1/2}`

    It is:

    .. math::
        M     &: v \mapsto \frac{1}{\sqrt{\text{regpar}}} v + \sum_{j=1}^{k} \left[\frac{1}{\sqrt{\lambda_j+\text{regpar}}}-\frac{1}{\sqrt{\text{regpar}}}\right] \langle v_j, v\rangle v_j \\
        M^{-1}&: v \mapsto \sqrt{\text{regpar}} v + \sum_{j=1}^{k} \left[\sqrt{\lambda_j+\text{regpar}} -\sqrt{\text{regpar}}\right] \langle v_j, v\rangle v_j.

    This only works for UniformGridFcts domains.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem. The domain of the operator has to be of type UniformGridFcts.
    data : array-like, default None
        The measured data. If it is None it is taken from the setting.
    regpar : float, default None
        The initial regularization parameter. Must be positive. If it is None it is taken from the setting.
    regpar_step : float, optional
        The factor by which to reduce the `regpar` in each iteration. Default: `2/3`.
    init : array-like, optional
        The initial guess. Default: the zero array.
    cg_pars : dict
        Parameter dictionary passed to the inner `regpy.solvers.linear.tikhonov.TikhonovCG` solver.
    precpars : dict
        Parameter dictionary passed to the computation of the spectral preconditioner
    """

    def __init__(
        self, setting, data=None, regpar=None, regpar_step=2 / 3, 
        init=None, cg_pars=None,cgstop =None, precpars=None
        ):
        if(not isinstance(setting.op.domain,UniformGridFcts)):
            raise ValueError(f"Computation of preconditioner requires UniformGridFcts, but got domain of type {type(setting.op.domain)}.")
        super().__init__(setting)
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        if(regpar is None):
            if(not setting.is_tikhonov):
                raise ValueError(Errors.value_error("Regularization parameter has to be included in setting or given directly."))
            regpar=setting.regpar
        self.data = data
        """The measured data."""
        if init is None:
            init = self.op.domain.zeros()
        self.init = init
        """The initial guess."""
        self.x = copy(self.init)
        self.y, self.deriv = self.op.linearize(self.x)
        self.regpar = regpar
        """The regularization parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        if cg_pars is None:
            cg_pars = {}
        self.cg_pars = cg_pars
        self.cgstop = cgstop
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        
        self.k=0
        """Counts the number of iterations"""

        if precpars is None:
            self.krylov_order = 6
            """Order of krylov space in which the spectral preconditioner is computed"""
            self.number_eigenvalues = 4
            """Spectral preconditioner computed only from the biggest eigenvalues """
        else: 
            self.krylov_order = precpars['krylov_order']
            self.number_eigenvalues = precpars['number_eigenvalues']

        self.krylov_basis = np.zeros((self.krylov_order, *self.h_domain.vecsp.shape),dtype=self.op.domain.dtype)
        """Orthonormal Basis of Krylov subspace"""
        self.krylov_basis_img = np.zeros((self.krylov_order, *self.h_codomain.vecsp.shape),dtype=self.op.codomain.dtype)
        """Image of the Krylov Basis under the derivative of the operator"""
        self.krylov_basis_img_2 = np.zeros((self.krylov_order, *self.h_codomain.vecsp.shape),dtype=self.op.codomain.dtype)
        """Gram matrix applied to image of the Krylov Basis"""
        self.need_prec_update = True
        """Is an update of the preconditioner needed"""
    
    def _next(self):
        if self.cgstop is not None:
            stoprule = CountIterations(self.cgstop)
            # Disable info logging, but don't override log level for all
            # CountIterations instances.
        else:
            stoprule = CountIterations(2**15)
        stoprule.log = self.log.getChild('CountIterations')
        stoprule.log.setLevel("WARNING")
        self.log.info('Running Tikhonov solver.')
        
        if self.need_prec_update:
            self.log.info('Spectral Preconditioner needs to be updated')
            step, _ = TikhonovCG(
                setting=Setting(self.deriv, self.h_domain, self.h_codomain),
                data=self.data - self.y,
                regpar=self.regpar,
                krylov_basis=self.krylov_basis,
                xref=self.init - self.x,
                **self.cg_pars
            ).run(stoprule=stoprule)
            for i in range(0, self.krylov_order):
                self.krylov_basis_img[i, :] = self.deriv(self.krylov_basis[i, :])
                self.krylov_basis_img_2[i, :] = self.h_codomain.gram(self.krylov_basis_img[i, :])
            self.need_prec_update = False
            self._preconditioner_update()
            self.log.info('Spectral preconditioner updated')
          
        else:
            step, _ = TikhonovCG(
                setting=Setting(self.deriv, self.h_domain, self.h_codomain),
                data=self.data - self.y,
                regpar=self.regpar,
                xref=self.init-self.x,
                preconditioner=self.preconditioner,
                **self.cg_pars
            ).run(stoprule=stoprule)

            
        self.x += step
        self.y, self.deriv = self.op.linearize(self.x)
        self.regpar *= self.regpar_step
        
        self.k+=1
        if (isqrt(self.k))**2 == self.k:
            self.need_prec_update = True
                       
    def _preconditioner_update(self):
        """perform lanzcos method to calculate the preconditioner"""
        L = np.zeros((self.krylov_order, self.krylov_order), dtype=self.op.domain.dtype)
        for i in range(0, self.krylov_order):
            L[i, i] = np.vdot(self.krylov_basis_img[i, :], self.krylov_basis_img_2[i, :])
            for j in range(i+1, self.krylov_order):
                L[i, j] = np.vdot(self.krylov_basis_img[i, :], self.krylov_basis_img_2[j, :])
                L[j, i] = L[i, j].conjugate()
        r"""Express `T*T` in Krylov_basis"""
        lamb, U = eigsh(L, self.number_eigenvalues, which='LM')
        """Perform the computation of eigenvalues and eigenvectors"""
        diag_lamb = np.diag( np.sqrt(1 / (lamb + self.regpar) ) - sqrt(1 / self.regpar) )
        M_krylov = self.op.domain.volume_elem*U @ diag_lamb @ U.transpose().conjugate()
        chars1 = ''.join(chr(i) for i in range(ord('a'), ord('a') + self.krylov_basis.ndim))
        chars2 = ''.join(chr(i) for i in range(ord('A'), ord('A') + self.krylov_basis.ndim))
        krylov_M_krylov=np.einsum(f"{chars1[0]+chars2[0]},{chars2}->{chars1[0]+chars2[1:]}",M_krylov,self.krylov_basis,optimize=True)
        tensors=(self.krylov_basis.conjugate(),krylov_M_krylov)
        subscript=f"{chars1[1:]},{chars1},{chars1[0]+chars2[1:]}"
        self.preconditioner=EinSum(subscript,self.op.domain,tensors=tensors,codomain=self.op.domain)+PtwMultiplication(self.op.domain,1/sqrt(self.regpar))
        """Compute preconditioner"""
