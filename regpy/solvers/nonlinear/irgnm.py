import logging

import numpy as np

from regpy.solvers import RegularizationSetting, RegSolver
from regpy.solvers.linear.tikhonov import TikhonovCG
from regpy.stoprules import CountIterations


class IrgnmCG(RegSolver):
    r"""The Iteratively Regularized Gauss-Newton Method method. In each iteration, minimizes

    .. math::
        \Vert(x_{n}) + T'[x_n] h - data\Vert^{2} + regpar_{n} \cdot \Vert x_{n} + h - init\Vert^{2}

    where :math:`T` is a Frechet-differentiable operator, using `regpy.solvers.linear.tikhonov.TikhonovCG`.
    :math:`regpar_n` is a decreasing geometric sequence of regularization parameters.

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem.
    data : array-like
        The measured data.
    regpar : float
        The initial regularization parameter. Must be positive.
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
               self, setting, data, regpar, regpar_step=2 / 3, 
                 init=None, 
                 cg_pars={'reltolx': 1/3., 'reltoly': 1/3.,'all_tol_criteria': False}, 
                cgstop=1000, 
                inner_it_logging_level = logging.WARNING, 
                simplified_op = None
         ):
        super().__init__(setting)
        self.data = data
        """The measured data."""
        if init is None:
            init = self.op.domain.zeros()
        self.init = np.asarray(init)
        """The initial guess."""
        self.x = np.copy(self.init)
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
        stoprule.log.setLevel(logging.WARNING)
        # Running Tikhonov solver
        step, _ = TikhonovCG(
            setting=RegularizationSetting(self.deriv, self.h_domain, self.h_codomain),
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
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem.
    data : array-like
        The measured data.
    regpar : float
        The initial regularization parameter. Must be positive.
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
               self, setting, data, regpar, regpar_step=2 / 3, 
                 init=None, 
                 cg_pars={'reltolx': 1/3., 'reltoly': 1/3.,'all_tol_criteria': False}, 
                cgstop=1000, 
                inner_it_logging_level = logging.WARNING, 
                simplified_op = None
         ):
        super().__init__(setting)
        self.data = data
        """The measured data."""
        if init is None:
            init = self.op.domain.zeros()
        self.init = np.asarray(init)
        """The initial guess."""
        self.x = np.copy(self.init)
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
        stoprule.log.setLevel(logging.WARNING)
        # Running Tikhonov solver
        step, _ = TikhonovCG(
            setting=RegularizationSetting(self.deriv, self.h_domain, self.h_codomain),
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


from regpy.operators import MatrixMultiplication
from regpy import util
from scipy.sparse.linalg import eigsh
        
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

    where :math:`A = (Gram_{domain}^{-1} T^t Gram_{codomain} T + \text{regpar} Id) = T^* T + \text{regpar} Id` 

    Note that the Tikhonov CG solver computes an orthonormal basis of vectors spanning the Krylov subspace of 
    the order of the number of iterations: :math:`\{v_j\}`
    We approximate A by the operator:

    .. math::
        C_k: v \mapsto \text{regpar} v +\sum_{j=1}^k \langle v, v_j\rangle lambda_j v_j

    where lambda are the biggest eigenvalues of :math:`T*T`.
    
    We choose: :math:`M = C_k^{-1/2} and M^{-1} = C_k^{1/2}`

    It is:

    .. math::
        M     &: v \mapsto \frac{1}{\sqrt{\text{regpar}}} v + \sum_{j=1}^{k} \left[\frac{1}{\sqrt{\lambda_j+\text{regpar}}}-\frac{1}{\sqrt{\text{regpar}}}\right] \langle v_j, v\rangle v_j \\
        M^{-1}&: v \mapsto \sqrt{\text{regpar}} v + \sum_{j=1}^{k} \left[\sqrt{\lambda_j+\text{regpar}} -\sqrt{\text{regpar}}\right] \langle v_j, v\rangle v_j.

    At the moment this method does not work for complex domains/codomains

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem.
    data : array-like
        The measured data.
    regpar : float
        The initial regularization parameter. Must be positive.
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
        self, setting, data, regpar, regpar_step=2 / 3, 
        init=None, cg_pars=None,cgstop =None, precpars=None
        ):
        super().__init__(setting)
        self.data = data
        """The measured data."""
        if init is None:
            init = self.op.domain.zeros()
        self.init = np.asarray(init)
        """The initial guess."""
        self.x = np.copy(self.init)
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
            self.krylov_order = 5
            """Order of krylov space in which the spetcral preconditioner is computed"""
            self.number_eigenvalues = 4
            """Spectral preonditioner computed only from the biggest eigenvalues """
        else: 
            self.krylov_order = precpars['krylov_order']
            self.number_eigenvalues = precpars['number_eigenvalues']

        self.krylov_basis = np.zeros((self.krylov_order, self.h_domain.vecsp.size),dtype=self.op.domain.dtype)
        """Orthonormal Basis of Krylov subspace"""
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
        stoprule.log.setLevel(logging.WARNING)
        self.log.info('Running Tikhonov solver.')
        
        if self.need_prec_update:
            self.log.info('Spectral Preconditioner needs to be updated')
            step, _ = TikhonovCG(
                setting=RegularizationSetting(self.deriv, self.h_domain, self.h_codomain),
                data=self.data - self.y,
                regpar=self.regpar,
                krylov_basis=self.krylov_basis,
                xref=self.init - self.x,
                **self.cg_pars
            ).run(stoprule=stoprule)
            self.need_prec_update = False
            self._preconditioner_update()
            self.log.info('Spectral preconditioner updated')
          
        else:
            preconditioner = MatrixMultiplication(self.M, domain=self.h_domain.vecsp, codomain=self.h_domain.vecsp)
            step, _ = TikhonovCG(
                setting=RegularizationSetting(self.deriv, self.h_domain, self.h_codomain),
                data=self.data - self.y,
                regpar=self.regpar,
                xref=self.init-self.x,
                preconditioner=preconditioner,
                **self.cg_pars
            ).run(stoprule=stoprule)
            step = self.M @ step
            
        self.x += step
        self.y, self.deriv = self.op.linearize(self.x)
        self.regpar *= self.regpar_step
        
        self.k+=1
        if (int(np.sqrt(self.k)))**2 == self.k:
            self.need_prec_update = True
                       
    def _preconditioner_update(self):
        """perform lanzcos method to calculate the preconditioner"""
        L = np.zeros((self.krylov_order, self.krylov_order))
        for i in range(0, self.krylov_order):
            L[i, :] = np.dot(self.krylov_basis, self.h_domain.gram_inv(
                self.deriv.adjoint(
                    self.h_codomain.gram(self.deriv((self.krylov_basis[i, :]))))))
        r"""Express `T*T` in Krylov_basis"""

        #TODO: Replace eigsh by Lanczos method to estimate the greatest eigenvalues, AND make shure it is a method that can handle complex matrices
        lamb, U = eigsh(L, self.number_eigenvalues, which='LM')
        """Perform the computation of eigenvalues and eigenvectors"""

        diag_lamb = np.diag( np.sqrt(1 / (lamb + self.regpar) ) - np.sqrt(1 / self.regpar) )
        M_krylov = np.float64(U @ diag_lamb @ U.transpose())
        self.M = self.krylov_basis.transpose() @ M_krylov @ self.krylov_basis + np.sqrt(1/self.regpar) * np.identity(self.krylov_basis.shape[1])
        """Compute preconditioner"""

        diag_lamb = np.diag ( np.sqrt(lamb + self.regpar) - np.sqrt(self.regpar) )
        M_krylov = np.float64(U @ diag_lamb @ U.transpose())
        self.M_inverse = self.krylov_basis.transpose() @ M_krylov @ self.krylov_basis + np.sqrt(self.regpar) * np.identity(self.krylov_basis.shape[1]) 
        """Compute inverse preconditioner matrix"""

