from math import sqrt
from copy import deepcopy,copy

from regpy.stoprules import CountIterations
from regpy.util import Errors

from ..general import RegSolver, Setting
from ..linear import SemismoothNewton_bilateral
from ..linear.tikhonov import GeometricSequence

__all__ = ["NewtonCG","NewtonCGFrozen","NewtonSemiSmoothFrozen","IterativelyRegularizedNewton"]

class NewtonCG(RegSolver):
    r"""The Newton-CG method. Solves the potentially non-linear, ill-posed equation:
    
    .. math::
        T(x) = y,

    where :math:`T` is a Frechet-differentiable operator. The Newton equations are solved by the
    conjugate gradient method applied to the normal equation (CGNE) using the regularizing
    properties of CGNE with early stopping (see Hanke 1997).

    If simplified_op is specified, it will be used to generate an approximation of the derivative 
    of the forward operator setting.op, which may be cheaper to evaluate. E.g., it may be the 
    derivative at the initial guess, which would yield a frozen Newton method. 

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The regularization setting includes the operator and penalty and data fidelity functionals.
    data : array-like
        The rhs y of the equation to be solved. Must be in setting.op.codomain.
    init : array-like, optional
        Initial guess to exact solution. (Default: setting.op.domain.zeros())
    cgmaxit : number, optional
        Maximal number of inner CG iterations. (Default: 50)
    rho : number, optional
        A fix number related to the termination (0<rho<1). (Default: 0.8)
    simplified_op : regpy.operators.Operator, optional
        Simplified operator to be used for the derivative. (Default: None)
    """

    def __init__(self, setting, data, init=None, cgmaxit=50, rho=0.8, simplified_op = None):
        super().__init__(setting)
        if init is not None and init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        if data not in self.op.codomain:
            raise ValueError(Errors.not_in_vecsp(data,self.op.codomain,vec_name="data",space_name="codomain"))
        self.data = data
        """The measured data."""

        self.x = init.copy() if init is not None else self.op.domain.zeros()
        if simplified_op:
            self.simplified_op = simplified_op
            """Simplified operator for derivative.
            """
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y, self.deriv = self.op.linearize(self.x)
        self.rho = rho
        r"""A fix number related to the termination :math:`(0<\rho<1)`."""
        self.cgmaxit = cgmaxit
        """Maximum number of iterations for inner CG solver."""
        self._k = 0
    
    def _next(self):
        self._k = 0
        self._s = self.data - self.y  
        # aux plays the role of s here to avoid storage for another vector in codomain
        self._x_k = self.op.domain.zeros()
        # self._s += - self.deriv(self._x_k)
        self._s2 = self.h_codomain.gram(self._s)
        self._norms0 = sqrt(self.op.codomain.vdot(self._s2, self._s).real)
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.h_domain.gram_inv(self._rtilde)
        self._d = self._r
        self._inner_prod = self.op.domain.vdot(self._r, self._rtilde).real
     
        while (self._k==0 or (sqrt(self.op.codomain.vdot(self._s2, self._s).real)
               > self.rho * self._norms0 and self._k < self.cgmaxit)):
            self._q = self.deriv(self._d)
            self._q2 = self.h_codomain.gram(self._q)
            self._alpha = self._inner_prod / self.op.codomain.vdot(self._q, self._q2).real
            self._x_k += self._alpha * self._d
            self._s += -self._alpha * self._q
            self._s2 += -self._alpha * self._q2
            self._rtilde = self.deriv.adjoint(self._s2)
            self._r = self.h_domain.gram_inv(self._rtilde)
            self._inner_prod = self.op.domain.vdot(self._r, self._rtilde).real
            self._beta = self.op.domain.vdot(self._r, self._rtilde).real / self._inner_prod
            self._d = self._r + self._beta * self._d
            self._k += 1
        self.log.info('Inner CG iteration required {} steps.'.format(self._k))
        self.x += self._x_k
        if hasattr(self,'simplified_op'):
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y , self.deriv = self.op.linearize(self.x)

    def nr_inner_its(self):
        return self._k

class NewtonCGFrozen(RegSolver):
    r"""The frozen Newton-CG method. Like Newton-CG but freezes the derivative for some time to avoid 
    recomputing it. 

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The regularization setting includes the operator and penalty and data fidelity functionals.
    data : array-like
        The rhs y of the equation to be solved. Must be in setting.op.codomain.
    init : array-like, optional
        Initial guess to exact solution. (Default: setting.op.domain.zeros())
    cgmaxit : number, optional
        Maximal number of inner CG iterations. (Default: 50)
    rho : number, optional
        A fix number related to the termination (0<rho<1). (Default: 0.8)
    """
    def __init__(self, setting, data, init = None, cgmaxit=50, rho=0.8):
        super().__init__(setting)
        if init is not None and init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        if data not in self.op.codomain:
            raise ValueError(Errors.not_in_vecsp(data,self.op.codomain,vec_name="data",space_name="codomain"))
        self.data = data
        
        self.x = init.copy() if init is not None else self.op.domain.zeros()
        _, self.deriv = self.op.linearize(self.x)
        self._n = 1
        self._op_copy = deepcopy(self.op)
        self._outer_update()
        self.rho = rho
        self.cgmaxit = cgmaxit

    def _outer_update(self):
        if int(self._n / 10) * 10 == self._n:
            _, self.deriv = self.op.linearize(self.x)
        self._x_k = self.op.domain.zeros()
        self.y = self._op_copy(self.x)
        self._residual = self.data - self.y
        self._s = self._residual - self.deriv(self._x_k)
        self._s2 = self.h_codomain.gram(self._s)
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.h_domain.gram_inv(self._rtilde)
        self._d = self._r
        self._inner_prod = self.op.domain.vdot(self._r, self._rtilde).real
        self._norms0 = sqrt(self.op.codomain.vdot(self._s2, self._s).real)
        self._k = 1
        self._n += 1

    def _inner_update(self):
        _, self.deriv = self.op.linearize(self.x)
        self._q = self.deriv(self._d)
        self._q2 = self.h_codomain.gram(self._q)
        self._alpha = self._inner_prod / self.op.codomain.vdot(self._q, self._q2).real
        self._s += -self._alpha * self._q
        self._s2 += -self._alpha * self._q2
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.h_domain.gram_inv(self._rtilde)
        self._beta = self.op.domain.vdot(self._r, self._rtilde).real / self._inner_prod

    def _next(self):
        while (
            sqrt(self.op.codomain.vdot(self._s2, self._s).real) > self.rho * self._norms0
            and self._k <= self.cgmaxit
        ):
            self._inner_update()
            self._x_k += self._alpha * self._d
            self._d = self._r + self._beta * self._d
            self._k += 1
        self.x += self._x_k
        self._outer_update()


class NewtonSemiSmoothFrozen(RegSolver):
    r"""The frozen Newton-CG method. Like Newton-CG adds constraints :math:`\psi_+` and :math:`\psi_-` and efficiently
    only updates the parts needed to be updated. 

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The regularization setting includes the operator and penalty and data fidelity functionals.
    data : array-like
        The data from which to recover. Initializes the rhs y of the equation to be solved. Must 
        be in setting.op.codomain.
    alphas: iterable object or tuple
        Either an iterable giving the grid of alphas or a tuple (alpha0,q)
        In the latter case the sequence :math:`(alpha0*q^n)_{n=0,1,2,...}` is generated.
    psi_minus : scalar
        lower constraint of the minimization. Must be larger then `psi_plus`
    psi_plus : scalar
        upper constraint of the minimization. Must be smaller then `psi_minus`
    init : array-like, optional
        Initial guess to exact solution. (Default: setting.op.domain.zeros())
    xref : array-like, optional
        Reference value in the Tikhonov functional. (Default: setting.op.domain.zeros())
    inner_NSS_iter_max : int, optional
        The number of maximal iterations when solving the linearized problem. (Default: 50)
    cg_pars : dictionary, optional
        Parameters of the CG method for minimizing the Tikhonov functional in the inner 
        Semi-Smooth Newton. (Default: None)
    """
    def __init__(self, setting, data, alphas, psi_minus, psi_plus, init = None, xref =None, inner_NSS_iter_max = 50, cg_pars = None):
        super().__init__(setting)
        if init is not None and init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        if data not in self.op.codomain:
            raise ValueError(Errors.not_in_vecsp(data,self.op.codomain,vec_name="data",space_name="codomain"))
        self.rhs = data
        """The rhs y of the equation to be solved. Initialized by data
        """
        self.x = init if init is not None else setting.op.domain.zeros()
        """The iterate of x.
        """
        self.xref = xref if xref is not None else setting.op.domain.zeros()
        """Reference value in the Tikhonov functional.
        """
        if isinstance(alphas,tuple) and len(alphas)==2:
            self._alphas = GeometricSequence(alphas[0],alphas[1])
        else:
            self._alphas = iter(alphas)
        self.alpha = next(self._alphas)
        r"""Initial regularization parameter :math:`\alpha`.
        """
        self.alpha_old = self.alpha
        self.psi_minus = psi_minus
        """lower constraint of the minimization.
        """
        self.psi_plus = psi_plus
        """upper constraint of the minimization.
        """
        self.cg_pars = cg_pars
        """Parameters passed to inner Semi Smooth Newton for the used Tikhonov Solver. 
        """
        self.inner_NSS_iter_max = inner_NSS_iter_max
        self.y, deriv = self.op.linearize(self.x)
        self.deriv = deepcopy(deriv)
        
        self.active_plus = (self.alpha*(self.x-self.psi_plus ))>=0 
        self.active_minus = (self.alpha*(self.x-self.psi_minus))>=0 

        self.lam_plus = setting.op.domain.zeros()
        self.lam_minus = setting.op.domain.zeros()
        

    def _next(self):
        self.lin_NSS = SemismoothNewton_bilateral(
            Setting(
                self.deriv,
                self.penalty,
                self.data_fid
                ),
            self.rhs-self.y+self.deriv(self.x),
            self.alpha,
            x0 = self.x,
            psi_minus=self.psi_minus,
            psi_plus=self.psi_plus,
            logging_level= "WARNING",
            cg_logging_level="WARNING",
            cg_pars = self.cg_pars
        )
        self.lin_NSS.lam_minus = (self.alpha/self.alpha_old)*self.lam_minus
        self.lin_NSS.lam_plus = (self.alpha/self.alpha_old)*self.lam_plus
        self.lin_NSS.active_minus = self.active_minus
        self.lin_NSS.active_plus = self.active_plus
        self.x, _ = self.lin_NSS.run(
            CountIterations(max_iterations=self.inner_NSS_iter_max)
        )
        self.y , deriv = self.op.linearize(self.x)
        self.deriv = deepcopy(deriv)

        self.lam_minus = self.lin_NSS.lam_minus
        self.lam_plus = self.lin_NSS.lam_plus
        self.active_minus = self.lin_NSS.active_minus         
        self.active_plus = self.lin_NSS.active_plus 
        
        try:
            self.alpha_old = self.alpha
            self.alpha = next(self._alphas)
        except StopIteration:
            return self.converge()



class IterativelyRegularizedNewton(RegSolver):
    r"""General method for iterated linearization of operator and subsequent solution of linearized problem. In each iteration, minimizes

    .. math::
        S_{data}(T'[x_n]h+T(x_n)) + regpar_{n} \cdot R(h+x_n)

    where :math:`T` is a Frechet-differentiable operator, using the given `inner_solver`.
    :math:`regpar_n` is a decreasing geometric sequence of regularization parameters.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem.
    inner_solver : Solver class
        Solver class used for the solution of the linearized problem.
    inner_solver_stoprule: stoprule or callable
        Either a callable that can be applied to a setting to yield a stoprule or a Stoprule which is used for the inner solver.
    inner_solver_pars : dict, optional default: dict()
        Parameter dictionary for the inner solver. Should contain all necessary parameters which are not included in the linearized setting.
    data : array-like, optional default: None
        The measured data. If None it is taken from setting.
    regpar : float, optional default: None
        The initial regularization parameter. Must be positive. If None it is taken from setting.
    regpar_step : float, optional
        The factor by which to reduce the `regpar` in each iteration. Default: :math:`2/3`.
    init : array-like, optional default: None
        The initial guess. If None it is set to the zero array.
    """
    def __init__(
               self, setting, 
               inner_solver,
               inner_solver_stoprule,
               inner_solver_pars=dict(),
               data=None, regpar=None,
                regpar_step=2 / 3, 
                 init=None, 
         ):
        super().__init__(setting)
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        else:
            setting.data=data#sets data in setting if there is no data
        if(regpar is None):
            if(not setting.is_tikhonov):
                raise ValueError(Errors.value_error("Regularization parameter has to be included in setting or given directly."))
            regpar=setting.regpar
        self.data=data
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
        self.inner_solver=inner_solver
        self.inner_solver_pars=inner_solver_pars
        self.inner_solver_stoprule=inner_solver_stoprule

    def _next(self):
        # Linearized setting
        inner_setting=Setting(self.deriv,self.setting.penalty,self.data_fid.shift(data_shift=-self.y),regpar=self.regpar,penalty_shift=-self.x)        
        if(callable(self.inner_solver_stoprule)):
            inner_stoprule=self.inner_solver_stoprule(inner_setting)
        else:
            inner_stoprule=copy(self.inner_solver_stoprule)
        # Running inner solver
        step, _ = self.inner_solver(inner_setting,**self.inner_solver_pars).run(stoprule=inner_stoprule)
        self.x += step
        self.y , self.deriv = self.op.linearize(self.x)
        self.regpar *= self.regpar_step
        self.log.info(f"its.{self.iteration_step_nr}: alpha={self.regpar}")