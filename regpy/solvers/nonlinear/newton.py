import numpy as np
from scipy.sparse import linalg as spla
from copy import deepcopy
from regpy.solvers import RegSolver, RegularizationSetting
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

class NewtonCG(RegSolver):
    r"""The Newton-CG method. Solves the potentially non-linear, ill-posed equation:
    
    .. math::
        T(x) = y,

    where \(T)\ is a Frechet-differentiable operator. The Newton equations are solved by the
    conjugate gradient method applied to the normal equation (CGNE) using the regularizing
    properties of CGNE with early stopping (see Hanke 1997).

    If simplified_op is specified, it will be used to generate an approximation of the derivative 
    of the forward operator setting.op, which may be cheaper to evaluate. E.g., it may be the 
    derivative at the initial guess, which would yield a frozen Newton method. 

    Parameters
    ----------
    setting : RegularizationSetting
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
        assert isinstance(setting,RegularizationSetting)
        super().__init__(setting)
        self.data = data
        """The measured data."""
        if init is None:
            init = self.op.domain.zeros()
        """The initial guess."""
        self.x = np.copy(init)
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
        self._norms0 = np.sqrt(np.vdot(self._s2, self._s).real)
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.h_domain.gram_inv(self._rtilde)
        self._d = self._r
        self._inner_prod = np.vdot(self._r, self._rtilde).real
     
        while (self._k==0 or (np.sqrt(np.vdot(self._s2, self._s).real)
               > self.rho * self._norms0 and self._k < self.cgmaxit)):
            self._q = self.deriv(self._d)
            self._q2 = self.h_codomain.gram(self._q)
            self._alpha = self._inner_prod / np.vdot(self._q, self._q2).real
            self._x_k += self._alpha * self._d
            self._s += -self._alpha * self._q
            self._s2 += -self._alpha * self._q2
            self._rtilde = self.deriv.adjoint(self._s2)
            self._r = self.h_domain.gram_inv(self._rtilde)
            self._inner_prod = np.vdot(self._r, self._rtilde).real
            self._beta = np.vdot(self._r, self._rtilde).real / self._inner_prod
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

from regpy.solvers.linear.semismoothNewton import SemismoothNewton_bilateral
from regpy.solvers.linear.tikhonov import GeometricSequence, TikhonovCG
from regpy.stoprules import CountIterations
class NewtonSemiSmoothFrozen(RegSolver):
    r"""The frozen Newton-CG method. Like Newton-CG adds constraints \(\psi_+)\ and \(\psi_-)\ and efficiently
    only updates the parts needed to be updated. 

    Parameters
    ----------
    setting : RegularizationSetting
        The regularization setting includes the operator and penalty and data fidelity functionals.
    data : array-like
        The data from which to recover. Initializes the rhs y of the equation to be solved. Must 
        be in setting.op.codomain.
    alphas: iterable object or tuple
        Either an iterable giving the grid of alphas or a tuple (alpha0,q)
        In the latter case the seuqence :math:`(alpha0*q^n)_{n=0,1,2,...}` is generated.
    psi_minus : np.number
        lower constraint of the minimization. Must be larger then `psi_plus`
    psi_plus : np.number
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
        assert isinstance(setting,RegularizationSetting)
        super().__init__(setting)
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
            RegularizationSetting(
                self.deriv,
                self.penalty,
                self.data_fid
                ),
            self.rhs-self.y+self.deriv(self.x),
            self.alpha,
            x0 = self.x,
            psi_minus=self.psi_minus,
            psi_plus=self.psi_plus,
            logging_level= logging.WARNING,
            cg_logging_level=logging.WARNING,
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
