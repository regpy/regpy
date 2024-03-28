import numpy as np
from scipy.sparse import linalg as spla

from regpy.solvers import Solver


class NewtonCG(Solver):
    r"""The Newton-CG method. Solves the potentially non-linear, ill-posed equation:
    $$
        T(x) = y,
    $$
    where $T$ is a Frechet-differentiable operator. The Newton equations are solved by the
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
    simplified_op : Operator, optional
        Simplified operator to be used for the derivative. (Default: None)
    """

    def __init__(self, setting, data, init=None, cgmaxit=50, rho=0.8, simplified_op = None):
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.data = data
        """The measured data."""
        if init is None:
            init = self.setting.op.domain.zeros()
        """The initial guess."""
        self.x = np.copy(init)
        if simplified_op:
            self.simplified_op = simplified_op
            """Simplified operator for derivative.
            """
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.setting.op(self.x)
        else:
            self.y, self.deriv = self.setting.op.linearize(self.x)
        self.rho = rho
        """A fix number related to the termination (0<rho<1)."""
        self.cgmaxit = cgmaxit
        """Maximum number of iterations for inner CG solver."""
        self._k = 0
    
    def _next(self):
        self._k = 0
        self._s = self.data - self.y  
        # aux plays the role of s here to avoid storage for another vector in codomain
        self._x_k = self.setting.op.domain.zeros()
        # self._s += - self.deriv(self._x_k)
        self._s2 = self.setting.h_codomain.gram(self._s)
        self._norms0 = np.sqrt(np.vdot(self._s2, self._s).real)
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.setting.h_domain.gram_inv(self._rtilde)
        self._d = self._r
        self._inner_prod = np.vdot(self._r, self._rtilde).real
     
        while (self._k==0 or (np.sqrt(np.vdot(self._s2, self._s).real)
               > self.rho * self._norms0 and self._k < self.cgmaxit)):
            self._q = self.deriv(self._d)
            self._q2 = self.setting.h_codomain.gram(self._q)
            self._alpha = self._inner_prod / np.vdot(self._q, self._q2).real
            self._x_k += self._alpha * self._d
            self._s += -self._alpha * self._q
            self._s2 += -self._alpha * self._q2
            self._rtilde = self.deriv.adjoint(self._s2)
            self._r = self.setting.h_domain.gram_inv(self._rtilde)
            self._inner_prod = np.vdot(self._r, self._rtilde).real
            self._beta = np.vdot(self._r, self._rtilde).real / self._inner_prod
            self._d = self._r + self._beta * self._d
            self._k += 1
        self.log.info('Inner CG iteration required {} steps.'.format(self._k))
        self.x += self._x_k
        if hasattr(self,'simplified_op'):
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.setting.op(self.x)
        else:
            self.y , self.deriv = self.setting.op.linearize(self.x)

    def nr_inner_its(self):
        return self._k

class NewtonCGFrozen(Solver):
    r"""The frozen Newton-CG method. Like Newton-CG but freezes the derivative for some time to avoid 
    recomputing it. 

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
    """
    def __init__(self, setting, data, init, cgmaxit=50, rho=0.8):
        super().__init__()
        self.setting = setting
        self.op = setting.op
        self.data = data
        self.x = init
        _, self.deriv = self.op.linearize(self.x)
        self._n = 1
        self._outer_update()
        self.rho = rho
        self.cgmaxit = cgmaxit

    def _outer_update(self):
        if int(self._n / 10) * 10 == self._n:
            _, self.deriv = self.op.linearize(self.x)
        self._x_k = self.op.domain.zeros()
        #        self._x_k = 1j*np.zeros(np.shape(self.x))
        self.y = self.op(self.x)
        self._residual = self.data - self.y
        #        _, self.deriv=self.op.linearize(self.x)
        self._s = self._residual - self.deriv(self._x_k)
        self._s2 = self.setting.codomain.gram(self._s)
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.setting.domain.gram_inv(self._rtilde)
        self._d = self._r
        self._inner_prod = self.setting.domain.inner(self._r, self._rtilde)
        self._norms0 = np.sqrt(np.real(self.setting.domain.inner(self._s2, self._s)))
        self._k = 1
        self._n += 1

    def _inner_update(self):
        _, self.deriv = self.op.linearize(self.x)
        self._q = self.deriv(self._d)
        self._q2 = self.setting.codomain.gram(self._q)
        self._alpha = (self._inner_prod
                       / np.real(self.setting.codomain.inner(self._q, self._q2)))
        self._s2 += -self._alpha * self._q2
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.setting.domain.gram_inv(self._rtilde)
        self._beta = (np.real(self.setting.codomain.inner(self._r, self._rtilde))
                      / self._inner_prod)

    def _next(self):
        while (
            np.sqrt(self.setting.domain.inner(self._s2, self._s)) > self.rho * self._norms0
            and self._k <= self.cgmaxit
        ):
            self._inner_update()
            self._x_k += self._alpha * self._d
            self._d = self._r + self._beta * self._d
            self._k += 1
        self.x += self._x_k
        self._outer_update()


class NewtonSemiSmooth(Solver):
    r"""The frozen Newton-CG method. Like Newton-CG adds constraints $\psi_+$ and $\psi_-$ and efficiently
    only updates the parts needed to be updated. 

    Parameters
    ----------
    setting : RegularizationSetting
        The regularization setting includes the operator and penalty and data fidelity functionals.
    rhs : array-like
        The rhs y of the equation to be solved. Must be in setting.op.codomain.
    init : array-like, optional
        Initial guess to exact solution. (Default: setting.op.domain.zeros())
    alpha : number, optional
        Initial regularization parameter $\alpha$.
    psi_minus : np.number
        lower constraint of the minimization. Must be larger then `psi_plus`
    psi_plus : np.number
        upper constraint of the minimization. Must be smaller then `psi_minus`
    """
    def __init__(self, setting, rhs, init, alpha, psi_minus, psi_plus):
        super().__init__()
        self.setting = setting
        """The regularization setting includes the operator and penalty and data fidelity functionals.
        """
        self.rhs = rhs
        """The rhs y of the equation to be solved.
        """
        self.x = init
        self.alpha = alpha
        """Initial regularization parameter $\alpha$.
        """
        self.psi_minus = psi_minus
        """lower constraint of the minimization.
        """
        self.psi_plus = psi_plus
        """upper constraint of the minimization.
        """

        self.size = init.shape[0]

        self.y = self.setting.op(self.x)

        self.b = self.setting.op.adjoint(self.rhs) + self.alpha * init

        self.lam_plus = np.maximum(np.zeros(self.size), self.b - self._A(self.x))
        self.lam_minus = -np.minimum(np.zeros(self.size), self.b - self._A(self.x))

        # sets where the upper constraint and the lower constarint are active
        self.active_plus = [self.lam_plus[j] + self.alpha * (self.x[j] - self.psi_plus) > 0 for j in
                            range(self.size)]
        self.active_minus = [self.lam_minus[j] - self.alpha * (self.x[j] - self.psi_minus) > 0 for j
                             in range(self.size)]

        # complte active and inactive sets, need to be computed in each step again
        self.active = np.empty(self.size)
        self.inactive = np.empty(self.size)

    def _next(self):
        self.active = [self.active_plus[j] or self.active_minus[j] for j in range(self.size)]
        self.inactive = [self.active[j] == False for j in range(self.size)]

        # On the active sets the solution takes the values of the constraints
        self.x[self.active_plus] = self.psi_plus
        self.x[self.active_minus] = self.psi_minus

        self.lam_plus[self.inactive] = 0
        self.lam_plus[self.active_minus] = 0
        self.lam_minus[self.inactive] = 0
        self.lam_minus[self.active_plus] = 0

        # A as spla.LinearOperator constrained to inactive set
        A_inactive = spla.LinearOperator(
            (np.count_nonzero(self.inactive), np.count_nonzero(self.inactive)),
            matvec=self._A_inactive,
            dtype=float)
        # Solve system on the different sets
        self.x[self.inactive] = self._gmres(A_inactive,
                                            self.b[self.inactive] + self.lam_minus[self.inactive] -
                                            self.lam_plus[self.inactive])
        z = self._A(self.x)
        self.lam_plus[self.active_plus] = self.b[self.active_plus] + self.lam_minus[
            self.active_plus] - z[self.active_plus]
        self.lam_minus[self.active_minus] = -self.b[self.active_minus] + self.lam_plus[
            self.active_minus] + z[self.active_minus]

        # Update active and inactive sets
        self.y = self.setting.op(self.x)
        self.active_plus = [self.lam_plus[j] + self.alpha * (self.x[j] - self.psi_plus) > 0 for j in
                            range(self.size)]
        self.active_minus = [self.lam_minus[j] - self.alpha * (self.x[j] - self.psi_minus) > 0 for j
                             in range(self.size)]

    def _gmres(self, op, rhs):
        result, info = spla.gmres(op, rhs.ravel())
        if info > 0:
            self.log.warn('Gmres failed to converge')
        elif info < 0:
            self.log.warn('Illegal Gmres input or breakdown')
        return result

    def _A(self, u):
        self.y = self.setting.op(u)
        return self.alpha * u + self.setting.op.adjoint(self.y)

    def _A_inactive(self, u):
        projection = np.zeros(self.size)
        projection[self.inactive] = u
        return self._A(projection)[self.inactive]
