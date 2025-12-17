from math import sqrt

from regpy.util import Errors

from ..general import RegSolver

__all__ = ["Landweber"]

class Landweber(RegSolver):
    r"""The Landweber method. Solves the potentially non-linear, ill-posed equation

    .. math::
        F(x) = g^\delta,

    where :math:`T` is a Frechet-differentiable operator, by gradient descent for the residual
    
    .. math::
        \Vert F(x) - g^\delta\Vert^2,

    where :math:`\Vert\cdot\Vert` is the Hilbert space norm in the codomain, and gradients are computed with
    respect to the Hilbert space structure on the domain.

    The number of iterations is effectively the regularization parameter and needs to be picked
    carefully.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem.
    init : array-like
        The initial guess.
    data : array-like, default None
        The measured data/right hand side. If None it is taken from setting.
    stepsize : float, optional
        The step length; must be chosen not too large. If omitted, it is guessed from the norm of
        the derivative at the initial guess. Alternatively, if backtracking is used, stepsize defines an
        initial guess for the step length which has to be sufficiently large. If omitted the initial step
        length will be 10**16.
    backtracking : boolean, optional
        Wether or not to use backtracking for finding a sufficient step length. Default: True.
    """

    def __init__(self, setting,init,data=None, stepsize=None, backtracking=True, eta = 0.5, op_norm_method = "lanczos"):
        super().__init__(setting)
        if self.op.linear:
            self.log.warning("Using non-linear Landweber with a linear Operator! Consider using the linear Landweber in the module solvers.linear")
        if init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        self.rhs = data
        """The right hand side gets initialized with the measured data."""
        self.x = init
        self.y, deriv = self.op.linearize(self.x)
        self.deriv = deriv
        """The derivative at the current iterate."""

        if backtracking:
            self.backtracking = True
            self.stepsize = stepsize or 10**16
            """The stepsize."""
        else:
            self.backtracking = False
            self.stepsize = stepsize or 0.9 / self.deriv.norm(setting.h_domain,setting.h_codomain, method = op_norm_method)**2
        
        self.eta = eta
        """Factor for decreasing the stepsize."""
        if not (0<self.eta<1):
            raise ValueError(Errors.value_error("The Step size reduction constant must be between 0 and 1!"))

        if self.backtracking:
            self._residual = self.y - self.rhs
            self._gy_residual = self.h_codomain.gram(self._residual)
            self._old_err = self.op.codomain.vdot(self._residual, self._gy_residual).real

    def _next(self):
        if not self.backtracking:
            self._residual = self.y - self.rhs
            self._gy_residual = self.h_codomain.gram(self._residual)
        self._update = self.deriv.adjoint(self._gy_residual)
        
        while self.backtracking:
            new_x = self.x - self.stepsize * self.h_domain.gram_inv(self._update)
            self._residual = self.op(new_x) - self.rhs
            self._gy_residual = self.h_codomain.gram(self._residual)
            new_err = self.op.codomain.vdot(self._residual, self._gy_residual).real
            if new_err < self._old_err:
                self._old_err = new_err
                break
            else:
                self.stepsize *= self.eta
        
        if self.backtracking:
            self.x = new_x
        else:
            self.x -= self.stepsize * self.h_domain.gram_inv(self._update)
        self.y, self.deriv = self.op.linearize(self.x)


        if self.log.isEnabledFor(20): # INFO=20
            if self.backtracking:
                norm_residual = sqrt(self._old_err)
            else:
                norm_residual = sqrt((self.op.codomain.vdot(self._residual, self._gy_residual)).real)
            self.log.info('|residual| = {}'.format(norm_residual))
