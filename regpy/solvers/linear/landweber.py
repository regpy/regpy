from math import sqrt

from regpy.util import Errors
from ..general import RegSolver

__all__ = ["Landweber"]

class Landweber(RegSolver):
    r"""The linear Landweber method. Solves the linear, ill-posed equation

    .. math::
        T(x) = g^\delta,

    in Hilbert spaces by gradient descent for the residual
    
    .. math::
        \Vert T(x) - g^\delta\Vert^2,

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
        the derivative at the initial guess.
    """

    def __init__(self, setting, init,data=None, stepsize=None, norm_method = None):
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="The linear Landweber requires the operator to be linear! Use the Landweber from non-linear module!"))
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        if data not in self.op.codomain:
            raise ValueError(Errors.not_in_vecsp(data,self.op.codomain,vec_name="data",space_name="codomain"))
        if init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        self.rhs = data
        """The right hand side gets initialized to measured data"""
        self.x = init
        self.y = self.op(self.x)
        norm = setting.op.norm(setting.h_domain,setting.h_codomain, method = norm_method)
        self.stepsize = stepsize or 1 / norm**2
        """The stepsize."""

    def _next(self):
        self._residual = self.y - self.rhs
        self._gy_residual = self.h_codomain.gram(self._residual)
        self._update = self.op.adjoint(self._gy_residual)
        self.x -= self.stepsize * self.h_domain.gram_inv(self._update)
        self.y = self.op(self.x)

        if self.log.isEnabledFor(20): # INFO = 20
            norm_residual = sqrt((self.op.codomain.vdot(self._residual, self._gy_residual)).real)
            self.log.info('|residual| = {}'.format(norm_residual))
