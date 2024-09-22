from regpy.solvers import RegSolver

import logging
import numpy as np

class Landweber(RegSolver):
    r"""The linear Landweber method. Solves the linear, ill-posed equation

    .. math::
        T(x) = g^\delta,

    in Hilbert spaces by gradient descent for the residual
    
    .. math::
        \Vert T(x) - g^\delta\Vert^2,

    where \(\Vert\cdot\Vert)\ is the Hilbert space norm in the codomain, and gradients are computed with
    respect to the Hilbert space structure on the domain.

    The number of iterations is effectively the regularization parameter and needs to be picked
    carefully.

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem.
    data : array-like
        The measured data/right hand side.
    init : array-like
        The initial guess.
    stepsize : float, optional
        The step length; must be chosen not too large. If omitted, it is guessed from the norm of
        the derivative at the initial guess.
    """

    def __init__(self, setting, data, init, stepsize=None):
        super().__init__(setting)
        self.rhs = data
        """The right hand side gets initialized to measured data"""
        self.x = init
        self.y = self.op(self.x)
        norm = setting.op_norm()
        self.stepsize = stepsize or 1 / norm**2
        """The stepsize."""

    def _next(self):
        self._residual = self.y - self.rhs
        self._gy_residual = self.h_codomain.gram(self._residual)
        self._update = self.op.adjoint(self._gy_residual)
        self.x -= self.stepsize * self.h_domain.gram_inv(self._update)
        self.y = self.op(self.x)

        if self.log.isEnabledFor(logging.INFO):
            norm_residual = np.sqrt(np.real(np.vdot(self._residual, self._gy_residual)))
            self.log.info('|residual| = {}'.format(norm_residual))
