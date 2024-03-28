from regpy.solvers import Solver

import logging
import numpy as np

class Landweber(Solver):
    r"""The Landweber method. Solves the potentially non-linear, ill-posed equation
    $$
        T(x) = g^\delta,
    $$
    where $T$ is a Frechet-differentiable operator, by gradient descent for the residual
    $$
        \Vert T(x) - g^\delta\Vert^2,
    $$
    where $\Vert\cdot\Vert$ is the Hilbert space norm in the codomain, and gradients are computed with
    respect to the Hilbert space structure on the domain.

    The number of iterations is effectively the regularization parameter and needs to be picked
    carefully.

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem.
    rhs : array-like
        The right hand side.
    init : array-like
        The initial guess.
    stepsize : float, optional
        The step length; must be chosen not too large. If omitted, it is guessed from the norm of
        the derivative at the initial guess.
    """

    def __init__(self, setting, rhs, init, stepsize=None):
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.rhs = rhs
        """The right hand side."""
        self.x = init
        self.y, deriv = self.setting.op.linearize(self.x)
        self.deriv = deriv
        """The derivative at the current iterate."""
        # compute norm of deriv.adjoint*deriv by power method
        h = deriv.domain.rand()
        norm = np.sqrt(np.real(np.vdot(h, h)))
        for count in range(10):
            h = h / norm
            derivh = deriv(h)
            derivh = setting.h_codomain.gram(derivh)
            h = deriv.adjoint(derivh)
            h = setting.h_domain.gram_inv(h)
            norm = np.sqrt(np.real(np.vdot(h, h)))
        self.stepsize = stepsize or 1 / norm
        """The stepsize."""

    def _next(self):
        self._residual = self.y - self.rhs
        self._gy_residual = self.setting.h_codomain.gram(self._residual)
        self._update = self.deriv.adjoint(self._gy_residual)
        self.x -= self.stepsize * self.setting.h_domain.gram_inv(self._update)
        self.y, self.deriv = self.setting.op.linearize(self.x)

        if self.log.isEnabledFor(logging.INFO):
            norm_residual = np.sqrt(np.real(np.vdot(self._residual, self._gy_residual)))
            self.log.info('|residual| = {}'.format(norm_residual))
