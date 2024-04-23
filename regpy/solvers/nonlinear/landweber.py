from regpy.solvers import Solver
from regpy.operators import SciPyLinearOperator
from scipy.sparse.linalg import eigsh

import logging
import numpy as np

class Landweber(Solver):
    r"""The Landweber method. Solves the potentially non-linear, ill-posed equation
    \[
        F(x) = g^\delta,
    \]
    where \(T)\ is a Frechet-differentiable operator, by gradient descent for the residual
    \[
        \Vert F(x) - g^\delta\Vert^2,
    \]
    where \(\Vert\cdot\Vert)\ is the Hilbert space norm in the codomain, and gradients are computed with
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
        super().__init__(setting)
        self.rhs = rhs
        """The right hand side."""
        self.x = init
        self.y, deriv = self.op.linearize(self.x)
        self.deriv = deriv
        """The derivative at the current iterate."""
        gramX = self.h_domain.gram
        gramY = self.h_codomain.gram
        norm =eigsh(SciPyLinearOperator(self.deriv.adjoint * gramY * self.deriv), 1, M=SciPyLinearOperator(gramX),tol=0.01)[0][0]

        self.stepsize = stepsize or 0.9 / norm
        """The stepsize."""

    def _next(self):
        self._residual = self.y - self.rhs
        self._gy_residual = self.h_codomain.gram(self._residual)
        self._update = self.deriv.adjoint(self._gy_residual)
        self.x -= self.stepsize * self.h_domain.gram_inv(self._update)
        self.y, self.deriv = self.op.linearize(self.x)

        if self.log.isEnabledFor(logging.INFO):
            norm_residual = np.sqrt(np.real(np.vdot(self._residual, self._gy_residual)))
            self.log.info('|residual| = {}'.format(norm_residual))
