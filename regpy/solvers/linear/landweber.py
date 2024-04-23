from regpy.solvers import RegSolver
from regpy.operators import SciPyLinearOperator
from scipy.sparse.linalg import eigsh

import logging
import numpy as np

class Landweber(RegSolver):
    r"""The linear Landweber method. Solves the linear, ill-posed equation
    \[
        T(x) = g^\delta,
    \]
    in Hilbert spaces by gradient descent for the residual
    \[
        \Vert T(x) - g^\delta\Vert^2,
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
        T = self.op
        gramX = self.h_domain.gram
        gramY = self.h_codomain.gram
        self.x = init
        self.y = T(self.x)
        norm =eigsh(SciPyLinearOperator(T.adjoint * gramY * T), 1, M=SciPyLinearOperator(gramX),tol=0.01)[0][0]
        self.stepsize = stepsize or 1 / norm
        """The stepsize."""

    def _next(self):
        T = self.op
        gramX_inv = self.h_domain.gram_inv
        gramY = self.h_codomain.gram
        self._residual = self.y - self.rhs
        self._gy_residual = gramY(self._residual)
        self._update = T.adjoint(self._gy_residual)
        self.x -= self.stepsize * gramX_inv(self._update)
        self.y = T(self.x)

        if self.log.isEnabledFor(logging.INFO):
            norm_residual = np.sqrt(np.real(np.vdot(self._residual, self._gy_residual)))
            self.log.info('|residual| = {}'.format(norm_residual))
