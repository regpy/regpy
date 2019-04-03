from . import Solver

import logging
import numpy as np


class Landweber(Solver):
    """The Landweber method.

    Solves the potentially non-linear, ill-posed equation:

        .. math:: T(x) = y,

    where :math:`T` is a Frechet-differentiable operator. The number of
    iterations is effectively the regularization parameter and needs to be
    picked carefully.

    Parameters
    ----------
    op : :class:`~itreg.operators.Operator`
        The forward operator.
    rhs : array
        The right hand side.
    init : array
        The initial guess.
    stepsize : float, optional
        The step length; must be chosen not too large. If omitted, it is
        guessed from the norm of the derivative at the initial guess.

    Attributes
    ----------
    op : :class:`~itreg.operators.Operator`
        The forward operator.
    rhs : array
        The right hand side.
    stepsize : float
        The step length to be used in the next step.
    """

    def __init__(self, op, rhs, init, stepsize=None):
        super().__init__()
        self.op = op
        self.rhs = rhs
        self.x = init
        self.y, self.deriv = self.op.linearize(self.x)
        self.stepsize = stepsize or 1 / self.deriv.norm()

    def _next(self):
        residual = self.y - self.rhs
        gy_residual = self.op.range.gram(residual)

        self.x -= self.stepsize * self.op.domain.gram_inv(self.deriv.adjoint(gy_residual))
        self.y, self.deriv = self.op.linearize(self.x)

        if self.log.isEnabledFor(logging.INFO):
            norm_residual = np.sqrt(np.real(np.vdot(residual, gy_residual)))
            self.log.info('|residual| = {}'.format(norm_residual))
