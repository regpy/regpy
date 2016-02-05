import logging
import numpy as np

from . import Solver

__all__ = ['Landweber']

log = logging.getLogger(__name__)


class Landweber(Solver):
    """The Landweber method.

    Solves the potentially non-linear, ill-posed equation ::

        T(x) = y,

    where `T` is a Frechet-differentiable operator. The number of iterations is
    effectively the regularization parameter and needs to be picked carefully.

    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    stepsize : float, optional
        The step length; must be chosen not too large. If omitted, it is
        guessed from the norm of the derivative at the initial guess.

    Attributes
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    x : array
        The current point.
    y : array
        The value at the current point.
    deriv : :class:`LinearOperator <itreg.operators.LinearOperator>`
        The derivative of the operator at the current point.
    stepsize : float
        The step length to be used in the next step.

    """

    def __init__(self, op, data, init, stepsize=None):
        super().__init__(log)
        self.op = op
        self.data = data
        self.setx(init)
        self.stepsize = stepsize or 1 / self.deriv.norm()

    def setx(self, x):
        """Set the current point of the solver.

        Update the function value and the derivative accordingly.

        Parameters
        ----------
        x : array
            The new point.

        """
        self.x = x
        self.y = self.op(self.x)
        self.deriv = self.op.derivative()

    def next(self):
        """Run a single Landweber iteration.

        Returns
        -------
        bool
            Always True, as the Landweber method never stops on its own.

        """
        residual = self.y - self.data
        gy_residual = self.op.domy.gram(residual)
        self.x -= self.stepsize * self.op.domx.gram_inv(
            self.deriv.adjoint(gy_residual))
        self.setx(self.x)

        if log.isEnabledFor(logging.INFO):
            norm_residual = np.sqrt(np.real(
                np.vdot(residual[:], gy_residual[:])))
            log.info('|residual| = {}'.format(norm_residual))

        return True
