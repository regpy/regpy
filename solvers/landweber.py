import logging
import numpy as np

from .solver import Solver

__all__ = ['Landweber']

log = logging.getLogger(__name__)


class Landweber(Solver):
    def __init__(self, op, data, init, stepsize=None):
        super().__init__(log)
        self.op = op
        self.data = data
        self.setx(init)
        self.stepsize = stepsize or (1 / self.deriv.norm())

    def setx(self, x):
        self.x = x
        self.y = self.op(self.x)
        self.deriv = self.op.derivative()

    def next(self):
        residual = self.y - self.data
        gy_residual = self.op.domy.gram(residual)
        self.x -= self.stepsize * self.op.domx.gram_inv(self.deriv.adjoint(gy_residual))
        self.setx(self.x)

        if log.isEnabledFor(logging.INFO):
            norm_residual = np.sqrt(np.real(np.vdot(residual[:], gy_residual[:])))
            log.info('|residual| = {}'.format(norm_residual))

        return True
