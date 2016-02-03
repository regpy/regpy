import logging
import numpy as np

from .operator import LinearOperator

__all__ = ['Volterra']

log = logging.getLogger(__name__)


class Volterra(LinearOperator):
    def __init__(self, domx, domy=None, spacing=1):
        super().__init__(domx, domy, log)
        assert(len(self.domx.shape) == 1)
        assert(self.domx.shape == self.domy.shape)
        self.spacing = spacing

    def __call__(self, x):
        return self.spacing * np.cumsum(x)

    def adjoint(self, x):
        return self.spacing * np.flipud(np.cumsum(np.flipud(x)))
