import logging
import numpy as np

__all__ = ['Space']


class Space:
    def __init__(self, shape, log=logging.getLogger()):
        self.shape = shape
        self.log = log

    def gram(self, x):
        raise NotImplementedError()

    def gram_inv(self, x):
        raise NotImplementedError()

    def inner(self, x, y):
        return np.vdot(x[:], self.gram(y)[:])

    def norm(self, x):
        return np.sqrt(np.real(self.inner(x, x)))
