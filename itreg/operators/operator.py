import logging
import numpy as np

__all__ = ['Operator', 'LinearOperator']


class Operator(object):
    def __init__(self, domx, domy=None, log=logging.getLogger()):
        self.domx = domx
        self.domy = domy or domx
        self.log = log

    def __call__(self, x):
        raise NotImplementedError()

    def derivative(self):
        raise NotImplementedError()


class LinearOperator(Operator):
    def adjoint(self, x):
        raise NotImplementedError()

    def derivative(self):
        return self

    def abs_squared(self, x):
        z = self(x)
        z = self.domy.gram(z)
        z = self.adjoint(z)
        z = self.domx.gram_inv(z)
        return z

    def norm(self, iterations=10):
        h = np.random.rand(*self.domx.shape)
        nrm = np.sqrt(np.sum(h**2))
        for i in range(iterations):
            h = h / nrm
            h = self.abs_squared(h)
            nrm = np.sqrt(np.sum(h**2))
        return np.sqrt(nrm)
