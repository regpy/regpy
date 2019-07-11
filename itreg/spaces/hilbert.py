import numpy as np
from functools import singledispatch

from . import discr
from .. import util, operators


class HilbertSpace:
    def __init__(self, discr):
        assert isinstance(discr, discr.Discretization)
        self.discr = discr

    @property
    def gram(self):
        """The gram matrix as a LinearOperator
        """
        raise NotImplementedError

    @property
    def gram_inv(self):
        """The inverse of the gram matrix as a LinearOperator
        """
        raise NotImplementedError

    def inner(self, x, y):
        """Compute the inner product between to elements.

        This is a convenience wrapper around :meth:`gram`.

        Parameters
        ----------
        x, y : arrays
            The elements for which the inner product should be computed.

        Returns
        -------
        float
            The inner product.
        """
        return np.real(np.vdot(x, self.gram(y)))

    def norm(self, x):
        """Compute the norm of an element.

        This is a convenience wrapper around :meth:`norm`.

        Parameters
        ----------
        x : array
            The elements for which the norm should be computed.

        Returns
        -------
        float
            The norm.
        """
        return np.sqrt(self.inner(x, x))


class HilbertPullBack(HilbertSpace):
    def __init__(self, space, op, inverse=None):
        assert isinstance(op, operators.LinearOperator)
        if not isinstance(space, HilbertSpace) and callable(space):
            space = space(op.codomain)
        assert isinstance(space, HilbertSpace)
        assert op.codomain == space.discr
        self.op = op
        self.space = space
        self.discr = op.domain
        if not inverse:
            self.inverse = None
        elif inverse == 'conjugate':
            self.log.info(
                'Note: Using using T* G^{-1} T as inverse of T* G T. This is probably not correct.')
            self.inverse = op.adjoint * space.gram_inv * op
        elif inverse == 'cholesky':
            self.inverse = operators.CholeskyInverse(self.gram)

    @util.memoized_property
    def gram(self):
        return self.op.adjoint * self.space.gram * self.op

    @property
    def gram_inv(self):
        if self.inverse:
            return self.inverse
        raise NotImplementedError


@singledispatch
def L2(discr):
    raise NotImplementedError(
        'L2 not implemented on {}'.format(type(discr).__qualname__))


@L2.register(discr.Discretization)
class L2Generic(HilbertSpace):
    @property
    def gram(self):
        return self.discr.identity

    @property
    def gram_inv(self):
        return self.discr.identity

# TODO L2 for grids, with proper weights


@singledispatch
def Sobolev(discr, index=1):
    raise NotImplementedError(
        'Sobolev not implemented on {}'.format(type(discr).__qualname__))


@Sobolev.register(discr.UniformGrid)
class SobolevUniformGrid(HilbertSpace):
    def __init__(self, discr, index):
        super().__init__(discr)
        self.index = index
        self.weights = (1 + np.linalg.norm(discr.dualgrid.coords, axis=0)**2) ** index

    @util.memoized_property
    def gram(self):
        ft = operators.FourierTransform(self.discr)
        mul = operators.PointwiseMultiplication(self.discr.dualgrid, self.weights)
        return ft.adjoint * mul * ft

    @util.memoized_property
    def gram_inv(self):
        ft = operators.FourierTransform(self.discr)
        mul = operators.PointwiseMultiplication(self.discr.dualgrid, 1/self.weights)
        return ft.adjoint * mul * ft
