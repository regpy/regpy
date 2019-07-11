import numpy as np
from functools import singledispatch

from . import discrs
from .. import util, operators


class HilbertSpace:
    def __init__(self, discr):
        assert isinstance(discr, discrs.Discretization)
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


@L2.register(discrs.Discretization)
class L2Generic(HilbertSpace):
    @property
    def gram(self):
        return self.discr.identity

    @property
    def gram_inv(self):
        return self.discr.identity

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.discr == other.discr

# TODO L2 for grids, with proper weights


@singledispatch
def Sobolev(discr, index=1):
    raise NotImplementedError(
        'Sobolev not implemented on {}'.format(type(discr).__qualname__))


@Sobolev.register(discrs.UniformGrid)
class SobolevUniformGrid(HilbertSpace):
    def __init__(self, discr, index):
        super().__init__(discr)
        self.index = index
        self.weights = (1 + np.linalg.norm(discr.dualgrid.coords, axis=0)**2) ** index

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and
            self.discr == other.discr and
            self.index == other.index
        )

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


class Product(HilbertSpace):
    def __init__(self, *factors):
        assert all(isinstance(f, HilbertSpace) for f in factors)
        self.factors = factors
        super().__init__(discrs.Product(*(f.discr for f in factors)))

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and
            len(self.factors) == len(other.factors) and
            all(f == g for f, g in zip(self.factors, other.factors))
        )

    @util.memoized_property
    def gram(self):
        return operators.BlockDiagonal(*(f.gram for f in self.factors))

    @util.memoized_property
    def gram_inv(self):
        return operators.BlockDiagonal(*(f.gram_inv for f in self.factors))


class GenericProduct:
    def __init__(self, *factors):
        assert all(callable(f) for f in factors)
        self.factors = factors

    def __call__(self, discr):
        assert isinstance(discr, discrs.Product)
        assert len(self.factors) == len(discr.factors)
        return Product(*(f(d) for f, d in zip(self.factors, discr.factors)))
