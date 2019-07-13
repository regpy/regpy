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

    def __mul__(self, other):
        # TODO weighted hilbert spaces, multiplication by constants
        if isinstance(other, HilbertSpace):
            return Product(self, other, flatten=True)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, HilbertSpace):
            return Product(other, self, flatten=True)
        else:
            return NotImplemented


class HilbertPullBack(HilbertSpace):
    def __init__(self, space, op, inverse=None):
        assert op.linear
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


@L2.register(discrs.Product)
def L2Product(discr):
    return Product(*(L2(f) for f in discr.factors), flatten=False)


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


@Sobolev.register(discrs.Product)
def SobolevProduct(discr):
    return Product(*(Sobolev(f) for f in discr.factors), flatten=False)


class Product(HilbertSpace):
    def __init__(self, *factors, weights=None, flatten=False):
        assert all(isinstance(f, HilbertSpace) for f in factors)
        if weights is None:
            weights = [1] * len(factors)
        assert len(weights) == len(factors)
        self.factors = []
        self.weights = []
        for w, f in zip(weights, factors):
            if flatten and isinstance(f, type(self)):
                self.factors.extend(f.factors)
                self.weights.extend(w * fw for fw in f.weights)
            else:
                self.factors.append(f)
                self.weights.append(w)
        super().__init__(discrs.Product(*(f.discr for f in self.factors), flatten=False))

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and
            len(self.factors) == len(other.factors) and
            all(f == g for f, g in zip(self.factors, other.factors)) and
            all(v == w for v, w in zip(self.weights, other.weights))
        )

    @util.memoized_property
    def gram(self):
        blocks = []
        for w, f in zip(self.weights, self.factors):
            if w == 1:
                blocks.append(f.gram)
            else:
                blocks.append(w * f.gram)
        return operators.BlockDiagonal(*blocks)

    @util.memoized_property
    def gram_inv(self):
        blocks = []
        for w, f in zip(self.weights, self.factors):
            if w == 1:
                blocks.append(f.gram_inv)
            else:
                blocks.append(1/w * f.gram_inv)
        return operators.BlockDiagonal(*blocks)


class GenericProduct:
    def __init__(self, *factors, weights=None):
        # TODO flatten GenericProduct factors
        assert all(callable(f) for f in factors)
        assert weights is None or len(weights) == len(factors)
        self.factors = factors
        self.weights = weights

    def __call__(self, discr):
        assert isinstance(discr, discrs.Product)
        assert len(self.factors) == len(discr.factors)
        return Product(
            *(f(d) for f, d in zip(self.factors, discr.factors)),
            weights=self.weights,
            flatten=False
        )
