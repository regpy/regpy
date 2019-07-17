from copy import copy
import numpy as np

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

    def __add__(self, other):
        if isinstance(other, HilbertSpace):
            return DirectSum(self, other, flatten=True)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, HilbertSpace):
            return DirectSum(other, self, flatten=True)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if np.isreal(other):
            return DirectSum((other, self), flatten=True)
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


class DirectSum(HilbertSpace):
    def __init__(self, *args, flatten=False):
        self.summands = []
        self.weights = []
        for arg in args:
            if isinstance(arg, tuple):
                w, s = arg
            else:
                w, s = 1, arg
            assert w > 0
            assert isinstance(s, HilbertSpace)
            if flatten and isinstance(s, type(self)):
                self.summands.extend(s.summands)
                self.weights.extend(w * sw for sw in s.weights)
            else:
                self.summands.append(s)
                self.weights.append(w)
        super().__init__(discrs.DirectSum(*(s.discr for s in self.summands), flatten=False))

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and
            len(self.summands) == len(other.summands) and
            all(s == t for s, t in zip(self.summands, other.summands)) and
            all(v == w for v, w in zip(self.weights, other.weights))
        )

    @util.memoized_property
    def gram(self):
        ops = []
        for w, s in zip(self.weights, self.summands):
            if w == 1:
                ops.append(s.gram)
            else:
                ops.append(w**2 * s.gram)
        if len(ops) == 1:
            return ops[0]
        else:
            return operators.DirectSum(*ops)

    @util.memoized_property
    def gram_inv(self):
        ops = []
        for w, s in zip(self.weights, self.summands):
            if w == 1:
                ops.append(s.gram_inv)
            else:
                ops.append(1/w**2 * s.gram_inv)
        if len(ops) == 1:
            return ops[0]
        else:
            return operators.DirectSum(*ops)


class AbstractSpace:
    def __add__(self, other):
        if callable(other):
            return AbstractSum(self, other, flatten=True)
        else:
            return NotImplemented

    def __radd__(self, other):
        if callable(other):
            return AbstractSum(other, self, flatten=True)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if np.isreal(other):
            return AbstractSum((other, self), flatten=True)
        else:
            return NotImplemented


class AbstractSpaceDispatcher(AbstractSpace):
    def __init__(self, name):
        self._registry = {}
        self.name = name
        self.args = {}

    def register(self, discr_type):
        def decorator(impl):
            self._registry[discr_type] = impl
            return impl
        return decorator

    def __call__(self, discr=None, **kwargs):
        if discr is None:
            clone = copy(self)
            clone.args = copy(self.args)
            clone.args.update(kwargs)
            return clone
        for cls in type(discr).mro():
            try:
                impl = self._registry[cls]
            except KeyError:
                continue
            kws = copy(self.args)
            kws.update(kwargs)
            return impl(discr, **kws)
        raise NotImplementedError(
            '{} not implemented on {}'.format(self.name, discr)
        )


class AbstractSum(AbstractSpace):
    def __init__(self, *args, flatten=False):
        self.summands = []
        self.weights = []
        for arg in args:
            if isinstance(arg, tuple):
                w, s = arg
            else:
                w, s = 1, arg
            assert w > 0
            assert callable(s)
            if flatten and isinstance(s, type(self)):
                self.summands.extend(s.summands)
                self.weights.extend(w * sw for sw in s.weights)
            else:
                self.summands.append(s)
                self.weights.append(w)

    def __call__(self, discr):
        assert isinstance(discr, discrs.DirectSum)
        assert len(self.summands) == len(discr.summands)
        return DirectSum(
            *((w, s(d)) for w, s, d in zip(self.weights, self.summands, discr.summands)),
            flatten=False
        )


L2 = AbstractSpaceDispatcher('L2')


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


@L2.register(discrs.UniformGrid)
class L2UniformGrid(HilbertSpace):
    @util.memoized_property
    def gram(self):
        return self.discr.volume_elem * self.discr.identity

    @util.memoized_property
    def gram_inv(self):
        return 1/self.discr.volume_elem * self.discr.identity


@L2.register(discrs.DirectSum)
def L2DirectSum(discr):
    return DirectSum(*(L2(s) for s in discr.summands), flatten=False)


Sobolev = AbstractSpaceDispatcher('Sobolev')


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
        mul = operators.Multiplication(self.discr.dualgrid, self.weights)
        return ft.adjoint * mul * ft

    @util.memoized_property
    def gram_inv(self):
        ft = operators.FourierTransform(self.discr)
        mul = operators.Multiplication(self.discr.dualgrid, 1/self.weights)
        return ft.adjoint * mul * ft


@Sobolev.register(discrs.DirectSum)
def SobolevDirectSum(discr):
    return DirectSum(*(Sobolev(s) for s in discr.summands), flatten=False)
