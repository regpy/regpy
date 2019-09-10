from copy import copy
import numpy as np

from . import discrs
from .. import util, operators

from ngsolve import *


class HilbertSpace:
    log = util.classlogger

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
    """Pullback of a hilbert space on the codomain of an operator to its
    domain.

    For `op : X -> Y` with Y a Hilbert space, the inner product on X is defined
    as

        <a, b> := <op(x), op(b)>

    (This really only works in finite dimensions due to completeness). The gram
    matrix of the pullback space is simply `G_X = op^* G_Y op`.

    Note that computation of the inverse of `G_Y` is not trivial.

    Parameters
    ----------
    space : `~itreg.spaces.hilbert.HilbertSpace`
        Hilbert space on the codomain of `op`.
    op : `~itreg.operators.Operator`
        The operator along which to pull back `space`
    inverse : one of None, 'conjugate' or 'cholesky'
        How to compute the inverse gram matrix.
        - None: no inverse will be implemented.
        - 'conjugate': the inverse will be computed as `op^* G_Y^{-1} op`.
          **This is in general not correct**, but may in some cases be an
          efficient approximation.
        - 'cholesky': Implement the inverse via Cholesky decomposition. This
          requires assembling the full matrix.
    """

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
    """The direct sum of an arbirtary number of hilbert spaces, with optional
    scaling of the respective norms. The underlying discretization will be the
    :class:`itreg.spaces.discrs.DirectSum` of the underlying discretizations of the
    summands.

    Note that constructing DirectSum instances can be done more comfortably
    simply by adding :class:`~itreg.spaces.hilbert.HilbertSpace` instances and
    by multiplying them with scalars, but see the documentation for
    :class:`itreg.spaces.discrs.DirectSum` for the `flatten` parameter.

    Parameters
    ----------
    *summands : variable number of :class:`~itreg.spaces.hilbert
        The Hilbert spaces to be summed. Alternatively, summands can be given
        as tuples `(scalar, HilbertSpace)`, which will scale the norm the
        respective summand. The gram matrices and hence the inner products will
        be scaled by `scalar**2`.
    flatten : bool, optional
        Whether summands that are themselves DirectSums should be merged into
        this instance.
    """

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
        return operators.DirectSum(*ops)

    @util.memoized_property
    def gram_inv(self):
        ops = []
        for w, s in zip(self.weights, self.summands):
            if w == 1:
                ops.append(s.gram_inv)
            else:
                ops.append(1/w**2 * s.gram_inv)
        return operators.DirectSum(*ops)


class AbstractSpace:
    """Class representing abstract hilbert spaces without reference to a
    concrete implementation.

    The motivation for using this construction is to be able to specify e.g. a
    Thikhonov penalty without requiring knowledge of the concrete discretization
    the forward operator uses. See the documentation of
    :class:`~itreg.spaces.hilbert.AbstractSpaceDispatcher` for more details.

    Abstract spaces do not have elements, their sole purpose is to pick the
    proper concrete implementation for a given discretization.

    This class only implements operator overloads so that scaling and adding
    abstract spaces works analogously to the concrete
    :class:`~itreg.spaces.hilbert.HilbertSpace` instances, returning
    :class:`~itreg.spaces.hilbert.AbstractSum` instances.
    """

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
    """An abstract Hilbert space that can be called on a discretization to get
    the corresponding concrete implementation.

    AbstractSpaceDispatchers provide two kinds of functionality:

    - A decorator method `register(discr_type)` that can be used to declare
      some class or function as the concrete implementation of this abstract
      space for discretizations of type `discr_type` or subclasses thereof,
      e.g.:

          @Sobolev.register(discrs.UniformGrid)
          class SobolevUniformGrid(HilbertSpace):
              ...

    - AbstractSpaces are callable. Calling them on a discretization and
      arbitrary optional keyword arguments finds the corresponding concrete
      :class:`~itreg.spaces.hilbert.HilbertSpace` among all registered
      implementations. If there are implementations for multiple base classes
      of the discretization type, the most specific one will be chosen. The
      chosen implementation will then be called with the discretization and the
      keyword arguments, and the result will be returned.

      If called without a discretization as positional argument, it returns a
      new abstract space with all passed keyword arguments remembered as
      defaults. This allows one e.g. to write

          H = Sobolev(index=2)

      after which `H(grid)` is the same as `Sobolev(grid, index=2)` (which in
      turn will be the same as something like `SobolevUniformGrid(grid, index=2)`,
      depending on the type of `grid`).

    Parameters
    ----------
    name : str
        A name for this abstract space. Currently, this is only used in error
        messages, when no implementation was found for some discretization.
    """

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
    """Weighted sum of abstract Hilbert spaces.

    The constructor arguments work like for concrete
    :class:`~itreg.spaces.hilbert.HilbertSpace`s, which see. Adding and scaling
    :class:`~itreg.spaces.hilbert.AbstractSpace` instances is again a more
    convenient way to construct AbstractSums.

    This abstract space can only be called on a
    :class:`itreg.spaces.discrs.DirectSum`, in which case it constructs the
    corresponding :class:`itreg.spaces.hilbert.DirectSum` obtained by matching
    up summands, e.g.

        (L2 + 2 * Sobolev(index=1))(grid1 + grid2) == L2(grid1) + 2 * Sobolev(grid2, index=1)
    """

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
    def __init__(self, discr, index=1):
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


@L2.register(discrs.NGSolveDiscretization)
class NGSolveFESSpace_L2(HilbertSpace):
    def __init__(self, discr):
        self.discr = discr

        u, v=self.discr.fes.TnT()
        self.discr.a+=SymbolicBFI(u*v)
        self.discr.a.Assemble()

        self.discr.b=self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())

    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse


@Sobolev.register(discrs.NGSolveDiscretization)
class NGSolveFESSpace_H1(HilbertSpace):
    def __init__(self, discr):
        self.discr = discr

        u, v=self.discr.fes.TnT()
        self.discr.a+=SymbolicBFI(u*v+grad(u)*grad(v))
        self.discr.a.Assemble()

        self.discr.b=self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())


    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse

# The boundary spaces are only defined for a circular boundary
# Need to define the definedon keyword argument outside of spaces
SobolevBoundary = AbstractSpaceDispatcher('SobolevBoundary')


@SobolevBoundary.register(discrs.NGSolveDiscretization)
class NGSolveFESSpace_H1_bdr(HilbertSpace):
    def __init__(self, discr):
        self.discr = discr

        u, v=self.discr.fes.TnT()
        self.discr.a+=SymbolicBFI(u.Trace()*v.Trace()+u.Trace().Deriv()*v.Trace().Deriv(), definedon=self.discr.fes.mesh.Boundaries("cyc"))
        self.discr.a.Assemble()

        self.discr.b=self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())

    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse


L2Boundary = AbstractSpaceDispatcher('L2Boundary')


@L2Boundary.register(discrs.NGSolveDiscretization)
class NGSolveFESSpace_L2_bdr(HilbertSpace):
    def __init__(self, discr):
        self.discr = discr

        u, v=self.discr.fes.TnT()
        self.discr.a+=SymbolicBFI(u.Trace()*v.Trace(), definedon=self.discr.fes.mesh.Boundaries("cyc"))
        self.discr.a.Assemble()

        self.discr.b=self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())

    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse
