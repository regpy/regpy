r"""Concrete and abstract Hilbert spaces on vector spaces.
"""

from copy import copy

import numpy as np

from regpy import util, functionals, operators, vecsps
from scipy.sparse import csc_matrix


class HilbertSpace:
    # TODO Make inheritance interface non-public (_gram), provide memoization and checks in public
    #   gram property

    r"""Base class for Hilbert spaces. Subclasses must at least implement the `gram` property, which
    should return a linear `regpy.operators.Operator` instance. To avoid recomputing it,
    `regpy.util.memoized_property` can be used.

    Hilbert spaces can be added, producing `DirectSum` instances on the direct sums of the
    underlying disretizations (see `regpy.vecsps.DirectSum` in the `regpy.vecsps` module).

    They can also be multiplied by scalars to scale the norm. Note that the Gram matrix will scale
    by the square of the factor. This is for consistency with the (not yet implemented) Banach space
    case.

    Parameters
    ----------
    vecsp : regpy.vecsps.VectorSpace
        The underlying vector space. Should be the domain and codomain of the Gram matrix.
    """

    log = util.classlogger

    def __init__(self, vecsp):
        assert isinstance(vecsp, vecsps.VectorSpace)
        self.vecsp = vecsp
        """The underlying vector space."""

    @property
    def gram(self):
        """The gram matrix as an `regpy.operators.Operator` instance."""
        raise NotImplementedError

    @property
    def gram_inv(self):
        r"""The inverse of the gram matrix as an `regpy.operators.Operator` instance. Needs only
        to be implemented if the `gram` property does not return an invertible operator (i.e. one
        that implements `regpy.operators.Operator.inverse`).
        """
        return self.gram.inverse

    def inner(self, x, y):
        r"""Compute the inner product between to elements.

        This is a convenience wrapper around `gram`.

        Parameters
        ----------
        x, y : array-like
            The elements for which the inner product should be computed.

        Returns
        -------
        float
            The inner product.
        """
        return np.real(np.vdot(x, self.gram(y)))

    def norm(self, x):
        r"""Compute the norm of an element.

        This is a convenience wrapper around `norm`.

        Parameters
        ----------
        x : array-like
            The elements for which the norm should be computed.

        Returns
        -------
        float
            The norm.
        """
        return np.sqrt(self.inner(x, x))

    @util.memoized_property
    def norm_functional(self):
        r"""The squared norm functional as a `regpy.functionals.Functional` instance.
        """
        return functionals.HilbertNorm(self)

    def dual_space(self):
        r"""The dual space for the dual pairing given by np.vdot. 
        The dual space coincides with the Hilbert space as `regpy.vecsps.VectorSpace`, but gram is replaced by gram_inv.

        Returns
        ---------
        HilbertSpace
        """
        return GramHilbertSpace(gram = self.gram_inv,gram_inv = self.gram)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.vecsp == other.vecsp
        else:
            return NotImplemented

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


class GramHilbertSpace(HilbertSpace):
    r"""
    Makes the domain of a given (positive, self-adjoint) operator a Hilbert space with the operator as Gram matrix. 
    
    Parameters
    ----------
    gram: operator
        The Gram matrix of the discrete Hilbert space.
    gram_inv: operator, default =None
        Inverse of the Gram matrix    
    """
    def __init__(self, gram, gram_inv=None):
        assert gram.domain == gram.codomain
        if gram_inv is not None:
            assert gram_inv.domain == gram_inv.codomain == gram.domain
        self._gram = gram
        self._gram_inv = gram_inv
        super().__init__(gram.domain)

    @property
    def gram(self):
        return self._gram

    @property
    def gram_inv(self):
        return self._gram_inv or self._gram.inverse


class HilbertPullBack(HilbertSpace):
    r"""Pullback of a Hilbert space on the codomain of an operator to its domain.

    For `op : X -> Y` with Y a Hilbert space, the inner product on X is defined as

        <a, b> := <op(x), op(b)>

    (This really only works in finite dimensions due to completeness). The gram matrix of the
    pullback space is simply `G_X = op.adjoint * G_Y * op`.

    Note that computation of the inverse of `G_X` is not trivial.

    Parameters
    ----------
    space : regpy.hilbert.HilbertSpace
        Hilbert space on the codomain of `op`.
    op : regpy.operators.Operator
        The operator along which to pull back `space`
    inverse : 'conjugate', 'cholesky' or None
        How to compute the inverse gram matrix.

        - 'conjugate': the inverse will be computed as `op.adjoint * G_Y.inverse * op`. **This is
          in general not correct**, but may in some cases be an efficient approximation.
        - 'cholesky': Implement the inverse via Cholesky decomposition. This requires assembling
          the full matrix.
        - None: no inverse will be implemented.
    """

    def __init__(self, space, op, inverse=None):
        assert op.linear
        if not isinstance(space, HilbertSpace) and callable(space):
            space = space(op.codomain)
        assert isinstance(space, HilbertSpace)
        assert op.codomain == space.vecsp
        self.op = op
        """The operator."""
        self.space = space
        """The codomain Hilbert space."""
        super().__init__(op.domain)
        # TODO only compute on demand
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
    r"""The direct sum of an arbirtary number of hilbert spaces, with optional
    scaling of the respective norms. The underlying vector space will be the
    `regpy.vecsps.DirectSum` of the underlying vector spaces of the summands.

    Note that constructing DirectSum instances can be done more comfortably
    simply by adding `regpy.hilbert.HilbertSpace` instances and
    by multiplying them with scalars, but see the documentation for
    `regpy.vecsps.DirectSum` for the `flatten` parameter.

    Parameters
    ----------
    *summands : HilbertSpace tuple
        The Hilbert spaces to be summed. Alternatively, summands can be given
        as tuples `(scalar, HilbertSpace)`, which will scale the norm the
        respective summand. The gram matrices and hence the inner products will
        be scaled by `scalar**2`.
    flatten : bool, optional
        Whether summands that are themselves DirectSums should be merged into
        this instance. Default: False.
    vecsp : vecsps.VectorSpace or callable, optional
        Either the underlying vector space or a factory function that will be
        called with all summands' vector spaces passed as arguments and should
        return a vecsps.DirectSum instance. Default: vecsps.DirectSum.
    """

    def __init__(self, *args, flatten=False, vecsp=None):
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

        if vecsp is None:
            vecsp = vecsps.DirectSum
        if isinstance(vecsp, vecsps.VectorSpace):
            pass
        elif callable(vecsp):
            vecsp = vecsp(*(s.vecsp for s in self.summands))
        else:
            raise TypeError('vecsp={} is neither a VectorSpace nor callable'.format(vecsp))
        assert all(s.vecsp == d for s, d in zip(self.summands, vecsp))

        super().__init__(vecsp)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (
                len(self.summands) == len(other.summands) and
                all(s == t for s, t in zip(self.summands, other.summands)) and
                all(v == w for v, w in zip(self.weights, other.weights))
            )
        else:
            return NotImplemented

    @util.memoized_property
    def gram(self):
        ops = []
        for w, s in zip(self.weights, self.summands):
            if w == 1:
                ops.append(s.gram)
            else:
                ops.append(w**2 * s.gram)
        return operators.DirectSum(*ops, domain=self.vecsp, codomain=self.vecsp)

    def __getitem__(self, item):
        return self.summands[item]

    def __iter__(self):
        return iter(self.summands)

class TensorProd(HilbertSpace):
    r"""The Tensor product of an arbirtary number of hilbert spaces, with optional
    scaling of the respective norms. The underlying vector space will be the
    `regpy.vecsps.Prod` of the underlying discretisations of the factors.

    Important note! The implementation of the Gram operator makes use of the
    BasisTransform Operator from regpy.operators.bases_transform in the sense, that
    the Gram matrix of the Tensor Product of discretised Hilbert spaces
    would be given as the Kronecker-product of all Gram matrices. Which is
    exacly given by the BasisTransform operator given that we interpret the
    Gram matrices as basis changes in each discretised Hilbert space.

    Therefore, please pay attention that to do that we have to actually evaluate
    the Gram-matrix for each Hilbert and store it.

    We want \(H_1 \otimes \dots H_l)\ and each \(H_i)\ is discretised by a basis
    of size \(n_i)\ then we get a memory consumption for the Gram matrices of
    \(O(\sum_{i=1}^l n_i)<=O(l\cdot n))\ with \(n = \max(n_i))\.

    Computing the Gram property itself can be easily seen to have the complexity
    \(O(\sum_{i=1}^l n_i\phi_i(n_i))<=O(l\cdot n\phi(n))))\. with \(\phi_i)\ being
    the complexity for evaluation the Gram operator of the Hilbert space \(H_i)\.
    Note that in the case that each Gram operator is a dense matrix this would be
    given by \(\phi_i(n_i)=n_i^2)\ leading to a complexity of \(O(l\cdot n^3))\.


    Parameters
    ----------
    *args : HilbertSpace or (scalar, HilbertSpace)
        The Hilbert spaces to be tensored. Alternatively, factors can be given
        as tuples `(scalar, HilbertSpace)`, which will scale the norm of the
        respective factor. The gram matrices and hence the inner products will
        be scaled by `scalar**2`.
    flatten : bool, optional
        Whether factors that are themselves TensorProds should be merged into
        this instance. Default: False.
    vecsp : vecsps.VectorSpace or callable, optional
        Either the underlying vector space or a factory function that will be
        called with all factors' vector spaces passed as arguments and should
        return a vecsps.Prod instance. Default: vecsps.Prod.
    """

    def __init__(self, *args, flatten=False, vecsp=None):
        self.factors = []
        self.weights = []
        for arg in args:
            if isinstance(arg, tuple):
                w, s = arg
            else:
                w, s = 1, arg
            assert w > 0
            assert isinstance(s, HilbertSpace)
            if flatten and isinstance(s, type(self)):
                self.factors.extend(s.factors)
                self.weights.extend(w * sw for sw in s.weights)
            else:
                self.factors.append(s)
                self.weights.append(w)

        if vecsp is None:
            vecsp = vecsps.Prod
        if isinstance(vecsp, vecsps.VectorSpace):
            pass
        elif callable(vecsp):
            vecsp = vecsp(*(s.vecsp for s in self.factors))
        else:
            raise TypeError('vecsp={} is neither a VectorSpace nor callable'.format(vecsp))
        assert all(s.vecsp == d for s, d in zip(self.factors, vecsp))

        super().__init__(vecsp)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (
                len(self.factors) == len(other.factors) and
                all(s == t for s, t in zip(self.factors, other.factors)) and
                all(v == w for v, w in zip(self.weights, other.weights))
            )
        else:
            return NotImplemented

    @util.memoized_property
    def gram(self):
        bases = []
        domains = []
        for w, s in zip(self.weights, self.factors):
            basis =[]
            domains.append(s.gram.domain)
            for v in s.gram.domain.real_space().iter_basis():
                if w == 1:
                    basis.append(s.vecsp.real_space().flatten(s.gram(v)))
                else:
                    basis.append(s.vecsp.real_space().flatten((w**2 * s.gram)(v)))
            bases.append(np.array(basis))
        from regpy.operators.bases_transform import BasisTransform
        return BasisTransform(vecsps.Prod(*domains),vecsps.Prod(*domains),bases,dtype=self.vecsp.dtype)

    def __getitem__(self, item):
        return self.factors[item]

    def __iter__(self):
        return iter(self.factors)


class AbstractSpaceBase:
    r"""Class representing abstract hilbert spaces without reference to a concrete implementation.

    The motivation for using this construction is to be able to specify e.g. a Tikhonov penalty
    without requiring knowledge of the concrete vector space the forward operator uses. See the
    documentation of `AbstractSpace` for more details.

    Abstract spaces do not have elements, properties or any other structure, their sole purpose is
    to pick the proper concrete implementation for a given vector space.

    This class only implements operator overloads so that scaling and adding abstract spaces works
    analogously to the concrete `HilbertSpace` instances, returning `AbstractSum` instances. The
    interesing stuff is in `AbstractSpace`.
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


class AbstractSpace(AbstractSpaceBase):
    r"""An abstract Hilbert space that can be called on a vector space to get the corresponding
    concrete implementation.

    AbstractSpaces provide two kinds of functionality:

    - A decorator method `register(vecsp_type)` that can be used to declare some class or function
      as the concrete implementation of this abstract space for vector spaces of type `vecsp_type`
      or subclasses thereof, e.g.:

      .. highlight:: python
      .. code-block:: python
      
            @Sobolev.register(vecsps.UniformGridFcts)
            class SobolevUniformGridFcts(HilbertSpace):
                ...

    - AbstractSpaces are callable. Calling them on a vector space and arbitrary optional
      keyword arguments finds the corresponding concrete `regpy.hilbert.HilbertSpace` among all
      registered implementations. If there are implementations for multiple base classes of the
      vector space type, the most specific one will be chosen. The chosen implementation will
      then be called with the vector space and the keyword arguments, and the result will be
      returned.

      If called without a vector space as positional argument, it returns a new abstract space
      with all passed keyword arguments remembered as defaults. This allows one e.g. to write

          H = Sobolev(index=2)

      after which `H(grid)` is the same as `Sobolev(grid, index=2)` (which in turn will be the
      same as something like `SobolevUniformGridFcts(grid, index=2)`, depending on the type of `grid`).

    Parameters
    ----------
    name : str
        A name for this abstract space. Currently, this is only used in error messages, when no
        implementation was found for some vector space.
    """

    def __init__(self, name):
        self._registry = {}
        self.name = name
        self.args = {}

    def register(self, vecsp_type, impl=None):
        if impl is not None:
            self._registry.setdefault(vecsp_type, []).append(impl)
        else:
            def decorator(i):
                self.register(vecsp_type, i)
                return i
            return decorator

    def __call__(self, vecsp=None, **kwargs):
        if vecsp is None:
            clone = copy(self)
            clone.args = copy(self.args)
            clone.args.update(kwargs)
            return clone
        for cls in type(vecsp).mro():
            try:
                impls = self._registry[cls]
            except KeyError:
                continue
            kws = copy(self.args)
            kws.update(kwargs)
            for impl in impls:
                result = impl(vecsp, **kws)
                if result is NotImplemented:
                    continue
                assert isinstance(result, HilbertSpace)
                return result
        raise NotImplementedError(
            '{} not implemented on {}'.format(self.name, vecsp)
        )


class AbstractSum(AbstractSpaceBase):
    r"""Weighted sum of abstract Hilbert spaces.

    The constructor arguments work like for concrete `regpy.hilbert.HilbertSpace`s, which see.
    Adding and scaling `regpy.hilbert.AbstractSpace` instances is again a more convenient way to
    construct AbstractSums.

    This abstract space can only be called on a `regpy.vecsps.DirectSum`, in which case it
    constructs the corresponding `regpy.hilbert.DirectSum` obtained by matching up summands, e.g.

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

    def __call__(self, vecsp):
        assert isinstance(vecsp, vecsps.DirectSum)
        return DirectSum(
            *((w, s(d)) for w, s, d in zip(self.weights, self.summands, vecsp.summands)),
            vecsp=vecsp
        )

    def __getitem__(self, item):
        return self.weights[item], self.summands[item]

    def __iter__(self):
        return iter(zip(self.weights, self.summands))


def as_hilbert_space(h, vecsp):
    r"""Convert h to HilbertSpace instance on vecsp.

    - If h is an Operator, it's wrapped in a GramHilbertSpace.
    - If h is callable, e.g. an AbstractSpace, it is called on vecsp to
      construct the concrete space.
    """
    if h is None:
        return None
    from regpy.operators import Operator  # imported here to avoid circular dependency
    if not isinstance(h, HilbertSpace):
        if isinstance(h, Operator):
            h = GramHilbertSpace(h)
        elif callable(h):
            h = h(vecsp)
    assert isinstance(h, HilbertSpace)
    assert h.vecsp == vecsp
    return h


L2 = AbstractSpace('L2')
r""":math:`L^2` `AbstractSpace`."""

Sobolev = AbstractSpace('Sobolev')
r"""Sobolev `AbstractSpace`"""

Hm = AbstractSpace('Hm')
r""":math:`H^m` `AbstractSpace`"""

Hm0 = AbstractSpace('Hm0')
r""":math:`H^m_0` `AbstractSpace`"""

L2Boundary = AbstractSpace('L2Boundary')
r""":math:`L^2` `AbstractSpace` on a boundary. Mostly for use with NGSolve."""

SobolevBoundary = AbstractSpace('SobolevBoundary')
r"""Sobolev `AbstractSpace` on a boundary. Mostly for use with NGSolve."""


def componentwise(dispatcher, cls=DirectSum):
    r"""Return a callable that iterates over the components of some vector space, constructing a
    `HilbertSpace` on each component, and joining the result. Intended to be used like e.g.

        L2.register(vecsps.DirectSum, componentwise(L2))

    to register a generic component-wise implementation of `L2` on `regpy.vecsps.DirectSum`
    vector spaces. Any vector space that allows iterating over components using Python's
    iterator protocol can be used, but `regpy.vecsps.DirectSum` is the only example of that right
    now.

    Parameters
    ----------
    dispatcher : callable
        The callable, most likely an `AbstractSpace`, to be applied in each component
        vector space to construct the `HilberSpace` instances.
    cls : callable, optional
        The callable, most likely a `HilbertSpace` subclass, to combine the individual
        `HilbertSpace` instances. Will be called with all spaces as arguments. Default: `DirectSum`.

    Returns
    -------
    callable
        A callable that can be used to register an `AbstractSpace` implementation on
        direct sums.
    """
    def factory(vecsp, **kwargs):
        return cls(*(dispatcher(s, **kwargs) for s in vecsp), vecsp=vecsp)
    return factory


class L2Generic(HilbertSpace):
    r"""`L2` implementation on a generic `regpy.vecsps.VectorSpace`.
    
    Parameters
    ----------
    vecsp : VectorSpace
        Underlying discretization
    weights : array-like
        Weight in the norm.
    """

    def __init__(self, vecsp, weights=None):
        super().__init__(vecsp)
        self.weights = weights

    @util.memoized_property
    def gram(self):
        if self.weights is None:
            return self.vecsp.identity
        else:
            return operators.PtwMultiplication(self.vecsp, self.weights)
        
class L2MeasureSpaceFcts(HilbertSpace):
    r"""`L2` implementation on a `regpy.vecsps.MeasureSpaceFcts`.
    
    Parameters
    ----------
    vecsp : MeasureSpaceFcts
        Underlying discretization
    weights : array-like
        Weight in the norm.
    """

    def __init__(self, vecsp, weights=None):
        assert isinstance(vecsp,vecsps.MeasureSpaceFcts)
        super().__init__(vecsp)
        self.weights = weights

    @util.memoized_property
    def gram(self):
        if self.weights is None:
            if np.all(self.vecsp.measure==1):
                return self.vecsp.identity
            else:
                return operators.PtwMultiplication(self.vecsp,self.vecsp.measure)
        else:
            return operators.PtwMultiplication(self.vecsp, self.weights*self.vecsp.measure)


class L2UniformGridFcts(HilbertSpace):
    r"""`L2` implementation on a `regpy.vecsps.UniformGridFcts`, taking into account the volume
    element.
    """

    def __init__(self, vecsp, weights=None):
        super().__init__(vecsp)
        self.weights = weights

    @util.memoized_property
    def gram(self):
        if self.weights is None:
            return self.vecsp.volume_elem * self.vecsp.identity
        else:
            return self.vecsp.volume_elem * operators.PtwMultiplication(self.vecsp, self.weights)


class SobolevUniformGridFcts(HilbertSpace):
    r"""`Sobolev` implementation on a `regpy.vecsps.UniformGridFcts`.

    Parameters
    ----------
    vecsp : UniformGridFcts
        Grid on which to define the Sobolev space.
    index : float, optional
        Sobolev index, Defaults: 1
    axes : list, optional
        List of axes for which to compute in default all axes, Defaults: None
    """
    def __init__(self, vecsp, index=1, axes=None):
        assert isinstance(vecsp,vecsps.UniformGridFcts)
        super().__init__(vecsp)
        self.index = index
        if axes is None:
            axes = range(vecsp.ndim)
        self.axes = list(axes)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (
                self.vecsp == other.vecsp and
                self.index == other.index
            )
        else:
            return NotImplemented

    @util.memoized_property
    def gram(self):
        ft = operators.FourierTransform(self.vecsp, axes=self.axes)
        mul = operators.PtwMultiplication(
            ft.codomain,
            self.vecsp.volume_elem * (
                1 + np.linalg.norm(ft.codomain.coords[self.axes], axis=0)**2
            )**self.index
        )
        return ft.adjoint * mul * ft

class HmDomain(HilbertSpace):
    r"""Implementation of a Sobolev space :math:`H^m(D)` for a subset :math:`D` of a `UniformGridFcts` grid.
    :math:`D` is characterized by a binary or integer-valued mask: `D={mask==1}`.
    `{mask==0}` are Dirichlet boundaries, and `{mask==-1}` Neumann boundaries.
    `mask` may also be boolean, in this case there are only Dirichlet boundaries.
    Boundary condition at the exterior boundaries are specified by `ext_bd_cond`, default is Neumann ('Neum')

    `m=index` is a non-negative integer, the order or index of the Sobolev space.
    The gram matrix is given by :math:`(\alpha I - \Delta)^{-m}`.

    By default it is assumed that the lengths in grid are given in physical dimensions,
    and a non-dimensionalization is carried out such that the largest side length (extent) of grid is 1.

    If `weight` is specified, the Gram matrix will approximate :math:`(\alpha I-{weight}\Delta)^{-m}`. `weight` should be slowly varying.

    Parameters
    ----------
    grid : UniformGridFcts
        Underlying grid functions.
    mask : array-type
        Mask to capture that subset :math:`D` on which the Sobolev space is defined. Can only contain 
        values `{-1,0,1}` or is a boolean. Shape has to match the shape of `grid`.
    h : tuple or None or string, optional
        The extent of the domain either given as a tuple or computed. Option key strings "physical" or 
        "normalized". (Defaults: "normalized)
    index : int, optional
        The Sobolev index :math:`m`. (Defaults: 1)
    weight : array-type, optional
        Weights to be applied to Laplacian in the gram matrix definition. (Defaults: None)
    ext_bd_cond : any, optional
        Exterior boundary conditions to be applied. If not "Neum" takes Dirichlet boundary conditions. (Defaults: "Neum")
    alpha : scalar, optional
        Parameter when computing the gram matrix as :math:`(\alpha I - \Delta)^{-m}`. (Defaults: 1)
    dtype : type, optional
        Type of underlying grid. (Defaults: float) 
    """

    def __init__(self,
                grid, 
                mask,
                h='normalized',
                index=1,
                weight=None,
                ext_bd_cond = 'Neum',
                alpha = 1,
                dtype = float):
        assert grid is None or isinstance(grid,vecsps.UniformGridFcts)
        if not (grid is None or mask is None):
            assert grid.shape == mask.shape
        assert type(index)== int and index>=0
        self.ndim = mask.ndim
        """Dimension of vector space
        """
        if type(h) == tuple:
            self.h_val = h
        elif grid is None:
            self.h_val=1./np.max(mask.shape)*np.ones((self.ndim,))
        elif h=='physical':
            self.h_val = grid.extents/(np.array(grid.shape)-1)
        elif h=='normalized':
            self.h_val = (2.*np.pi/np.max(grid.extents))* (grid.extents/(np.array(grid.shape)-1))
        else:
            raise NotImplemented

        self.index = index
        """Sobolev index.
        """
        self.alpha = alpha
        """Regularizer for Gram matrix.
        """
        self.grid = grid
        """Underlying gird.
        """
        self.mask = (mask==1)
        """Mask to determine the subspace D.
        """
        self.dtype = grid.dtype if grid else dtype
        """Type of the underlying grid. 
        """
        # impose exterior Neumann boundary conditions
        mask = np.pad(mask.astype(int),1,'constant',constant_values= -1 if ext_bd_cond=='Neum' else 0)
        vecsp = vecsps.VectorSpace((np.count_nonzero(mask==1),),dtype= self.dtype)
        super().__init__(vecsp)
        self.G = np.zeros(mask.shape,dtype=int)
        interior_ind = mask==1
        self.G[interior_ind] = 1+np.arange(np.count_nonzero(interior_ind))
        self.G[mask==-1] = -1

        if weight is None:
            self.weight = None
        else:
            self.weight = np.pad(weight,1,'edge')

    def I_minus_Delta(self):
        r"""
        I_minus_Delta is the sparse form of the sum of the `alpha*identity` and the negative Laplacian on the domain D 
        defined by masking with `mask`.
        """
        if not self.weight is None:
            w = self.weight.ravel()
        # Indices of interior points
        G1 = self.G.ravel()
        p = np.where(G1>0)[0] # list of numbers of interior points in flattened array
        N = len(p)
        # Connect interior points to themselves with 4's.
        i = []   # row indices of matrix entries
        j = []   # column indices of matrix entries
        s = []   # values of matrix entries
        dia = self.alpha * np.ones((len(p),))   # values of diagonal matrix entries; ones correspond to identity matrix
        # for k = north, east, south, west

        kval= [1]
        for d in range(self.ndim-1,0,-1):
            kval = np.concatenate([kval,[kval[-1]*self.G.shape[d]] ])
        # If G.shape = [m,n], then kval= [1,n].
        # If G.shape = [l,m,n], then kval = [1,n,m*n]

        for dir,h in enumerate(self.h_val):
            for k in  kval[dir]*np.array([-1,1]):
                # Possible neighbors in k-th direction
                Q=np.zeros_like(p)
                Q = G1[p+k]
                # Indices of points with interior neighbors
                q = np.where(Q>0)[0]
                # Connect interior points to neighbors
                i = np.concatenate([i, G1[p[q]]-1])
                j = np.concatenate([j,Q[q]-1])
                entries = np.ones(q.shape)/h**2
                if not self.weight is None:
                    entries = entries * np.sqrt(w[p[q]]*w[p[q]+k])
                s = np.concatenate([s,-entries ])
                dia[G1[p[q]]-1] += entries
                # Indices of points with neighbors on Dirichlet boundary
                q_diri = np.where(Q==0)[0]
                entries = np.ones(q_diri.shape)/h**2
                if not self.weight is None:
                    entries = entries * np.sqrt(w[p[q_diri]]*w[p[q_diri]+k])
                dia[G1[p[q_diri]]-1] += entries
        i = np.concatenate([i, G1[p]-1])
        j = np.concatenate([j, G1[p]-1])
        s = np.concatenate([s,dia])
        return csc_matrix((s, (i,j)),(N,N))

    @util.memoized_property
    def gram(self):
        return operators.Pow(
            operators.MatrixMultiplication(self.I_minus_Delta(),inverse='superLU',dtype = self.dtype),
            self.index
            )


def _register_spaces():
    r"""Auxiliary method to register abstract spaces for various vector spaces. Using the decorator
    method described in `AbstractSpace` does not work due to circular depenencies when
    loading modules.

    This is called from the `regpy` top-level module once, and can be ignored otherwise.
    """

    L2.register(vecsps.Prod, componentwise(L2,cls=TensorProd))
    L2.register(vecsps.DirectSum, componentwise(L2))
    L2.register(vecsps.VectorSpace, L2Generic)
    L2.register(vecsps.MeasureSpaceFcts,L2MeasureSpaceFcts)
    L2.register(vecsps.UniformGridFcts, L2UniformGridFcts)

    Sobolev.register(vecsps.DirectSum, componentwise(Sobolev))
    Sobolev.register(vecsps.UniformGridFcts, SobolevUniformGridFcts)

    Hm.register(vecsps.Prod, componentwise(HmDomain,cls=TensorProd))
    Hm.register(vecsps.DirectSum, componentwise(HmDomain))
    Hm.register(vecsps.UniformGridFcts,HmDomain)

    L2Boundary.register(vecsps.DirectSum, componentwise(L2Boundary))

    SobolevBoundary.register(vecsps.DirectSum, componentwise(SobolevBoundary))
