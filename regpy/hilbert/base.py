from copy import copy,deepcopy
from math import sqrt

import numpy as np

from regpy import vecsps
from regpy.util import Errors, memoized_property, ClassLogger
from regpy.operators import CholeskyInverse,PtwMultiplication, Operator
from regpy.operators import DirectSum as DirectSumOp
from regpy.operators.bases_transform import BasisTransform


__all__ = ["HilbertSpace","HilbertPullBack","GramHilbertSpace","DirectSum","TensorProd","L2Generic","AbstractSpace"]

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
    vecsp : regpy.vecsps.VectorSpaceBase
        The underlying vector space. Should be the domain and codomain of the Gram matrix.
    """

    log = ClassLogger()

    def __init__(self, vecsp):
        if not isinstance(vecsp, vecsps.VectorSpaceBase):
            raise TypeError(Errors.not_instance(vecsp,vecsps.VectorSpaceBase,add_info="Hilbert spaces are only defined on vector spaces defined on subsidies of the RegPy VectorSpaceBase"))
        self.vecsp = vecsp
        """The underlying vector space."""
        self._no_pickle = {}

    def __deepcopy__(self, memo):
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in self._no_pickle:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

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
        return (self.vecsp.vdot(x, self.gram(y))).real

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
        return sqrt(self.inner(x, x))

    @memoized_property
    def norm_functional(self):
        r"""The squared norm functional as a `regpy.functionals.Functional` instance.
        """
        from regpy.functionals import HilbertNorm
        return HilbertNorm(self)

    def dual_space(self):
        r"""The dual space for the dual pairing given by `domain.vdot`. 
        The dual space coincides with the Hilbert space as `regpy.vecsps.VectorSpaceBase`, but gram is replaced by gram_inv.

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
        if isinstance(other,float) or isinstance(other,int):
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
        if not isinstance(gram,Operator):
            raise TypeError(Errors.not_instance(gram,Operator,"To define a GramHilbertSpace the gram operator has to be a proper RegPy operator."))
        if gram.domain != gram.codomain:
            raise ValueError(Errors.value_error("The domain and codomain of the gram operator for the GramHilbertSpace has to be identical. Was given:"+"\n\t "+f"gram = {gram}"))
        if gram_inv is not None:
            if not isinstance(gram_inv,Operator):
                raise TypeError(Errors.not_instance(gram_inv,Operator,"To define a GramHilbertSpace the with inverse, the gram_inv operator has to be a proper RegPy operator."))
            if gram_inv.domain != gram_inv.codomain or gram_inv.domain != gram.domain:
                raise ValueError(Errors.value_error("The domain and codomain of the gram_inv operator for the GramHilbertSpace has to be identical and match with the domain of the gram operator. Was given:"+"\n\t "+f"gram = {gram}"+"\n\t "+f"gram_inv={gram_inv}"))
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
        if not isinstance(op,Operator):
            raise TypeError(Errors.not_instance(op,Operator,"To define a HilbertPullBack the operator has to be a proper RegPy operator."))
        if not op.linear:
            raise ValueError(Errors.not_linear_op(op,add_info="To define a HilbertPullBack space the operator has to be linear!"))
        if not isinstance(space, HilbertSpace) and callable(space):
            space = space(op.codomain)
        if not isinstance(space, HilbertSpace):
            raise TypeError(Errors.not_instance(space,HilbertSpace,add_info="The space for a HilbertSpacePullBack has to be a method returning a Hilbert space or a HilbertSpace"))
        if op.codomain != space.vecsp:
            raise ValueError(Errors.not_equal(op.codomain,space.vecsp,add_info="The vector space codomain of the operator and vector space of the Hilbert space to be pulled back need to match."))
        self.op = op
        """The operator."""
        self.space = space
        """The codomain Hilbert space."""
        super().__init__(op.domain)
        if not inverse:
            self.inverse = None
        elif inverse == 'conjugate':
            self.log.info(
                'Note: Using T* G^{-1} T as inverse of T* G T. This is probably not correct.')
            self.inverse = op.adjoint * space.gram_inv * op
        elif inverse == 'cholesky':
            self.inverse = CholeskyInverse(self.gram)

    @memoized_property
    def gram(self):
        return self.op.adjoint * self.space.gram * self.op

    @property
    def gram_inv(self):
        if self.inverse:
            return self.inverse
        raise NotImplementedError


class DirectSum(HilbertSpace):
    r"""The direct sum of an arbitrary number of hilbert spaces, with optional
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
    vecsp : vecsps.VectorSpaceBase or callable, optional
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
                if not np.isscalar(w): raise TypeError(Errors.type_error("Weights in the DirectSum Hilbert Space need to be scalars!"))
                if w <= 0:
                    raise ValueError(Errors.value_error("Weights in the DirectSum Hilbert space need to be positive!"))
            else:
                w, s = 1, arg
            if not isinstance(s, HilbertSpace):
                raise TypeError(Errors.not_instance(s,HilbertSpace,add_info="The spaces in the direct sum hilbert space need to be proper HilbertSpaces!"))
            if flatten and isinstance(s, type(self)):
                self.summands.extend(s.summands)
                self.weights.extend(w * sw for sw in s.weights)
            else:
                self.summands.append(s)
                self.weights.append(w)
        if isinstance(vecsp,vecsps.DirectSum):
            pass
        elif vecsp is None:
            vecsp = vecsps.DirectSum(*[h_space.vecsp for h_space in self.summands])
        elif callable(vecsp):
            vecsp = vecsp(*(s.vecsp for s in self.summands))
        else:
            raise TypeError(Errors.type_error('vecsp={} is neither a VectorSpaceBase nor callable'.format(vecsp)))
        if not isinstance(vecsp, vecsps.DirectSum):
            raise TypeError(Errors.not_instance(vecsp,vecsps.DirectSum,"The vector space for a DirectSum Hilbert space need to be a DirectSum!"))
        if len(self.summands) != len(vecsp.summands):
            raise ValueError(Errors.value_error("The number of summands of in the vector space does not match the number of Hilbert spaces to be summed!",self))
        if any(s.vecsp != d for s, d in zip(self.summands, vecsp)):
            raise ValueError(Errors.value_error("The vector spaces in the direct sum vector space does not match the vector spaces in the Hilbert spaces to be summed!"))
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

    @memoized_property
    def gram(self):
        ops = []
        for w, s in zip(self.weights, self.summands):
            if w == 1:
                ops.append(s.gram)
            else:
                ops.append(w**2 * s.gram)
        return DirectSumOp(*ops, domain=self.vecsp, codomain=self.vecsp)

    def __getitem__(self, item):
        return self.summands[item]

    def __iter__(self):
        return iter(self.summands)

class TensorProd(HilbertSpace):
    r"""The Tensor product of an arbitrary number of hilbert spaces, with optional
    scaling of the respective norms. The underlying vector space will be the
    `regpy.vecsps.Prod` of the underlying discretizations of the factors.

    Important note! The implementation of the Gram operator makes use of the
    BasisTransform Operator from regpy.operators.bases_transform in the sense, that
    the Gram matrix of the Tensor Product of discretised Hilbert spaces
    would be given as the Kronecker-product of all Gram matrices. Which is
    exactly given by the BasisTransform operator given that we interpret the
    Gram matrices as basis changes in each discretised Hilbert space.

    Therefore, please pay attention that to do that we have to actually evaluate
    the Gram-matrix for each Hilbert and store it.

    We want :math:`H_1 \otimes \dots H_l` and each :math:`H_i` is discretised by a basis
    of size :math:`n_i` then we get a memory consumption for the Gram matrices of
    :math:`O(\sum_{i=1}^l n_i)<=O(l\cdot n)` with :math:`n = \max(n_i)`.

    Computing the Gram property itself can be easily seen to have the complexity
    :math:`O(\sum_{i=1}^l n_i\phi_i(n_i))<=O(l\cdot n\phi(n)))`. with :math:`\phi_i` being
    the complexity for evaluation the Gram operator of the Hilbert space :math:`H_i`.
    Note that in the case that each Gram operator is a dense matrix this would be
    given by :math:`\phi_i(n_i)=n_i^2` leading to a complexity of :math:`O(l\cdot n^3)`.


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
    vecsp : vecsps.VectorSpaceBase or callable, optional
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
                if not np.isscalar(w): raise TypeError(Errors.type_error("Weights in the TensorProd Hilbert Space need to be scalars!"))
                if w <= 0:
                    raise ValueError(Errors.value_error("Weights in the TensorProd Hilbert space need to be positive!"))
            else:
                w, s = 1, arg
            if not isinstance(s, HilbertSpace):
                raise TypeError(Errors.not_instance(s,HilbertSpace,add_info="The spaces in the TensorProd Hilbert space need to be proper HilbertSpaces!"))
            if flatten and isinstance(s, type(self)):
                self.factors.extend(s.factors)
                self.weights.extend(w * sw for sw in s.weights)
            else:
                self.factors.append(s)
                self.weights.append(w)

        if vecsp is None:
            vecsp = vecsps.Prod
        if isinstance(vecsp, vecsps.VectorSpaceBase):
            pass
        elif callable(vecsp):
            vecsp = vecsp(*(s.vecsp for s in self.factors))
        else:
            raise TypeError(Errors.type_error('vecsp={} is neither a VectorSpaceBase nor callable'.format(vecsp)))
        if not isinstance(vecsp, vecsps.Prod):
            raise TypeError(Errors.not_instance(vecsp,vecsps.Prod,"The vector space for a TensorProd Hilbert space need to be a Prod!"))
        if len(self.factors) != len(vecsp.factors):
            raise ValueError(Errors.value_error("The number of factors of in the vector space does not match the number of Hilbert spaces to be tensored!",self))
        if any(s.vecsp != d for s, d in zip(self.factors, vecsp)):
            raise ValueError(Errors.value_error("The vector spaces in the product vector space does not match the vector spaces in the Hilbert spaces to be tensored!"))
        

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

    @memoized_property
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
        return BasisTransform(vecsps.Prod(*domains),vecsps.Prod(*domains),bases,dtype=self.vecsp.dtype)

    def __getitem__(self, item):
        return self.factors[item]

    def __iter__(self):
        return iter(self.factors)


class L2Generic(HilbertSpace):
    r"""`L2` implementation on a generic `regpy.vecsps.VectorSpaceBase`.
    
    Parameters
    ----------
    vecsp : regpy.vecsps.VectorSpaceBase
        Underlying discretization
    weights : array-like
        Weight in the norm.
    """

    def __init__(self, vecsp, weights=None):
        super().__init__(vecsp)
        self.weights = weights

    @memoized_property
    def gram(self):
        if self.weights is None:
            return self.vecsp.identity
        else:
            return PtwMultiplication(self.vecsp, self.weights)


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

    log = ClassLogger()

    def __add__(self, other):
        if callable(other):
            return AbstractSum(self, other, flatten=True)
        else:
            return NotImplemented
    
    def __iadd__(self,other):
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
        if isinstance(other,float):
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
            self.__doc__ += "-"*125 + "\n" + f"--- Implementation for {vecsp_type.__name__} is given by {impl.__name__} with the following documentation ---" +"\n"+ f"{impl.__doc__}" + "\n" + "-"*125
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
                if not isinstance(result, HilbertSpace):
                    raise RuntimeError(Errors.not_instance(result,HilbertSpace,add_info=f"The Abstract Hilbert space {self} did not construct a proper Hilbert space on {vecsp}. The result was:"+"\n\t"+f"result = {result}."))
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
                if w <= 0:
                    raise ValueError(Errors.value_error("Weights in the AbstractDirectSum Hilbert space need to be positive!"))
            else:
                w, s = 1, arg
            if not callable(s):
                raise TypeError(Errors.not_instance(s,callable,add_info="The spaces in the AbstractDirectSum hilbert space need to be proper HilbertSpaces!"))
            if flatten and isinstance(s, type(self)):
                self.summands.extend(s.summands)
                self.weights.extend(w * sw for sw in s.weights)
            else:
                self.summands.append(s)
                self.weights.append(w)

    def __call__(self, vecsp):
        if not isinstance(vecsp, vecsps.DirectSum):
            raise TypeError(Errors.not_instance(vecsp,vecsps.DirectSum,f"The Abstract direct sum is only callable on a DirectSum VectorSpace. Not a {vecsp}"))
        return DirectSum(
            *((w, s(d)) for w, s, d in zip(self.weights, self.summands, vecsp.summands)),
            vecsp=vecsp
        )

    def __getitem__(self, item):
        return self.weights[item], self.summands[item]

    def __iter__(self):
        return iter(zip(self.weights, self.summands))


