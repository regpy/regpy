from __future__ import annotations

from copy import copy,deepcopy
from math import sqrt
from dataclasses import dataclass
from types import NotImplementedType
from typing import Any, Callable, Iterator, Protocol, TYPE_CHECKING

from typing_extensions import Self


import numpy as np

if TYPE_CHECKING:
    from regpy.operators import Operator

from regpy.util import ClassLogger,Errors,make_repr,memoized_property,get_rng, set_rng_seed

__all__ = ["Vector", "TupleVector", "VectorSpaceBase", "DirectSum"]


Scalar = int | float | complex | np.number


class Vector(Protocol):
    r"""Structural type for vectors accepted by generic RegPy code.

    A vector supports only the linear operations documented by
    :class:`VectorSpaceBase`.  Allocation, copying, scalar products and other
    representation-dependent operations belong to the vector space instead.

    Vectors are duck-typed. This class is only used for typing and for clarity,
    but does not influence the behavior of the code. In particular, vector classes used 
    in RegPy do not need to inherit from it. 
    """

    def __add__(self, other: Self, /) -> Self: ...

    def __radd__(self, other: Self, /) -> Self: ...

    def __iadd__(self, other: Self, /) -> Self: ...

    def __sub__(self, other: Self, /) -> Self: ...

    def __rsub__(self, other: Self, /) -> Self: ...

    def __isub__(self, other: Self, /) -> Self: ...

    def __mul__(self, scalar: Scalar, /) -> Self: ...

    def __rmul__(self, scalar: Scalar, /) -> Self: ...

    def __imul__(self, scalar: Scalar, /) -> Self: ...

    def __truediv__(self, scalar: Scalar, /) -> Self: ...

    def __itruediv__(self, scalar: Scalar, /) -> Self: ...

    def __neg__(self) -> Self: ...

@dataclass 
class TupleVector:
    v: list[Vector]

    __array_ufunc__ = None

    log = ClassLogger()

    def copy(self) -> Self:
        return copy(self)

    def __post_init__(self) -> None:
        self.v = list(self.v)
        self.types = [type(v_i) for v_i in self.v]
        self.ndim = len(self.types)

    def __len__(self) -> int:
        return self.ndim
    
    def conj(self) -> Self:
        return TupleVector([v_i.conj() for v_i in self])

    @property
    def real(self) -> Self:
        return TupleVector([v_i.real for v_i in self])
    
    @property
    def imag(self) -> Self:
        return TupleVector([v_i.imag for v_i in self])

    def __eq__(self, other: object) -> TupleVector:
        if isinstance(other,TupleVector) and self.ndim == other.ndim:
            return TupleVector([s_i == o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i == other for s_i in self.v])
    
    def __lt__(self, other: object) -> TupleVector:
        if isinstance(other,TupleVector) and self.ndim == other.ndim:
            return TupleVector([s_i < o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i < other for s_i in self.v])

    def __le__(self, other: object) -> TupleVector:
        if isinstance(other,TupleVector) and self.ndim == other.ndim:
            return TupleVector([s_i <= o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i <= other for s_i in self.v])

    def __ne__(self, other: object) -> bool:
        return not (self == other)
    
    def __ge__(self, other: object) -> TupleVector:
        if isinstance(other,TupleVector) and self.ndim == other.ndim:
            return TupleVector([s_i >= o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i >= other for s_i in self.v])
    
    def __gt__(self, other: object) -> TupleVector:
        if isinstance(other,TupleVector) and self.ndim == other.ndim:
            return TupleVector([s_i > o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i > other for s_i in self.v])

    def all(self) -> bool:
        return all((v_i.all() for v_i in self.v))

    def any(self) -> bool:
        return any((v_i.any() for v_i in self.v))
    
    def __or__(self, other: TupleVector) -> TupleVector:
        if not isinstance(other,TupleVector) or self.ndim != other.ndim:
            raise ValueError(Errors.value_error("Comparing with TupleVector only supported for TupleVectors of identical dimension!"))
        return TupleVector([s_i | o_i for s_i,o_i in zip(self.v,other.v)])
    
    def sum(self, **kwargs: Any) -> Scalar:
        z = []
        for v_i in self.v:
            if isinstance(v_i,np.ndarray):
                s = np.sum(v_i,**kwargs)
            else:
                try:
                    s = sum(v_i)
                except TypeError:
                    s = v_i 
            if isinstance(s,np.number):
                z.append(s.item())
            else:
                z.append(s)
        return sum(z)
    
    def __iadd__(self, other: TupleVector) -> Self:
        if not isinstance(other,TupleVector) or other.ndim != self.ndim or any([t_o!=t_s for t_o,t_s in zip(other.types,self.types)]):
            raise ValueError(Errors.value_error("Adding TupleVector only supported for TupleVectors of identical dimension and identical type in each component!"))
        for k,v_o in enumerate(other.v):
            self.v[k] += v_o
        return self

    def __isub__(self, other: TupleVector) -> Self:
        if not isinstance(other,TupleVector) or other.ndim != self.ndim or any([t_o!=t_s for t_o,t_s in zip(other.types,self.types)]):
            raise ValueError(Errors.value_error("Subtracting TupleVector only supported for TupleVectors of identical dimension and identical type in each component!"))
        for k,v_o in enumerate(other.v):
            self.v[k] -= v_o
        return self
    
    def __neg__(self) -> Self:
        return TupleVector([-v_i for v_i in self])
    
    def __add__(self, other: TupleVector) -> Self:
        if not isinstance(other,TupleVector) or other.ndim != self.ndim or any([t_o!=t_s for t_o,t_s in zip(other.types,self.types)]):
            raise ValueError(Errors.value_error("Adding TupleVector only supported for TupleVectors of identical dimension and identical type in each component!"))
        return TupleVector([s_k + o_k for s_k,o_k in zip(self,other)])
    
    def __radd__(self, other: TupleVector) -> Self:
        return self + other
    
    def __sub__(self, other: TupleVector) -> Self:
        return self + (-1*other)
    
    def __rsub__(self, other: TupleVector) -> Self:
        return (-1*self) + other
    
    def __imul__(self, other: Scalar) -> Self:
        if not isinstance(other,(float,int,complex)):
            raise ValueError(Errors.value_error("Multiplying TupleVector only supported for scalars (int, float, or complex)!"))
        for k in range(self.ndim):
            self.v[k] *= other
        return self
    
    def __itruediv__(self, other: Scalar) -> Self:
        if not isinstance(other,(float,int,complex)):
            raise ValueError(Errors.value_error("Division TupleVector only supported for scalars (int, float, or complex)!"))
        for k in range(self.ndim):
            self.v[k] /= other
            return self
    
    def __mul__(self, other: Scalar | Operator) -> Self | Operator:
        from regpy.operators.base import Operator,PtwMultiplication
        if isinstance(other,float) or isinstance(other,int) or isinstance(other,complex):
            return TupleVector([other * s_k for s_k in self])
        elif isinstance(other,Operator):
            return PtwMultiplication(other.codomain, self) * other
        else:
            raise NotImplementedError(Errors.generic_message(f"Multiplication of TupleVector with {type(other)} is not defined. It has to be either a number eg float, int or complex or an Operator."))
        
    def __rmul__(self, other: Scalar | Operator) -> Self | Operator:
        return self * other

    def __truediv__(self, other: Scalar) -> Self:
        if not isinstance(other,(float,int,complex)):
            raise ValueError(Errors.value_error("Division TupleVector only supported for scalars (int, float, or complex)!"))
        return TupleVector([s_k / other for s_k in self])

    def __iter__(self) -> Iterator[Vector]:
        return iter(self.v)
    
    def __getitem__(self, key: Any) -> Any:
        if isinstance(key,slice) or isinstance(key,int):
            return self.v[key]
        elif (isinstance(key,TupleVector) and self.ndim == key.ndim):
            return TupleVector([v_i[k_i] for v_i,k_i in zip(self,key)])
        elif isinstance(key,(list,tuple)) and len(key) == self.ndim: 
            return TupleVector([v_i[k_i] for v_i,k_i in zip(self,key)])
        else:
            raise KeyError(Errors.indexation(key,self,add_info=f"keys of type {type(key)} are not supported either int or list of length {self.ndim}"))
    
    def __setitem__(self, key: Any, item: Any) -> None:
        if isinstance(key,slice) or isinstance(key,int):
            self.v[key] = item
        elif isinstance(key,(list,tuple)) and len(key) <= self.ndim:
            if (isinstance(item,(list,tuple)) and len(item) == len(key)):
                for k_i,item_i in zip(key,item):
                    self.v[k_i] = item_i
            elif isinstance(item,TupleVector) and item.ndim == len(key):
                for k_i,item_i in zip(key,item.v):
                    self.v[k_i] = item_i
            else:
                raise TypeError(Errors.type_error(f"If keys are lists or tuples of length smaller than the length of the Vector  the items can be list or TupleVectors of length same length. Got key list of length {len(key)} and item of length {len(item)}"))
        elif (isinstance(key,TupleVector) and self.ndim == key.ndim): 
            if (isinstance(item,TupleVector) and item.ndim == key.ndim):
                for v_i,k_i,item_i in zip(self.v,key,item):
                    v_i[k_i] = item_i
            elif np.isscalar(item):
                for v_i,k_i in zip(self.v,key):
                    v_i[k_i] = item
            else:
                raise TypeError(Errors.type_error("items has to be a TupleVector if key is TupleVector of ndim {} not {} type".format(self.ndim,type(item))))         
        else:
            raise KeyError(Errors.indexation(key,self,add_info=f"keys of type {type(key)} are not supported either int or list of length {self.ndim} or TupleVector"))
    
    def __copy__(self) -> Self:
        return deepcopy(self)
    
    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    
    def component_wise(self, method: Callable[[Vector], Vector]) -> Self:
        if not callable(method):
            raise TypeError(Errors.type_error(f"To apply a method componentwise to a TupleVector the method has to be a callable. {method} is not callable"))
        return TupleVector([method(s_k) for s_k in self])

class VectorSpaceBase:
    r"""Base class for vector representations and their storage operations.

    Vectors themselves are duck typed. Generic RegPy code may rely only on
    linear operations: addition, subtraction, scalar multiplication and
    division, unary negation, and their in-place variants. It must not assume
    that a vector is a NumPy array, exposes ``shape`` or ``dtype``, supports
    indexing or componentwise comparisons, or provides a ``copy`` method.

    A concrete ``VectorSpaceBase`` connects those vectors to operations that
    cannot be expressed purely through linear algebra. In particular, it must
    recognize its elements via :meth:`__contains__` and provide appropriate
    allocation and copying operations. Generic code that needs an independent
    copy should allocate it with :meth:`empty` and populate it with
    :meth:`copyto`.

    Subclasses may override the methods directly. Alternatively, ``type`` may
    be an adapter object that supplies methods such as ``zeros``, ``ones``,
    ``empty``, ``vdot``, ``copyto``, and the logical operations. Specialized
    vector spaces may expose further capabilities such as indexing, masking,
    flattening, or pointwise operations; algorithms using those capabilities
    should explicitly restrict their accepted vector spaces.

    Vector spaces can be added to construct :class:`DirectSum` instances.

    Parameters
    ----------
    vec_type : object
        Class used to recognize vectors belonging to this space.
    shape : tuple(int) or int
        The general shape of the vectors.
    complex : boolean, optional
        Whether vectors have complex coefficients. Defaults to ``False``.
    type : {None,object}, optional
        Optional adapter implementing storage operations for ``vec_type``.
        Defaults to ``None``.
    """

    log = ClassLogger()

    def __init__(self, 
                 vec_type: type[Vector] | tuple[type[Vector], ...],
                 shape: int | tuple[int, ...],
                 complex : bool = False, 
                 type: Any | None = None) -> None:
        self.vec_type = vec_type
        """The vector type"""
        self.shape = (shape,) if isinstance(shape,int) else shape
        """The vector space's shape"""
        self.is_complex = complex
        self.type = type 
        """Type of the vectors if different"""
        self._no_pickle = {'type', 'vec_type'}
        """A dictionary containing modules kept extra in the copy"""

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in self._no_pickle:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def zeros(self) -> Vector:
        """Return the zero vector of the space.
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.zeros(shape = self.shape)
    
    def ones(self) -> Vector:
        """Return a vector of the space containing all ones.
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.ones(shape = self.shape)

    def empty(self) -> Vector:
        """Return an uninitialized element of the space.
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.empty(shape = self.shape)
    
    def set_rng_seed(self, seed: int | None) -> None:
        set_rng_seed(seed)
    
    def _draw_sample(
        self,
        distribution: str,
        size: int | tuple[int, ...] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray | np.generic:
        """Draw samples from a NumPy random-generator distribution.

        Parameters
        ----------
        distribution : str
            Name of a method on :class:`numpy.random.Generator`.
        size : None | int | tuple(int), optional
            Sample size passed to the distribution. Defaults to
            :attr:`realsize`.
        kwargs : dict
            Additional arguments passed to the distribution method.

        Returns
        -------
        ndarray or scalar
            Samples that :meth:`fromflat` can convert into a vector.
        """
        if size is None:
            size = self.realsize
        elif isinstance(size,int) and size != self.realsize:
            self.log.debug(f"You are sampling on a size specified as an integer that is different from the realsize. This might lead to vectors that are not in the space.")
        elif isinstance(size,tuple) and np.prod(size) != self.realsize:
            self.log.debug(f"You are sampling on a size specified as a tuple of integers that taken as a product is different from the realsize. This might lead to vectors that are not in the space.")
        if seed is not None:
            set_rng_seed(seed)
        try:
            dist = getattr(get_rng(),distribution)
            return dist(size = size, **kwargs)
        except AttributeError:
            raise AttributeError(Errors.generic_message(f"The given distribution {distribution} is unknown to numpy maybe you misspelled, please check the documentation."))
        except TypeError:
            raise TypeError(Errors.type_error(f"The chosen distribution {distribution} might not have an argument size or one of your additional keyword arguments is unknown. Please, only use distributions listed in NumPy that have specified a size Argument."))

    def rand(self, distribution: str = "uniform", **kwargs: Any) -> Vector:
        """Return a randomly sampled vector.

        This distribution is defined by the vector-space representation and
        need not correspond to a canonical distribution for an associated
        Hilbert space.

        For complex vectors, real and imaginary parts are sampled independently.

        The distribution for sampling the random numbers can be specified as a string and is then 
        used by the `draw_sample` function and the on initiation created Generator to draw samples.

        Parameters
        ----------
        distribution : string, optional
            The method name of the distribution as an attribute of the NumPy Generator. Default : "uniform"
        kwargs : dict, optional
            Arguments that can be passed to the sampling method.
        
        Returns
        -------
        Vector
            A vector in the space. The vector is created as a flat NumPy array and then cast to the actual 
            vector and type by the `fromflat` method
        """
        return self.fromflat(self._draw_sample(distribution=distribution))
    
    def poisson(self, x: Vector, **kwargs: Any) -> Vector:
        """Return a Poisson-distributed vector with rate ``x``.

        Parameters
        ----------
        x : self.vec_type
            Poisson rate vector.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="poisson sampling requires the x to be in the vector space!"))
        return self.rand(distribution="poisson", lam = self.flatten(x), **kwargs)
    
    def randn(self, **kwargs: Any) -> Vector:
        """Like `rand`, but using a standard normal distribution."""
        return self.rand(distribution="standard_normal", **kwargs)
    
    def vdot(self, x: Vector, y: Vector) -> float | complex:
        r"""Return the vector dot product defined by this space.

        For complex vectors, the second argument is conjugated.

        Parameters
        ----------
        x : self.type
            First vector.
        y : self.type
            second vector

        Returns
        -------
        float or complex
            The dot product of x and y
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.vdot(x,y)

    def copyto(self, x: Vector, y: Vector) -> None:
        r"""Copy the values of ``y`` into the existing vector ``x``.

        The identity of ``x`` is preserved. Concrete vector spaces may
        override this method when their vectors require a special assignment
        protocol.

        Parameters
        ----------
        x : self.vec_type
            Destination vector.
        y : self.vec_type
            Source vector whose values are copied into ``x``. Generic callers
            should use this method instead of invoking ``y.copy()``.

        """
        if x is y:
            return
        if self.type is None:
            raise NotImplementedError
        self.type.copyto(x, y)

    def fill(self, x: Vector, value: Scalar) -> None:
        r"""Fill the existing vector ``x`` with a scalar value.

        The identity of ``x`` is preserved. Concrete vector spaces may
        override this method when their vectors require a special assignment
        protocol.

        Parameters
        ----------
        x : self.vec_type
            Vector to modify in place.
        value : scalar
            Scalar value assigned to every element of ``x``.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, vec_name="vector", add_info="fill argument is not in the vector space."
            ))
        if not np.isscalar(value):
            raise TypeError(Errors.type_error(
                f"fill value must be a scalar. Got {type(value)}."
            ))
        try:
            x.fill(value)
        except AttributeError as exc:
            raise NotImplementedError from exc

    def logical_and(self, x: Vector, y: Vector) -> Vector:
        """Logical and of two boolean vectors
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.logical_and(x,y) 

    def logical_or(self, x: Vector, y: Vector) -> Vector:
        """Logical or of two boolean vectors
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.logical_or(x,y) 
    
    def logical_not(self, x: Vector) -> Vector:
        """Logical not of a boolean vectors
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.logical_not(x) 
    
    def logical_xor(self, x: Vector, y: Vector) -> Vector:
        """Logical xor of two boolean vectors
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.logical_xor(x,y) 

    def iter_basis(self) -> Iterator[Vector]:
        r"""Generator iterating over the standard basis of the vector space. For efficiency,
        the same vector should be returned in each step and subsequently modified in-place. If you need
        the vector longer than that, perform a copy. In case of a complex vector space, after each
        vector with a real one it should return the same vector with \(1i\)
        in its place.
        """
        raise NotImplementedError
    
    def __iter__(self) -> Iterator[Vector]:
        return self.iter_basis()
    
    @property
    def size(self) -> int:
        """The number of coefficients used to represent an element."""
        return np.prod(self.shape)

    @property
    def realsize(self) -> int:
        """The dimension of the vector space as a real vector space. For complex
        coefficients, this is twice :attr:`size`."""
        if self.is_complex:
            return 2 * np.prod(self.shape)
        else:
            return np.prod(self.shape)

    @property
    def ndim(self) -> int:
        """The number of representation dimensions, i.e. the length of ``shape``."""
        return len(self.shape)

    @memoized_property
    def identity(self) -> Operator:
        """The `regpy.operators.Identity` operator on this vector space. """
        from regpy.operators import Identity
        return Identity(self)

    def __contains__(self, x: object) -> bool:
        return isinstance(x,self.vec_type) and all(xis == si for xis,si in zip(x.shape, self.shape))

    def flatten(self, x: Vector) -> np.ndarray:
        r"""Transform the vector `x`, an element of the vector space, into a flattened vector. Inverse
        to `fromflat`.

        Parameters
        ----------
        x : self.vec_type
            The vector to transform.

        Returns
        -------
        array
            The flattened array. If memory layout allows, it will be a view into `x`.
        """
        raise NotImplementedError

    def fromflat(self, x: np.ndarray) -> Vector:
        """Transform a flattened vector into an element of the vector space. Inverse to `flatten`.

        Parameters
        ----------
        x : numpy.ndarray
            The flat vector to transform

        Returns
        -------
        Vector
            The reconstructed vector.
        """
        raise NotImplementedError

    def complex_space(self) -> VectorSpaceBase:
        """Compute the corresponding complex vector space.

        Returns
        -------
        regpy.vecsps.VectorSpaceBase
            The complex space corresponding to this vector space.
        """
        raise NotImplementedError

    def real_space(self) -> VectorSpaceBase:
        """Compute the corresponding real vector space.

        Returns
        -------
        regpy.vecsps.VectorSpaceBase
            The real space corresponding to this vector space.
        """
        raise NotImplementedError
    
    def masked_space(self, mask: object) -> VectorSpaceBase:
        """Gives a masked space given a mask.

        Parameters
        ----------
        mask : boolean
            mask for masking the vector space
        
        Returns
        -------
        regpy.vecsps.VectorSpaceBase
            The masked Space depending on the vector space.
        """
        raise NotImplementedError
    
    def IfPos(self, x: Vector) -> object:
        """Analyses which components contribute to the positive part of the 
        function corresponding to the vector.

        Parameters
        ----------
        x : Vector
            The vector to analyse.

        Returns
        -------
        mask : object
            Mask for the vector components contributing to the positive part of a function.
        """
        raise NotImplementedError
    
    def norm(self, x: Vector) -> float:
        return sqrt(self.vdot(x,x).real)

    def __eq__(self, other: VectorSpaceBase) -> bool:
        if not isinstance(other, type(self)):
            return False
        return (self.type == other.type and
                self.shape == other.shape and
                self.is_complex == other.is_complex and
                self.vec_type == other.vec_type)
    
    def compatible(self,other: VectorSpaceBase) -> bool:
        """Tests compatibility of vectors! This is not as restrictive as `__eq__`
        in the sense, that we only compare the underlying vector type, shape and
        if both are complex. This guarantees that vectors of the space can be 
        also considered as vectors of the other space. Note that the interpretation
        of the vectors might be significantly different! 

        Parameters
        ----------
        other : object
            Another vector space to be compared with.

        Returns
        -------
        boolean
            True if the vectors are compatible
        
        Raise
        -----
        ValueError
            If other is not a `regpy.vecsps.VectorSpaceBase`.
        """
        if not isinstance(other, VectorSpaceBase):
            raise ValueError(Errors.not_instance(other,VectorSpaceBase,add_info="TO have a compatible vector space the other object has to be a vector space in RegPy."))
        return (self.vec_type == other.vec_type and 
                self.shape == other.shape and
                self.is_complex == other.is_complex
                )
        
    def __iadd__(self, other: object) -> DirectSum | NotImplementedType:
        if isinstance(other, VectorSpaceBase):
            return DirectSum(self, other, flatten=True)
        else:
            return NotImplemented

    def __add__(self, other: object) -> DirectSum | NotImplementedType:
        if isinstance(other, VectorSpaceBase):
            return DirectSum(self, other, flatten=True)
        else:
            return NotImplemented

    def __radd__(self, other: object) -> DirectSum | NotImplementedType:
        if isinstance(other, VectorSpaceBase):
            return DirectSum(other, self, flatten=True)
        else:
            return NotImplemented
        
    def __pow__(self, power: int) -> VectorSpaceBase:
        if not isinstance(power, int):
            raise TypeError(Errors.not_instance(power,int,add_info="Construction of powers of a vector space only supported for integer!"))
        domain = self
        for i in range(power-1):
            domain = DirectSum(domain, self, flatten=True)
        return domain
    
    def __repr__(self) -> str:
        return make_repr(self,self.shape,self.is_complex)


class DirectSum(VectorSpaceBase):
    r"""The direct sum of an arbitrary number of vector spaces.

    Elements of the direct sum are represented by :class:`TupleVector` instances.

    Note that constructing DirectSum instances can be done more comfortably simply by adding
    `regpy.vecsps.VectorSpaceBase` instances. However, for generic code, when it's not known whether the summands
    are themselves direct sums, it's better to avoid the `+` overload due the `flatten` parameter
    (see below), since otherwise the number of summands is not fixed.

    DirectSum instances can be indexed and iterated over, returning / yielding the component
    vector spaces.

    Parameters
    ----------
    *summands : tuple(regpy.vecsps.VectorSpaceBase)
        The vector spaces to be summed.
    flatten : bool, optional
        Whether summands that are themselves `DirectSum`s should be merged into this instance. If
        False, DirectSum is not associative, but the join and split methods behave more
        predictably. Default: False, but will be set to True when constructing the DirectSum via
        VectorSpaceBase.__add__, i.e. when using the `+` operator, in order to make repeated sums
        like `A + B + C` unambiguous.
    """

    def __init__(self, *summands: VectorSpaceBase, flatten: bool = False) -> None:
        if any(not isinstance(s, VectorSpaceBase) for s in summands):
            raise TypeError(Errors.type_error(f"The list of summands for a DirectSum vector space contains elements which are not a RegPy vector space! Given "+"\n\t "+f"summands = {summands}"))
        if flatten:
            self.summands = []
            for s in summands:
                if isinstance(s,DirectSum):
                    self.summands.extend(s.summands)
                else:
                    self.summands.append(s)
        else:
            self.summands = summands
        self.n_components = len(self.summands)
        shape = tuple(s.shape for s in self.summands)
        super().__init__(vec_type=TupleVector,shape=shape,complex=all((s.is_complex for s in self.summands)))

    @property
    def size(self) -> int:
        return sum([s.size for s in self.summands]) 
    
    @property
    def realsize(self) -> int:
        return sum([s.realsize for s in self.summands]) 

    def zeros(self) -> TupleVector:
        return TupleVector([s.zeros() for s in self.summands])
    
    def ones(self)-> TupleVector:
        return TupleVector([s.ones() for s in self.summands])
    
    def empty(self)-> TupleVector:
        return TupleVector([s.empty() for s in self.summands])
    
    def rand(self, distribution: str = "uniform", **kwargs: Any) -> TupleVector:
        return TupleVector([s.rand(distribution=distribution, **kwargs) for s in self.summands])
    
    def poisson(self, x: TupleVector) -> TupleVector:
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="Argument in poisson not in the VectorSpace."))
        return TupleVector([s.poisson(x_k) for x_k,s in zip(x,self.summands)])

    def vdot(self, x : TupleVector, y : TupleVector) -> float | complex:
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="First argument in vdot not in the VectorSpace."))
        if y not in self:
            raise ValueError(Errors.not_in_vecsp(y,self,add_info="Second argument in vdot not in the VectorSpace."))
        return sum([s_i.vdot(x_i, y_i) for x_i,y_i,s_i in zip(x,y,self.summands) ])

    def copyto(self, x: TupleVector, y: TupleVector) -> None:
        r"""Copy a direct-sum vector componentwise into ``x``.

        Each component is copied by its own summand. This also supports direct
        sums containing NGSolve vector spaces without replacing component
        objects or allocating a new :class:`TupleVector`.
        """
        if x is y:
            return
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, vec_name="destination", add_info="copyto destination is not in the DirectSum."
            ))
        if not isinstance(y, TupleVector) or y.ndim != self.n_components:
            raise ValueError(Errors.not_in_vecsp(
                y, self, vec_name="source", add_info="copyto source is not compatible with the DirectSum."
            ))
        for summand, x_i, y_i in zip(self.summands, x, y):
            summand.copyto(x_i, y_i)

    def fill(self, x: TupleVector, value: Scalar) -> None:
        r"""Fill a direct-sum vector componentwise with ``value``."""
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, vec_name="vector", add_info="fill argument is not in the DirectSum."
            ))
        if not np.isscalar(value):
            raise TypeError(Errors.type_error(
                f"fill value must be a scalar. Got {type(value)}."
            ))
        for summand, x_i in zip(self.summands, x):
            summand.fill(x_i, value)

    def logical_and(self, x: TupleVector, y: TupleVector) -> TupleVector:
        return TupleVector([s.logical_and(x_i,y_i) for x_i,y_i,s in zip(x.v,y.v,self.summands)])
    
    def logical_or(self, x: TupleVector, y: TupleVector) -> TupleVector:
        return TupleVector([s.logical_or(x_i,y_i) for x_i,y_i,s in zip(x.v,y.v,self.summands)])
    
    def logical_not(self, x: TupleVector) -> TupleVector:
        return TupleVector([s.logical_not(x_i) for x_i,s in zip(x.v,self.summands)])
    
    def logical_xor(self, x: TupleVector, y: TupleVector) -> TupleVector:
        return TupleVector([s.logical_xor(x_i,y_i) for x_i,y_i,s in zip(x.v,y.v,self.summands)])

    def complex_space(self) -> DirectSum:
        return DirectSum(*[s.complex_space() for s in self.summands])

    def real_space(self) -> DirectSum:
        return DirectSum(*[s.real_space() for s in self.summands])
    
    def flatten(self, x : TupleVector) -> np.ndarray:
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="Argument in flatten not in the VectorSpace."))
        return np.asarray([s.flatten(x_i) for x_i,s in zip(x.v,self.summands)])
    
    def fromflat(self, x : np.ndarray) -> TupleVector:
        if x.ndim == 1 and np.isreal(x).all() and x.size == self.realsize:
            ret = []
            ind = 0
            for s in self.summands:
                if s.is_complex:
                    ret.append(s.fromflat(x[ind:ind+2*s.size]))
                    ind += 2*s.size
                else:
                    ret.append(s.fromflat(x[ind:ind+s.size]))
                    ind += s.size
            return TupleVector(ret)
        else:
            raise ValueError(Errors.value_error("Construction of a vector from a flat vector only supported for x an np.ndarray not {}".format(type(x))))
    
    def compatible(self, other: VectorSpaceBase) -> bool:
        return (
            isinstance(other, DirectSum)
            and len(self.summands) == len(other.summands)
            and all(
                own.compatible(other_)
                for own, other_ in zip(
                    self.summands, other.summands
                )
            )
        )

    def iter_basis(self) -> Iterator[TupleVector]:
        for i,s in enumerate(self.summands()):
            vec = self.zeros()
            for b in s.iter_basis():
                vec.v[i] = b
                yield vec

    def masked_space(self, mask: object) -> DirectSum:
        if isinstance(mask,int):
            self.summands[mask]
        x = self.rand()
        _ = x[mask]
        x[mask] = self.ones()[mask]
        return DirectSum(*[s_i.masked_space(m_i) for s_i,m_i in zip(self.summands,mask)])
    
    def IfPos(self, x: TupleVector) -> tuple[object, ...]:
        """Analyses which components contribute to the positive part of the 
        function corresponding to the vector.

        Parameters
        ----------
        x : TupleVector
            The vector to analyse.

        Returns
        -------
        mask : tuple(BitArray)
            Tuple of masks for the vector components contributing to the positive part of a function.
        """
        if not x in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="IfPos only defined for vectors in the space!"))
        return tuple(s_i.IfPos(x_i) for s_i,x_i in zip(self.summands,x.v))

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if isinstance(other, type(self)):
            return (
                len(self.summands) == len(other.summands) and
                all(s == t for s, t in zip(self.summands, other.summands))
            )
        else:
            return NotImplemented
        
    def __contains__(self, x: object) -> bool:
        return isinstance(x,TupleVector) and x.ndim == self.n_components and all([x_i in s_i for x_i,s_i in zip(x,self.summands)])
        

    def join(self, *xs: Vector) -> TupleVector:
        r"""Transform a collection of elements of the summands to an element of the direct sum.

        Parameters
        ----------
        *xs : tuple of Vector
            The elements of the summands. The number should match the number of summands,
            and for all `i`, `xs[i]` should be an element of `self[i]`.

        Returns
        -------
        TupleVector
            An element of the direct sum
        """
        if any(x not in s for s, x in zip(self.summands, xs)):
            raise ValueError(Errors.value_error(f"One of the given vectors to join does not belong to the corresponding summand: "+"\n\t "+f"xs = {xs},"+"\n\t "+f"summands = {self.summands} "))
        return TupleVector(list(xs))

    def split(self, x: TupleVector) -> tuple[Vector, ...]:
        r"""Split an element of the direct sum into a tuple of elements of the summands.

        The result contains the component vectors of `x`.

        Parameters
        ----------
        x : TupleVector
            An element of the direct sum.

        Returns
        -------
        tuple of Vector
            The components of x for the summands.
        """
        if not x in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="split only defined for vectors in the space!"))
        return tuple(x.v)

    def __getitem__(
        self, item: int | slice
    ) -> VectorSpaceBase | list[VectorSpaceBase] | tuple[VectorSpaceBase, ...]:
        return self.summands[item]

    def __iter__(self) -> Iterator[VectorSpaceBase]:
        return iter(self.summands)

    def __len__(self) -> int:
        return len(self.summands)
