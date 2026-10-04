from __future__ import annotations

from typing import Literal, overload

from typing import TYPE_CHECKING
from unittest import result
if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike
    from regpy.operators.base import Operator

from copy import copy,deepcopy
from math import sqrt
from dataclasses import dataclass
from types import NotImplementedType
from typing import Any, Callable, Iterable, Iterator, Literal, overload, Protocol, runtime_checkable, TYPE_CHECKING

from typing_extensions import Self


import numpy as np
import warnings


from regpy.util import ClassLogger,Errors,make_repr,memoized_property,get_rng, set_rng_seed

__all__ = [
    "Scalar", "ReadOnlyVector", "Vector", "TupleVector", "VectorSpaceBase", "VectorSpace", "DirectSum"
]


Scalar = int | float | complex | np.number


class ReadOnlyVector(Protocol):
    r"""Structural type for vectors that generic RegPy code may only read.

    The protocol contains the non-mutating linear operations of a vector. It
    deliberately excludes item assignment and all in-place operators. As with
    :class:`Vector`, this is a structural typing aid and does not require
    concrete vector classes to inherit from it.
    """

    def __add__(self, other: Self, /) -> Self: ...

    def __radd__(self, other: Self, /) -> Self: ...

    def __sub__(self, other: Self, /) -> Self: ...

    def __rsub__(self, other: Self, /) -> Self: ...

    def __mul__(self, scalar: Scalar, /) -> Self: ...

    def __rmul__(self, scalar: Scalar, /) -> Self: ...

    def __truediv__(self, scalar: Scalar, /) -> Self: ...

    def __neg__(self) -> Self: ...


class Vector(ReadOnlyVector, Protocol):
    r"""Structural type for writable vectors accepted by generic RegPy code.

    In addition to the non-mutating operations of :class:`ReadOnlyVector`, a
    writable vector supports the in-place linear operations documented by
    :class:`VectorSpaceBase`. Allocation, copying, scalar products and other
    representation-dependent operations belong to the vector space instead.

    Vectors are duck-typed. This class is only used for typing and for clarity,
    but does not influence the behavior of the code. In particular, vector classes used 
    in RegPy do not need to inherit from it. 
    """

    def __iadd__(self, other: Self, /) -> Self: ...

    def __isub__(self, other: Self, /) -> Self: ...

    def __imul__(self, scalar: Scalar, /) -> Self: ...

    def __itruediv__(self, scalar: Scalar, /) -> Self: ...

@dataclass 
class TupleVector:
    v: list[Vector]
    read_only: bool = False

    __array_ufunc__ = None

    log = ClassLogger()

    def copy(self) -> Self:
        r"""Copy.
        
        
        Returns
        -------
        Self
            Result described above.
        """
        return copy(self)

    def __post_init__(self) -> None:
        if not isinstance(self.read_only, (bool, np.bool_)):
            raise TypeError(
                f"read_only must be boolean. Got {type(self.read_only)}."
            )
        self.read_only = bool(self.read_only)
        self.v = list(self.v)
        self.types = [type(v_i) for v_i in self.v]
        self.ndim = len(self.types)

    def _require_writable(self, operation: str) -> None:
        if self.read_only:
            raise ValueError(
                f"Cannot perform {operation} on a read-only TupleVector."
            )

    def __len__(self) -> int:
        return self.ndim
    

    

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
        r"""All.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return all((v_i.all() for v_i in self.v))

    def any(self) -> bool:
        r"""Any.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return any((v_i.any() for v_i in self.v))

    def __or__(self, other: TupleVector) -> TupleVector:
        if not isinstance(other,TupleVector) or self.ndim != other.ndim:
            raise ValueError(Errors.value_error("Comparing with TupleVector only supported for TupleVectors of identical dimension!"))
        return TupleVector([s_i | o_i for s_i,o_i in zip(self.v,other.v)])
    
    def sum(self, **kwargs: Any) -> Scalar:
        r"""Sum.
        
        Parameters
        ----------
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        Scalar
            Computed scalar value.
        """
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
        self._require_writable("in-place addition")
        if not isinstance(other,TupleVector) or other.ndim != self.ndim or any([t_o!=t_s for t_o,t_s in zip(other.types,self.types)]):
            raise ValueError(Errors.value_error("Adding TupleVector only supported for TupleVectors of identical dimension and identical type in each component!"))
        for k,v_o in enumerate(other.v):
            self.v[k] += v_o
        return self

    def __isub__(self, other: TupleVector) -> Self:
        self._require_writable("in-place subtraction")
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
        self._require_writable("in-place multiplication")
        if not isinstance(other,(float,int,complex)):
            raise ValueError(Errors.value_error("Multiplying TupleVector only supported for scalars (int, float, or complex)!"))
        for k in range(self.ndim):
            self.v[k] *= other
        return self
    
    def __itruediv__(self, other: Scalar) -> Self:
        self._require_writable("in-place division")
        if not isinstance(other,(float,int,complex)):
            raise ValueError(Errors.value_error("Division TupleVector only supported for scalars (int, float, or complex)!"))
        for k in range(self.ndim):
            self.v[k] /= other
        return self
    
    def __mul__(self, other: Scalar | Operator) -> Self | Operator:
        r"""Multiply by a scalar or compose an operator with a multiplier.
        
        Multiplication of two vectors is deliberately unsupported because it
        is not part of the :class:`Vector` protocol. Pointwise multiplication
        is provided by :class:`regpy.operators.PtwMultiplication`, whose
        generic implementation uses the coefficient interface of the
        corresponding vector space.
        
        Parameters
        ----------
        other: Scalar | Operator
            Other.
        
        
        Returns
        -------
        Self | Operator
            Result described above.
        """
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
        self._require_writable("item assignment")
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
        result = type(self).__new__(type(self))
        memo[id(self)] = result

        for key, value in self.__dict__.items():
            if key.startswith("__memoized_"):
                continue
            setattr(result, key, deepcopy(value, memo))

        return result

         
    def component_wise(self, method: Callable[[Vector], Vector]) -> Self:
        r"""Component wise.
        
        Parameters
        ----------
        method: Callable[[Vector], Vector]
            Writable vector used by this operation.
        
        
        Returns
        -------
        Self
            Result described above.
        """
        if not callable(method):
            raise TypeError(Errors.type_error(f"To apply a method componentwise to a TupleVector the method has to be a callable. {method} is not callable"))
        return TupleVector([method(s_k) for s_k in self])

@runtime_checkable
class VectorSpace(Protocol):
    """Structural scalar-field vector-space contract.

    A backend need not provide shape, ndim, real_space, complex_space or
    componentwise vector attributes. Mathematical coordinates, basis vectors
    and flattening must all agree and have length dimension. Implementations
    may derive from VectorSpaceBase to reuse its defaults.
    """
    is_complex: bool
    as_real_space: bool

    @property
    def dimension(self) -> int: ...
    @overload
    def zeros(self, read_only: Literal[False]=False) -> Vector:
        ...

    @overload
    def zeros(self, read_only: Literal[True]) -> ReadOnlyVector:
        ...

    @overload
    def zeros(self, read_only: bool) -> Vector | ReadOnlyVector:
        ...

    def zeros(self, read_only: bool = False) -> Vector | ReadOnlyVector: ...
    @overload
    def ones(self, read_only: Literal[False]=False) -> Vector:
        ...

    @overload
    def ones(self, read_only: Literal[True]) -> ReadOnlyVector:
        ...

    @overload
    def ones(self, read_only: bool) -> Vector | ReadOnlyVector:
        ...

    def ones(self, read_only: bool = False) -> Vector | ReadOnlyVector: ...
    def empty(self) -> Vector: ...
    def read_only(self, x: ReadOnlyVector) -> ReadOnlyVector: ...
    def __contains__(self, x: object) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def compatible(self, other: object) -> bool: ...
    def copyto(self, x: Vector, y: ReadOnlyVector) -> None: ...
    def fill(self, x: Vector, value: Scalar) -> None: ...
    def vdot(self, x: ReadOnlyVector, y: ReadOnlyVector) -> Scalar: ...
    def norm(self, x: ReadOnlyVector) -> float: ...
    def iter_coefficients(self, x: ReadOnlyVector) -> Iterator[Scalar]: ...
    def set_coefficients(self, x: ReadOnlyVector, coefficients: Iterable[Scalar]) -> None: ...
    def iter_basis(self) -> Iterator[ReadOnlyVector]: ...
    def flatten(self, x: ReadOnlyVector) -> np.ndarray: ...
    def fromflat(self, x: np.ndarray) -> Vector: ...
    def rand(self, distribution: str = "uniform", **kwargs: Any) -> Vector: ...
    def randn(self, **kwargs: Any) -> Vector: ...


class VectorSpaceBase:
    r"""Base implementation of the VectorSpace protocol.
    
    No shape, ndim, vector indexing, real/imag/conj attributes or storage
    layout are assumed. Concrete spaces provide allocation, membership,
    equality, compatibility and coefficient access. Coefficients describe
    coordinates over R when as_real_space=True, otherwise over C. Complex
    storage is necessary, but not sufficient, for a complex vector space.
    
    dimension in this constructor counts native complex coordinates when
    complex=True; the dimension property doubles it for real scalars.
    real_space/complex_space conversions
    are optional backend capabilities, not part of this base interface.
    
    Parameters
    ----------
    vec_type: type[Vector]
        Writable vector used by this operation.
    dimension: int
        Dimension.
    complex: bool [default: False]
        Complex.
    type: DTypeLike [default: None]
        Type.
    as_real_space: bool [default: True]
        Whether to use real scalars. With complex storage, True treats the
        space as real and counts real and imaginary coordinates separately.
    """

    log = ClassLogger()


    def __init__(self, vec_type: type[Vector], dimension: int, complex: bool = False,
                 type: DTypeLike=None, as_real_space: bool = True) -> None:
        if not isinstance(dimension, (int, np.integer)) or dimension < 0:
            raise ValueError("dimension must be a nonnegative integer.")
        self.vec_type = vec_type
        self.is_complex = bool(complex)
        self._coordinate_count = int(dimension)
        self.type = type
        self._no_pickle = {'type', 'vec_type'}
        self.as_real_space = as_real_space

    @property
    def as_real_space(self) -> bool:
        r"""True for real scalars; False for complex scalars.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self._as_real_space

    @as_real_space.setter
    def as_real_space(self, value: bool) -> None:
        r"""As real space.
        
        Parameters
        ----------
        value: bool
            Value.
        """
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("as_real_space must be boolean.")
        if not value and not self.is_complex:
            raise ValueError("A non-complex representation requires as_real_space=True.")
        self._as_real_space = bool(value)

    @property
    def dimension(self) -> int:
        r"""Dimension over the chosen scalar field; not a storage shape.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        return self._coordinate_count * (2 if self.is_complex and self.as_real_space else 1)

    @property
    def _native_dimension(self) -> int:
        return self._coordinate_count

    @staticmethod
    def _validate_read_only_flag(read_only: bool) -> bool:
        if not isinstance(read_only, (bool, np.bool_)):
            raise TypeError(
                f"read_only must be boolean. Got {type(read_only)}."
            )
        return bool(read_only)

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k.startswith("__memoized_"):
                continue
            if k in self._no_pickle:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    @overload
    def zeros(self, read_only: Literal[False]=False) -> Vector:
        ...

    @overload
    def zeros(self, read_only: Literal[True]) -> ReadOnlyVector:
        ...

    @overload
    def zeros(self, read_only: bool) -> Vector | ReadOnlyVector:
        ...

    def zeros(self, read_only: bool = False) -> Vector | ReadOnlyVector:
        r"""Backend-defined representation operation.
        
        Parameters
        ----------
        read_only: bool [default: False]
            Whether to return a read-only view.
        
        
        Returns
        -------
        Vector | ReadOnlyVector
            Result described above.
        """
        raise NotImplementedError


    
    @overload
    def ones(self, read_only: Literal[False]=False) -> Vector:
        ...

    @overload
    def ones(self, read_only: Literal[True]) -> ReadOnlyVector:
        ...

    @overload
    def ones(self, read_only: bool) -> Vector | ReadOnlyVector:
        ...

    def ones(self, read_only: bool = False) -> Vector | ReadOnlyVector:
        r"""Backend-defined representation operation.
        
        Parameters
        ----------
        read_only: bool [default: False]
            Whether to return a read-only view.
        
        
        Returns
        -------
        Vector | ReadOnlyVector
            Result described above.
        """
        raise NotImplementedError



    def read_only(self, x: ReadOnlyVector) -> ReadOnlyVector:
        r"""Return a read-only view of ``x``.
        
        The base implementation returns ``x`` unchanged. Backends that can
        enforce write protection should override this method. No independent
        copy is required or implied.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        ReadOnlyVector
            Read-only vector view.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, vec_name="vector",
                add_info="read_only argument is not in the vector space."
            ))
        return x

    def empty(self) -> Vector:
        r"""Backend-defined representation operation.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        raise NotImplementedError
    
    def set_rng_seed(self, seed: int | None) -> None:
        r"""Set rng seed.
        
        Parameters
        ----------
        seed: int | None
            Seed.
        """
        set_rng_seed(seed)
    
    def _draw_sample(
        self,
        distribution: str,
        size: int | tuple[int, ...] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray | np.generic:
        r"""Draw samples from a NumPy random-generator distribution.
        
        Parameters
        ----------
        distribution: str
            Name of a method on :class:`numpy.random.Generator`.
        size: int | tuple[int, ...] | None [default: None]
            Sample size passed to the distribution.
        seed: int | None [default: None]
            Seed.
        **kwargs: Any
            Additional arguments passed to the distribution method.
        
        Returns
        -------
        np.ndarray | np.generic
            Samples that :meth:`fromflat` can convert into a vector.
        """
        if size is None:
            size = self.dimension
        elif isinstance(size,int) and size != self.dimension:
            self.log.debug(f"You are sampling on a size specified as an integer that is different from the realsize. This might lead to vectors that are not in the space.")
        elif isinstance(size,tuple) and np.prod(size) != self.dimension:
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
        r"""Return a randomly sampled vector.
        
        This distribution is defined by the vector-space representation and
        need not correspond to a canonical distribution for an associated
        Hilbert space.
        
        For complex vectors, real and imaginary parts are sampled independently.
        
        The distribution for sampling the random numbers can be specified as a string and is then 
        used by the `draw_sample` function and the on initiation created Generator to draw samples.
        
        Parameters
        ----------
        distribution: str [default: 'uniform']
            The method name of the distribution as an attribute of the NumPy Generator. Default : "uniform"
        **kwargs: Any
            Arguments that can be passed to the sampling method.
        
        Returns
        -------
        Vector
            A vector in the space. The vector is created as a flat NumPy array and then cast to the actual 
            vector and type by the `fromflat` method
        """
        values = self._draw_sample(distribution=distribution, **kwargs)
        if not self.as_real_space:
            values = values + 1j * self._draw_sample(distribution=distribution, **kwargs)
        return self.fromflat(values)
    
    def poisson(self, x: ReadOnlyVector, **kwargs: Any) -> Vector:
        r"""Return a Poisson-distributed vector with rate ``x``.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Poisson rate vector.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="poisson sampling requires the x to be in the vector space!"))
        return self.rand(distribution="poisson", lam = self.flatten(x), **kwargs)
    
    def randn(self, **kwargs: Any) -> Vector:
        r"""Like `rand`, but using a standard normal distribution.
        
        Parameters
        ----------
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        return self.rand(distribution="standard_normal", **kwargs)
    
    def vdot(
        self, x: ReadOnlyVector, y: ReadOnlyVector
    ) -> Scalar:
        r"""Return the vector dot product defined by this space.
        
        For complex vectors, the second argument is conjugated.
        
        Parameters
        ----------
        x: ReadOnlyVector
            First vector.
        y: ReadOnlyVector
            second vector
        
        Returns
        -------
        Scalar
            The dot product of x and y
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.vdot(x,y)

    def copyto(self, x: Vector, y: ReadOnlyVector) -> None:
        r"""Copy the values of ``y`` into the existing vector ``x``.
        
        The identity of ``x`` is preserved. Concrete vector spaces may
        override this method when their vectors require a special assignment
        protocol.
        
        Parameters
        ----------
        x: Vector
            Destination vector.
        y: ReadOnlyVector
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
        x: Vector
            Vector to modify in place.
        value: Scalar
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


    def iter_coefficients(self, x: ReadOnlyVector) -> Iterator[Scalar]:
        r"""Yield dimension scalars in basis order.
        
        For complex storage interpreted over R, each real part is immediately
        followed by its imaginary part. Backends may override this public
        method directly; it never requires a shape attribute.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Yields
        ------
        Scalar
            Result described above.
        """
        for value in self._iter_native_coefficients(x):
            if self.is_complex and self.as_real_space:
                yield np.real(value)
                yield np.imag(value)
            else:
                yield value

    def _iter_native_coefficients(self, x: ReadOnlyVector) -> Iterator[Scalar]:
        r"""Backend hook used by coordinate-storage implementations.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Iterator[Scalar]
            Result described above.
        """
        raise NotImplementedError


    def set_coefficients(self, x: Vector, coefficients: Iterable[Scalar]) -> None:
        r"""Write exactly dimension coordinates, preserving x's identity.
        
        Realified complex coordinates must be consecutive (real, imaginary)
        pairs. Input length and scalar field are checked before any write.
        
        Parameters
        ----------
        x: Vector
            Writable vector used by this operation.
        coefficients: Iterable[Scalar]
            Coefficients.
        """
        values = np.asarray(list(coefficients))
        if values.shape != (self.dimension,):
            raise ValueError("Expected exactly dimension coefficients.")
        if self.as_real_space:
            if np.any(np.imag(values) != 0):
                raise ValueError("Real coordinates must be real.")
            values = values.real
        if self.is_complex and self.as_real_space:
            values = values[0::2] + 1j * values[1::2]
        self._set_native_coefficients(x, values)

    def _set_native_coefficients(self, x: ReadOnlyVector, coefficients: Iterable[Scalar]) -> None:
        r"""Backend hook used by coordinate-storage implementations.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        coefficients: Iterable[Scalar]
            Coefficients.
        """
        raise NotImplementedError


    
    


    def iter_basis(self) -> Iterator[ReadOnlyVector]:
        r"""Yield the coordinate basis; the mutable work vector may be reused.
        
        
        Yields
        ------
        ReadOnlyVector
            Result described above.
        """
        values = np.zeros(self.dimension, dtype=float if self.as_real_space else complex)
        result = self.empty()
        for j in range(self.dimension):
            values[j] = 1
            self.set_coefficients(result, values)
            yield result
            values[j] = 0
    
    def __iter__(self) -> Iterator[ReadOnlyVector]:
        return self.iter_basis()
    

    @property
    def size(self) -> int:
        r"""Deprecated alias of dimension over the selected scalar field.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        warnings.warn("size is deprecated; use dimension.", DeprecationWarning, stacklevel=2)
        return self.dimension


    @property
    def realsize(self) -> int:
        r"""Deprecated alias of dimension over the selected scalar field.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        warnings.warn("realsize is deprecated; use dimension.", DeprecationWarning, stacklevel=2)
        return self.dimension


    @memoized_property
    def identity(self) -> Operator:
        r"""The `regpy.operators.Identity` operator on this vector space. 
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        from regpy.operators import Identity
        return Identity(self)

    def __contains__(self, x: ReadOnlyVector) -> bool:
        r"""Backend-defined representation operation.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        raise NotImplementedError


    def flatten(self, x: ReadOnlyVector) -> np.ndarray:
        r"""Coordinates over the selected scalar field, in basis order.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        return np.asarray(list(self.iter_coefficients(x)),
                          dtype=float if self.as_real_space else complex)


    def fromflat(self, x: np.ndarray) -> Vector:
        r"""Reconstruct from exactly dimension coordinates; inverse of flatten.
        
        Parameters
        ----------
        x: np.ndarray
            X.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        values = np.asarray(x)
        if values.shape != (self.dimension,):
            raise ValueError("Expected a one-dimensional array of length dimension.")
        if self.as_real_space and np.any(np.imag(values) != 0):
            raise ValueError("Real coordinates must have zero imaginary part.")
        result = self.empty()
        self.set_coefficients(result, values.real if self.as_real_space else values)
        return result




    
    
    
    def norm(self, x: ReadOnlyVector) -> float:
        r"""Norm.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return sqrt(self.vdot(x,x).real)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        r"""Backend-defined representation operation.
        
        Parameters
        ----------
        other: object
            Other.
        
        
        Returns
        -------
        bool | NotImplementedType
            Result described above.
        """
        raise NotImplementedError
    
    def compatible(self, other: object) -> bool:
        r"""Backend-defined representation operation.
        
        Parameters
        ----------
        other: object
            Other.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        raise NotImplementedError
        
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
        return make_repr(self, self.dimension, as_real_space=self.as_real_space)


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
    flatten: bool [default: False]
        Whether summands that are themselves `DirectSum`s should be merged into this instance. If
        False, DirectSum is not associative, but the join and split methods behave more
        predictably.__add__, i.e. when using the `+` operator, in order to make repeated sums
        like `A + B + C` unambiguous.
    *summands: VectorSpaceBase
        The vector spaces to be summed.
    """

    def __new__(cls, *summands: VectorSpaceBase, **kwargs: Any) -> Self:
        # Local import avoids a base -> maskable -> base import cycle.
        if cls is DirectSum:
            from .maskable import MaskableVectorSpace, MaskableDirectSum
            if all(isinstance(s, MaskableVectorSpace) for s in summands):
                return object.__new__(MaskableDirectSum)
        return object.__new__(cls)

    def __init__(self, *summands: VectorSpaceBase, flatten: bool = False) -> None:
        if any(not isinstance(s, VectorSpace) for s in summands):
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
        flags = {s.as_real_space for s in self.summands}
        if len(flags) > 1:
            warnings.warn("Mixed scalar fields in DirectSum; interpreting all summands over R.",
                          RuntimeWarning, stacklevel=2)
            for summand in self.summands:
                summand.as_real_space = True
        super().__init__(vec_type=TupleVector, dimension=0,
                         complex=bool(self.summands) and all(s.is_complex for s in self.summands),
                         as_real_space=all(s.as_real_space for s in self.summands))


    

    @property
    def dimension(self) -> int:
        r"""Dimension.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        return sum(s.dimension for s in self.summands)

    @property
    def as_real_space(self) -> bool:
        r"""As real space.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        flags = {s.as_real_space for s in self.summands}
        if len(flags) > 1:
            raise ValueError("DirectSum summands no longer share a scalar field.")
        return next(iter(flags), True)

    @as_real_space.setter
    def as_real_space(self, value: Scalar | ArrayLike) -> None:
        r"""As real space.
        
        Parameters
        ----------
        value: Scalar | ArrayLike
            Value.
        """
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("as_real_space must be boolean.")
        if not value and not self.is_complex:
            raise ValueError("Mixed/real storage cannot be a complex vector space.")
        for summand in self.summands:
            summand.as_real_space = value
        self._as_real_space = bool(value)

    def _iter_native_coefficients(self, x: ReadOnlyVector) -> Iterator[Scalar]:
        for space, component in zip(self.summands, x, strict=True):
            yield from space._iter_native_coefficients(component)

    def _set_native_coefficients(self, x: ReadOnlyVector, coefficients: Iterable[Scalar]) -> None:
        from itertools import islice
        iterator = iter(coefficients)
        for space, component in zip(self.summands, x, strict=True):
            space._set_native_coefficients(component, list(islice(iterator, space._native_dimension)))
        if next(iterator, None) is not None:
            raise ValueError("Too many coefficients.")

    @property
    def _native_dimension(self) -> int:
        return sum(s._native_dimension for s in self.summands)

    @overload
    def zeros(self, read_only: Literal[False]=False) -> TupleVector:
        ...

    @overload
    def zeros(self, read_only: Literal[True]) -> ReadOnlyVector:
        ...

    @overload
    def zeros(self, read_only: bool) -> TupleVector | ReadOnlyVector:
        ...

    def zeros(self, read_only: bool = False) -> Vector | ReadOnlyVector:
        r"""Create a vector whose coefficients are zero.
        
        Parameters
        ----------
        read_only: bool [default: False]
            Whether to return a read-only view.
        
        
        Returns
        -------
        Vector | ReadOnlyVector
            Result described above.
        """
        read_only = self._validate_read_only_flag(read_only)
        return TupleVector(
            [s.zeros(read_only=read_only) for s in self.summands],
            read_only=read_only,
        )
    
    @overload
    def ones(self, read_only: Literal[False]=False) -> TupleVector:
        ...

    @overload
    def ones(self, read_only: Literal[True]) -> ReadOnlyVector:
        ...

    @overload
    def ones(self, read_only: bool) -> TupleVector | ReadOnlyVector:
        ...

    def ones(self, read_only: bool = False) -> Vector | ReadOnlyVector:
        r"""Create a vector whose coefficients are one.
        
        Parameters
        ----------
        read_only: bool [default: False]
            Whether to return a read-only view.
        
        
        Returns
        -------
        Vector | ReadOnlyVector
            Result described above.
        """
        read_only = self._validate_read_only_flag(read_only)
        return TupleVector(
            [s.ones(read_only=read_only) for s in self.summands],
            read_only=read_only,
        )

    def read_only(self, x: TupleVector) -> ReadOnlyVector:
        r"""Read only.
        
        Parameters
        ----------
        x: TupleVector
            X.
        
        
        Returns
        -------
        ReadOnlyVector
            Read-only vector view.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, vec_name="vector",
                add_info="read_only argument is not in the DirectSum."
            ))
        return TupleVector(
            [s.read_only(x_i) for s, x_i in zip(self.summands, x)],
            read_only=True,
        )
    
    def empty(self)-> TupleVector:
        r"""Allocate a writable vector with uninitialized coefficients.
        
        
        Returns
        -------
        TupleVector
            Result described above.
        """
        return TupleVector([s.empty() for s in self.summands])
    
    def rand(self, distribution: str = "uniform", **kwargs: Any) -> TupleVector:
        r"""Rand.
        
        Parameters
        ----------
        distribution: str [default: 'uniform']
            Distribution.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        TupleVector
            Result described above.
        """
        return TupleVector([s.rand(distribution=distribution, **kwargs) for s in self.summands])
    
    def poisson(self, x: TupleVector) -> TupleVector:
        r"""Poisson.
        
        Parameters
        ----------
        x: TupleVector
            X.
        
        
        Returns
        -------
        TupleVector
            Result described above.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="Argument in poisson not in the VectorSpace."))
        return TupleVector([s.poisson(x_k) for x_k,s in zip(x,self.summands)])

    def vdot(self, x : TupleVector, y : TupleVector) -> float | complex:
        r"""Vdot.
        
        Parameters
        ----------
        x: TupleVector
            X.
        y: TupleVector
            Y.
        
        
        Returns
        -------
        float | complex
            Result described above.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="First argument in vdot not in the VectorSpace."))
        if y not in self:
            raise ValueError(Errors.not_in_vecsp(y,self,add_info="Second argument in vdot not in the VectorSpace."))
        return sum([s_i.vdot(x_i, y_i) for x_i,y_i,s_i in zip(x,y,self.summands) ])

    def copyto(self, x: Vector, y: ReadOnlyVector) -> None:
        r"""Copy a direct-sum vector componentwise into ``x``.
        
        Each component is copied by its own summand. This also supports direct
        sums containing NGSolve vector spaces without replacing component
        objects or allocating a new :class:`TupleVector`.
        
        Parameters
        ----------
        x: Vector
            Writable vector used by this operation.
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        """
        if x is y:
            return
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, vec_name="destination", add_info="copyto destination is not in the DirectSum."
            ))
        x._require_writable("copyto")
        if not isinstance(y, TupleVector) or y.ndim != self.n_components:
            raise ValueError(Errors.not_in_vecsp(
                y, self, vec_name="source", add_info="copyto source is not compatible with the DirectSum."
            ))
        for summand, x_i, y_i in zip(self.summands, x, y):
            summand.copyto(x_i, y_i)

    def fill(self, x: Vector, value: Scalar) -> None:
        r"""Fill a direct-sum vector componentwise with ``value``.
        
        Parameters
        ----------
        x: Vector
            Writable vector used by this operation.
        value: Scalar
            Value.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, vec_name="vector", add_info="fill argument is not in the DirectSum."
            ))
        x._require_writable("fill")
        if not np.isscalar(value):
            raise TypeError(Errors.type_error(
                f"fill value must be a scalar. Got {type(value)}."
            ))
        for summand, x_i in zip(self.summands, x):
            summand.fill(x_i, value)

    def iter_coefficients(
        self, x: ReadOnlyVector
    ) -> Iterator[Scalar]:
        r"""Iter coefficients.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Yields
        ------
        Scalar
            Result described above.
        """
        if not isinstance(x, TupleVector) or x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x,
                self,
                vec_name="vector",
                add_info="iter_coefficients argument is not in the DirectSum."
            ))
        for index, (summand, component) in enumerate(zip(self.summands, x)):
            try:
                yield from summand.iter_coefficients(component)
            except NotImplementedError as error:
                raise NotImplementedError(Errors.generic_message(
                    "Coefficient iteration is unavailable for DirectSum "
                    f"component {index} ({type(summand).__name__})."
                )) from error

    def set_coefficients(
        self,
        x: ReadOnlyVector,
        coefficients: Iterable[Scalar],
    ) -> None:
        r"""Set coefficients.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        coefficients: Iterable[Scalar]
            Coefficients.
        """
        if not isinstance(x, TupleVector) or x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x,
                self,
                vec_name="vector",
                add_info="set_coefficients destination is not in the DirectSum."
            ))
        x._require_writable("setting coefficients")
        iterator = iter(coefficients)
        consumed = 0
        for index, (summand, component) in enumerate(zip(self.summands, x)):
            component_coefficients = []
            for _ in range(int(summand.dimension)):
                try:
                    component_coefficients.append(next(iterator))
                    consumed += 1
                except StopIteration as error:
                    raise ValueError(Errors.value_error(
                        f"Expected {self.dimension} coefficients for the "
                        f"DirectSum, but received only {consumed}."
                    )) from error
            try:
                summand.set_coefficients(component, component_coefficients)
            except NotImplementedError as error:
                raise NotImplementedError(Errors.generic_message(
                    "Setting coefficients is unavailable for "
                    f"DirectSum component {index} ({type(summand).__name__})."
                )) from error
        try:
            next(iterator)
        except StopIteration:
            return
        raise ValueError(Errors.value_error(
            f"Expected {self.dimension} coefficients for the DirectSum, but "
            "received more."
        ))

    
    
    

    def complex_space(self) -> DirectSum:
        r"""Complex space.
        
        
        Returns
        -------
        DirectSum
            The direct sum of the complex versions of all its summand spaces.
        """
        return DirectSum(*[s.complex_space() for s in self.summands])

    def real_space(self) -> DirectSum:
        r"""Real space.
        
        
        Returns
        -------
        DirectSum
            The direct sum of the real versions of all its summand spaces.
        """
        return DirectSum(*[s.real_space() for s in self.summands])
    
    def flatten(self, x : TupleVector) -> np.ndarray:
        r"""Flatten.
        
        Parameters
        ----------
        x: TupleVector
            X.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="Argument in flatten not in the VectorSpace."))
        return np.concatenate([
            np.asarray(s.flatten(x_i)).reshape(-1)
            for x_i, s in zip(x.v, self.summands)
        ])
    
    def fromflat(self, x: ReadOnlyVector) -> Vector:
        r"""Fromflat.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        return super().fromflat(x)


    
    def compatible(self, other: VectorSpaceBase) -> bool:
        r"""Compatible.
        
        Parameters
        ----------
        other: VectorSpaceBase
            Other.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
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
        r"""Iter basis.
        
        
        Yields
        ------
        TupleVector
            Result described above.
        """
        for i,s in enumerate(self.summands()):
            vec = self.zeros()
            for b in s.iter_basis():
                vec.v[i] = b
                yield vec

    

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
        

    @overload
    def join(self, *xs: ReadOnlyVector, read_only: Literal[False]=False) -> TupleVector:
        ...

    @overload
    def join(self, *xs: ReadOnlyVector, read_only: Literal[True]) -> ReadOnlyVector:
        ...

    @overload
    def join(self, *xs: ReadOnlyVector, read_only: bool) -> TupleVector | ReadOnlyVector:
        ...

    def join(self, *xs: ReadOnlyVector, read_only: bool = False) -> TupleVector | ReadOnlyVector:
        r"""Transform a collection of elements of the summands to an element of the direct sum.
        
        Parameters
        ----------
        read_only: bool [default: False]
            Whether to return a read-only view.
        *xs: ReadOnlyVector
            The elements of the summands. The number should match the number of summands,
            and for all `i`, `xs[i]` should be an element of `self[i]`.
        
        Returns
        -------
        TupleVector | ReadOnlyVector
            An element of the direct sum
        """
        if any(x not in s for s, x in zip(self.summands, xs)):
            raise ValueError(Errors.value_error(f"One of the given vectors to join does not belong to the corresponding summand: "+"\n\t "+f"xs = {xs},"+"\n\t "+f"summands = {self.summands} "))
        return TupleVector(list(xs), read_only=read_only)

    def split(self, x: TupleVector) -> tuple[Vector, ...]:
        r"""Split an element of the direct sum into a tuple of elements of the summands.
        
        The result contains the component vectors of `x`.
        
        Parameters
        ----------
        x: TupleVector
            An element of the direct sum.
        
        Returns
        -------
        tuple[Vector, ...]
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
