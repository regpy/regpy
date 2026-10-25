from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, Iterator, Literal
    from types import NotImplementedType
    from numpy.typing import ArrayLike
    from regpy.vecsps.base import Vector
    from regpy._typing import Mask
from copy import copy, deepcopy
from warnings import warn
from typing import *

import numpy as np

from regpy.util import is_complex_dtype,is_real_dtype, Errors, complex2real,real2complex, make_repr, set_rng_seed
from .base import ReadOnlyVector, Scalar, VectorSpaceBase
from .maskable import MaskableVectorSpace, MaskedVectorSpace

__all__ = ["NumPyVectorSpace", "MeasureSpaceFcts", "MaskedMeasureSpaceFcts", "GridFcts", "UniformGridFcts", "Prod"]

class NumPyVectorSpace(MaskableVectorSpace):
    r"""Arrays over real or complex storage with a selectable scalar field.
    
    `regpy.vecsps.VectorSpaceBase`s can be added, producing `DirectSum` instances.
    
    Parameters
    ----------
    shape: tuple | int
        The shape of the arrays representing elements of this vector space.
    dtype: type [default: float]
        The elements' dtype. Should usually be either `float` or `complex`.
    as_real_space: bool [default: True]
        . Complex storage then has twice as many real coordinates.
        False selects complex scalars and requires a complex dtype.
    """

    def __init__(self, shape: tuple | int, dtype: type = float, as_real_space: bool = True) -> None:
        self.shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
        super().__init__(vec_type=np.ndarray, dimension=int(np.prod(self.shape)),
                         complex=is_complex_dtype(np.array([], dtype=dtype)),
                         type=np, as_real_space=as_real_space)
        self.dtype = dtype

    @overload
    def zeros(self, read_only: Literal[False] = False) -> np.ndarray: ...

    @overload
    def zeros(self, read_only: Literal[True]) -> ReadOnlyVector: ...

    def zeros(self, read_only: bool = False) -> np.ndarray | ReadOnlyVector:
        r"""Return a zero vector, optionally as a broadcast read-only array.
        
        Parameters
        ----------
        read_only: bool [default: False]
            Whether to return a read-only view.
        
        
        Returns
        -------
        np.ndarray | ReadOnlyVector
            Result described above.
        """
        read_only = self._validate_read_only_flag(read_only)
        if read_only:
            return np.broadcast_to(
                np.asarray(0, dtype=self.dtype), self.shape
            )
        return np.zeros(shape=self.shape, dtype=self.dtype)
    
    @overload
    def ones(self, read_only: Literal[False] = False) -> np.ndarray: ...

    @overload
    def ones(self, read_only: Literal[True]) -> ReadOnlyVector: ...

    def ones(self, read_only: bool = False) -> np.ndarray | ReadOnlyVector:
        r"""Return a vector of ones, optionally as a broadcast read-only array.
        
        Parameters
        ----------
        read_only: bool [default: False]
            Whether to return a read-only view.
        
        
        Returns
        -------
        np.ndarray | ReadOnlyVector
            Result described above.
        """
        read_only = self._validate_read_only_flag(read_only)
        if read_only:
            return np.broadcast_to(
                np.asarray(1, dtype=self.dtype), self.shape
            )
        return np.ones(shape=self.shape, dtype=self.dtype)

    def read_only(self, x: np.ndarray) -> ReadOnlyVector:
        r"""Return a non-writable NumPy view sharing ``x``'s storage.
        
        Parameters
        ----------
        x: np.ndarray
            X.
        
        
        Returns
        -------
        ReadOnlyVector
            Read-only vector view.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, vec_name="vector",
                add_info="read_only argument is not in the NumPyVectorSpace."
            ))
        view = x.view()
        view.flags.writeable = False
        return view

    def empty(self) -> Vector:
        r"""Return an uninitialized element of the space.
                
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        return np.empty(shape = self.shape,dtype=self.dtype)


    def is_mask(self, mask: object) -> bool:
        r"""Check whether the object is a valid mask for this space.
        
        Parameters
        ----------
        mask: object
            Coordinate-selection mask in the format used by the vector space.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return (isinstance(mask, np.ndarray) and mask.dtype == bool
                and mask.shape == self.shape)

    def iter_mask(self, mask: Mask) -> Iterator[bool]:
        r"""Iterate over the mask entries in coefficient order.
        
        Parameters
        ----------
        mask: Mask
            Coordinate-selection mask in the format used by the vector space.
        
        
        Returns
        -------
        Iterator[bool]
            Result described above.
        """
        if not self.is_mask(mask):
            raise ValueError("Mask must be boolean and have the vector shape.")
        return iter(mask.flat)

    def mask_from_iterable(self, values: Iterable[bool]) -> Mask:
        r"""Construct a backend mask from boolean entries.
        
        Parameters
        ----------
        values: Iterable[bool]
            Values.
        
        
        Returns
        -------
        Mask
            Result described above.
        """
        return super().mask_from_iterable(values).reshape(self.shape)

    def nrTrues(self, mask: Mask) -> int:
        r"""Count the selected coordinates in a mask.
        
        Parameters
        ----------
        mask: Mask
            Coordinate-selection mask in the format used by the vector space.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        if not self.is_mask(mask):
            raise ValueError("Invalid mask.")
        return int(np.count_nonzero(mask))

    def _logical(self, x: Mask, y: Mask, operation: Callable[[Any, Any], Any]) -> Mask:
        if not self.is_mask(x) or not self.is_mask(y):
            raise ValueError("Invalid mask.")
        return operation(x, y)

    def logical_and(self, x: Mask, y: Mask) -> Mask:
        r"""Compute the intersection of two coordinate masks.
        
        Parameters
        ----------
        x: Mask
            X.
        y: Mask
            Y.
        
        
        Returns
        -------
        Mask
            Result described above.
        """
        return self._logical(x, y, np.logical_and)

    def logical_or(self, x: Mask, y: Mask) -> Mask:
        r"""Compute the union of two coordinate masks.
        
        Parameters
        ----------
        x: Mask
            X.
        y: Mask
            Y.
        
        
        Returns
        -------
        Mask
            Result described above.
        """
        return self._logical(x, y, np.logical_or)

    def logical_xor(self, x: Mask, y: Mask) -> Mask:
        r"""Select coordinates present in exactly one of the masks.
        
        Parameters
        ----------
        x: Mask
            X.
        y: Mask
            Y.
        
        
        Returns
        -------
        Mask
            Result described above.
        """
        return self._logical(x, y, np.logical_xor)

    def logical_not(self, x: Mask) -> Mask:
        r"""Complement a coordinate mask.
        
        Parameters
        ----------
        x: Mask
            X.
        
        
        Returns
        -------
        Mask
            Result described above.
        """
        if not self.is_mask(x):
            raise ValueError("Invalid mask.")
        return np.logical_not(x)

    def _comparison(self, x: ReadOnlyVector, predicate: Callable[..., bool], *, ordered: bool=False) -> Mask:
        if x not in self:
            raise ValueError("Vector outside space.")
        if ordered and not self.is_ordered:
            raise TypeError("Complex spaces have no coordinate ordering.")
        return predicate(x)

    def _compare_pair(self, x: ReadOnlyVector, y: ReadOnlyVector, predicate: Callable[..., bool], *, ordered: bool=False) -> Mask:
        if x not in self or y not in self:
            raise ValueError("Vectors outside the space.")
        if ordered and not self.is_ordered:
            raise TypeError("Complex spaces have no coordinate ordering.")
        return predicate(x, y)

    def copyto(self, x: Vector, y: ReadOnlyVector, where: Mask | None=None) -> None:
        r"""Copy coefficients into an existing writable vector.
        
        Parameters
        ----------
        x: Vector
            Writable vector used by this operation.
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        where: Mask | None [default: None]
            Mask selecting the coordinates to write; None selects all coordinates.
        """
        if x not in self or y not in self:
            raise ValueError("Vector outside space.")
        if where is not None and not self.is_mask(where):
            raise ValueError("Invalid mask.")
        np.copyto(x, y, where=True if where is None else where)

    def fill(self, x: Vector, value: Scalar, where: Mask | None=None) -> None:
        r"""Fill selected coefficients of a writable vector with a scalar.
        
        Parameters
        ----------
        x: Vector
            Writable vector used by this operation.
        value: Scalar
            Value.
        where: Mask | None [default: None]
            Mask selecting the coordinates to write; None selects all coordinates.
        """
        if x not in self or not np.isscalar(value):
            raise ValueError("Expected a vector and scalar value.")
        if where is not None and not self.is_mask(where):
            raise ValueError("Invalid mask.")
        np.copyto(x, value, where=True if where is None else where)

    def restrict(self, x: ReadOnlyVector, mask: Mask) -> Vector:
        r"""Restrict.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        mask: Mask
            Coordinate-selection mask in the format used by the vector space.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        if x not in self or not self.is_mask(mask):
            raise ValueError("Invalid vector or mask.")
        return x[mask].copy()

    def extend(self, x: ReadOnlyVector, mask: Mask) -> Vector:
        r"""Extend.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        mask: Mask
            Coordinate-selection mask in the format used by the vector space.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        if not self.is_mask(mask):
            raise ValueError("Invalid mask.")
        if not isinstance(x, np.ndarray) or x.shape != (self.nrTrues(mask),):
            raise ValueError("Invalid restricted vector.")
        if not np.can_cast(x.dtype, np.dtype(self.dtype), casting="same_kind"):
            raise TypeError("Restricted vector has an incompatible dtype.")
        result = self.zeros()
        result[mask] = x
        return result

    def _iter_native_coefficients(self, x: ReadOnlyVector) -> Iterator[Scalar]:
        r"""Iterate over native NumPy coefficients in C order.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Iterator[Scalar]
            Result described above.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x,
                self,
                vec_name="vector",
                add_info="iter_coefficients argument is not in the NumPyVectorSpace."
            ))
        return iter(np.asarray(x).flat)

    def _set_native_coefficients(
        self,
        x: np.ndarray,
        coefficients: Iterable[Scalar],
    ) -> None:
        r"""Overwrite an array from native coefficients.
        
        Parameters
        ----------
        x: np.ndarray
            X.
        coefficients: Iterable[Scalar]
            Coefficients.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x,
                self,
                vec_name="vector",
                add_info="set_coefficients destination is not in the NumPyVectorSpace."
            ))
        if not x.flags.writeable:
            raise ValueError(Errors.value_error(
                "Cannot write coefficients into a read-only NumPy vector."
            ))
        iterator = iter(coefficients)
        for index in range(int(self._native_dimension)):
            try:
                coefficient = next(iterator)
            except StopIteration as error:
                raise ValueError(Errors.value_error(
                    f"Expected {self._native_dimension} coefficients, but received only "
                    f"{index}."
                )) from error
            if not self.is_complex and np.iscomplexobj(coefficient):
                coefficient = np.asarray(coefficient).item()
                if coefficient.imag != 0:
                    raise ValueError(Errors.value_error(
                        "A real NumPyVectorSpace cannot consume a non-real "
                        f"coefficient {coefficient!r}."
                    ))
                coefficient = coefficient.real
            x.flat[index] = coefficient
        try:
            next(iterator)
        except StopIteration:
            return
        raise ValueError(Errors.value_error(
            f"Expected {self._native_dimension} coefficients, but received more."
        ))

    def rand(self, distribution: str = "uniform", seed: int | None = None,**kwargs: Any) -> Vector:
        r"""Rand.
        
        Parameters
        ----------
        distribution: str [default: 'uniform']
            Distribution.
        seed: int | None [default: None]
            Seed.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        if seed is not None:
            set_rng_seed(seed)
        r = self._draw_sample(distribution=distribution,size = self.shape, **kwargs)
        if not np.can_cast(r.dtype, self.dtype):
            raise ValueError(Errors.value_error(
                'random generator with distribution {} can not produce values of dtype {}'.format(distribution, self.dtype)))
        if is_complex_dtype(np.array([],dtype=self.dtype)) and not is_complex_dtype(r.dtype):
            c = np.empty(self.shape, dtype=self.dtype)
            c.real = r
            c.imag = self._draw_sample(distribution=distribution, size=self.shape, **kwargs)
            return c
        else:
            return np.asarray(r, dtype=self.dtype)

    def poisson(self, x: ReadOnlyVector, seed: int | None=None, **kwargs: Any) -> Vector:
        r"""Sample Poisson counts in a space with real storage.
        
        Complex spaces are unsupported, including complex storage interpreted
        as a real vector space (as_real_space=True). Rates must be finite,
        nonnegative real vectors belonging to this space.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        seed: int | None [default: None]
            Seed.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        if self.is_complex:
            raise TypeError("poisson is available only for vector spaces with real storage.")
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, add_info="poisson rates must belong to the vector space."))
        rates = np.asarray(x)
        if np.any(np.imag(rates) != 0):
            raise ValueError("Poisson rates must be real-valued.")
        rates = np.real(rates)
        if np.any(~np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("Poisson rates must be finite and nonnegative.")
        return self.rand(distribution="poisson", lam=rates, seed=seed, **kwargs)
    
    def __contains__(self, x: ReadOnlyVector) -> bool:
        if not isinstance(x, np.ndarray) or x.shape != self.shape:
            return False
        elif is_complex_dtype(x.dtype):
            return self.is_complex
        elif is_real_dtype(x.dtype):
            return True
        else:
            return False
        

    def flatten(self, x: ReadOnlyVector) -> np.ndarray:
        r"""Flat coordinates; complex realification interleaves real and imag.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        if x not in self:
            raise ValueError("Vector outside this space.")
        x = np.asarray(x, dtype=self.dtype)
        if self.is_complex and self.as_real_space:
            return complex2real(x).ravel()
        return x.reshape(-1)


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
        x = np.asarray(x)
        if x.shape != (self.dimension,):
            raise ValueError("Expected dimension flat coordinates.")
        if self.as_real_space:
            if np.any(np.imag(x) != 0):
                raise ValueError("Real coordinates must be real.")
            x = x.real
        if self.is_complex and self.as_real_space:
            x = x[0::2] + 1j * x[1::2]
        return np.asarray(x, dtype=self.dtype).reshape(self.shape)



    def complex_space(self) -> VectorSpaceBase:
        r"""Compute the corresponding complex vector space.
        
        Returns
        -------
        VectorSpaceBase
            The complex space corresponding to this vector space as a shallow copy with modified
            dtype.
        """
        other = deepcopy(self)
        other.dtype = np.result_type(1j, self.dtype)
        other.is_complex = True
        return other

    def real_space(self) -> VectorSpaceBase:
        r"""Compute the corresponding real vector space.
        
        Returns
        -------
        VectorSpaceBase
            The real space corresponding to this vector space as a shallow copy with modified
            dtype.
        """
        other = deepcopy(self)
        other.dtype = np.empty(0, dtype=self.dtype).real.dtype
        other.is_complex = False
        other.as_real_space = True
        return other
    
    

    def iter_basis(self) -> Iterator[ReadOnlyVector]:
        r"""Generator iterating over the standard basis of the vector space. For efficiency,
        the same array is returned in each step, and subsequently modified in-place. If you need
        the array longer than that, perform a copy. In case of complex a vector space after each
        each array modified in its place with a real one it returns the same vector with \(1i\)
        in its place.   
        
        
        Yields
        ------
        ReadOnlyVector
            Result described above.
        """
        elm = self.zeros()
        for idx in np.ndindex(self.shape):
            elm[idx] = 1
            yield elm
            if self.is_complex and self.as_real_space:
                elm[idx] = 1j
                yield elm
            elm[idx] = 0


    def __eq__(self, other: object) -> bool | NotImplementedType:
        return (isinstance(other, type(self)) and self.shape == other.shape
                and self.dtype == other.dtype and self.as_real_space == other.as_real_space)

    def compatible(self, other: object) -> bool:
        r"""Compatible.
        
        Parameters
        ----------
        other: object
            Other.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return (isinstance(other, NumPyVectorSpace) and self.shape == other.shape
                and self.is_complex == other.is_complex
                and self.as_real_space == other.as_real_space)

    @property
    def ndim(self) -> int:
        r"""Ndim.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        return len(self.shape)

    def __repr__(self) -> str:
        if hasattr(self,"mask"):
            return make_repr(self,self.shape,self.is_complex,f"mask = {self.mask}")
        else: 
            return make_repr(self,self.shape,self.is_complex)

    def __mul__(self, other: Any) -> Prod:
        if isinstance(other, NumPyVectorSpace):
            return Prod(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> Prod:
        if isinstance(other, NumPyVectorSpace):
            return Prod(other, self)
        else:
            return NotImplemented


class MeasureSpaceFcts(NumPyVectorSpace):
    r"""Space of functions :math:`\mathbb{R}^N\to \mathbb{R}^M` or :math:`\mathbb{R}^N\to \mathbb{C}^M` (viewed as a real
    space) with an additional measure on `\mathbb{R}^N` given by `N` non-negative weights.
    Both `N` and `M` can be multi-dimensional, i.e., tuples of integers. By default, `M` is  `()` (scalar functions). 
    Either the measure or the shape have to be specified. The measure defaults to the constant 1 measure for each point if it is not given.
    
    Notes
    -----
    
    The measure is stored as a NumPy array of shape `shape + (1,)*len(shape_codomain)`, i.e., the measure is broadcasted
    
    Parameters
    ----------
    measure: np.ndarray | float | None [default: None]
        The non negative array representing the point measures. If it is not given the measures are set to 1 for each point. The shape of the measure has to be shape+(1,)*len(shape_codomain)
    shape: tuple[int, ...] | int | None [default: None]
        The shape of the domain of the functions (`N`). If it is not given the shape is taken from measure.
    shape_codomain: tuple[int, ...] | int [default: ()]
        The shape of the codomain of the functions (`M`).
    dtype: type [default: float]
        The elements' dtype. Should usually be either `float` or `complex`.
    as_real_space: bool [default: True]
        Whether to use real scalars. With complex storage, True treats the
        space as real and counts real and imaginary coordinates separately.
    """
    @overload
    def __init__(self,measure : None, shape : tuple[int, ...] | int, shape_codomain : tuple[int, ...] | int = (), dtype : type = float, as_real_space: bool = True) -> None: ...

    @overload
    def __init__(self,measure : np.ndarray, shape : None, shape_codomain : tuple[int, ...] | int = (), dtype : type = float, as_real_space: bool = True) -> None: ...

    def __init__(self,
                 measure : np.ndarray | float | None = None, 
                 shape : tuple[int, ...] | int | None = None, 
                 shape_codomain : tuple[int, ...] | int = (), 
                 dtype : type = float, as_real_space: bool = True) -> None:
        if(not isinstance(measure,np.ndarray) and shape is None):
            raise ValueError(Errors._compose_message("Invalid Init",'Either measure or shape have to be set to determine shape of space.'))
        if shape is None:
            shape=measure.shape
        if measure is None:
            measure=1
        if isinstance(shape,int):
            shape=(shape,)
        elif not isinstance(shape,tuple):
            raise ValueError(Errors._compose_message("Wrong Value",'The shape of a MeasureSpaceFcts has to be an int or a tuple of ints.'))
        if(isinstance(shape_codomain,int)):
            shape_codomain=(shape_codomain,)
        elif not isinstance(shape_codomain,tuple):
            raise ValueError(Errors._compose_message("Wrong Value",'The shape_codomain of a MeasureSpaceFcts has to be an int or a tuple of ints or an empty tuple.'))
        super().__init__(shape=shape + shape_codomain, dtype=dtype, as_real_space=as_real_space)
        self.shape_domain = shape
        r"""The shape of the domain of the functions (`N`)."""
        self.shape_codomain = shape_codomain
        r"""The shape of the codomain of the functions (`M`)."""
        self.measure=measure
        r"""The measure on the domain of the functions (`N`)."""

    def get_masked_space(self, mask) -> MaskedMeasureSpaceFcts:
        """Restrict coordinates while retaining the spatial integration weights.

        A masked space stores its selected coefficients as a flat vector.
        Accordingly, its measure contains one weight per selected coefficient;
        for vector-valued functions the spatial weights are repeated over the
        point-value components before applying the coordinate mask.
        """
        return MaskedMeasureSpaceFcts(self, mask)

    @property
    def ndim_domain(self) -> int:
        r"""The number of domain dimensions.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        return len(self.shape_domain)
    
    @property
    def ndim_codomain(self) -> int:
        r"""The number of codomain dimensions.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        return len(self.shape_codomain)

    def scalar_space(self) -> MeasureSpaceFcts:
        r"""Returns the corresponding scalar-valued function space.
        
        That is given a space of functions :math:`\mathbb{R}^N\to \mathbb{R}^M` or :math:`\mathbb{R}^N\to \mathbb{C}^M`,
        it returns the space of functions :math:`\mathbb{R}^N\to \mathbb{R}` or :math:`\mathbb{R}^N\to \mathbb{C}`.
        
        Returns
        -------
        MeasureSpaceFcts
            The scalar-valued function space corresponding to this vector space as a shallow copy with modified
            shape_codomain.
        """
        res = deepcopy(self)
        res.shape_codomain = ()
        res.shape = res.shape_domain
        res._coordinate_count = int(np.prod(res.shape))
        res.measure = np.squeeze(self.measure, axis = tuple(range(-self.ndim+self.ndim_domain,0)))
        try:
            del res.identity
        except AttributeError:
            pass  # No cached identity has been constructed yet.
        return res

    def vector_valued_space(self,shape_codomain: int | tuple[int, ...]) -> MeasureSpaceFcts:
        r"""Returns a corresponding vector-valued function space with given shape_codomain.
        
        That is given a space of functions :math:`\mathbb{R}^N\to \mathbb{R}` or :math:`\mathbb{R}^N\to \mathbb{C}`,
        it returns the space of functions :math:`\mathbb{R}^N\to \mathbb{R}^M` or :math:`\mathbb{R}^N\to \mathbb{C}^M`.
        
        Parameters
        ----------
        shape_codomain: int | tuple[int, ...]
            Shape codomain.
        
        Returns
        -------
        MeasureSpaceFcts
            The vector-valued function space corresponding to this vector space as a shallow copy with modified
            shape_codomain.
        """
        if isinstance(shape_codomain, int):
            shape_codomain = (shape_codomain,)
        elif not isinstance(shape_codomain, tuple) or not all(isinstance(s,int) for s in shape_codomain):
            raise ValueError(Errors.not_instance(shape_codomain,tuple,'The shape_codomain of a MeasureSpaceFcts has to be an int or a tuple of ints or an empty tuple.'))

        res = deepcopy(self)
        res.shape_codomain = shape_codomain
        res.shape = res.shape_domain + res.shape_codomain
        res._coordinate_count = int(np.prod(res.shape))
        res.measure = np.squeeze(self.measure, axis = tuple(range(-self.ndim+self.ndim_domain,0)))
        try:
            del res.identity
        except AttributeError:
            pass  # No cached identity has been constructed yet.
        return res

    @property
    def measure(self) -> float | np.ndarray:
        r"""Stores values of point measures 
        
        
        Returns
        -------
        float | np.ndarray
            Result described above.
        """
        return self._measure
    
    @measure.setter
    def measure(self,new_measure: Scalar | ArrayLike) -> None:
        r"""Measure.
        
        Parameters
        ----------
        new_measure: Scalar | ArrayLike
            New measure.
        """
        broadcasted_measure=self.update_measure(new_measure)
        self._measure=broadcasted_measure

    def update_measure(self,new_measure: Scalar | ArrayLike) -> None:
        r"""Update measure.
        
        Parameters
        ----------
        new_measure: Scalar | ArrayLike
            New measure.
        """
        try:
            broadcasted=np.broadcast_to(new_measure,self.shape_domain)
        except ValueError as e:
            raise ValueError(Errors._compose_message("Invalid Measure",f"The measure with shape {new_measure.shape} can not be broadcasted to the domain shape of the domain {self.shape_domain}. Note that the shape of the space is decomposed into shape_domain + shape_codomain given by {self.shape_domain} + {self.shape_codomain} and the measure has to be at broadcastable to the shape_domain!")) from e
        broadcasted = np.expand_dims(broadcasted, axis=tuple(range(len(self.shape_domain), len(self.shape))))
        if(not (np.issubdtype(broadcasted.dtype, np.floating) or np.issubdtype(broadcasted.dtype, np.integer))):
            raise ValueError(Errors._compose_message("Mismatch of dtype", f'Type {broadcasted.dtype} is invalid type for measure.'))
        if(broadcasted.size and np.min(broadcasted)<0):
            raise ValueError(Errors._compose_message("Not a Measure"),'Negative values are not allowed in measure.')
        return broadcasted
    
    def __eq__(self, other: object) -> bool | NotImplementedType:
        if(not super().__eq__(other)):
            return False
        return np.allclose(self.measure,other.measure) and self.shape_codomain==other.shape_codomain
    
    def __repr__(self) -> str:
        if hasattr(self,"mask"):
            return make_repr(self,self.shape,self.is_complex,f"mask = {self.mask}",f"measure= ({self.measure})",f"shape_codomain= {self.shape_codomain}")
        else: 
            return make_repr(self,self.shape,self.is_complex,f"measure= ({self.measure})",f"shape_codomain= {self.shape_codomain}")


class MaskedMeasureSpaceFcts(MaskedVectorSpace, MeasureSpaceFcts):
    r"""A masked coefficient space that retains a restricted point measure.

    Instances are both :class:`MaskedVectorSpace` and
    :class:`MeasureSpaceFcts`. They keep the original full space and immutable
    bitmap, so ``restrict`` and ``extend`` have their usual meanings.

    The masked representation is a flat, scalar-valued coefficient array:
    ``shape_domain == (mask.sum(),)`` and ``shape_codomain == ()``. Even when
    the full space is vector-valued, each selected point-value coefficient
    retains the measure of its spatial point. This supports arbitrary masks,
    including masks that select only some components at a spatial point.
    """

    def __init__(self, full_space: MeasureSpaceFcts, mask: np.ndarray) -> None:
        if not isinstance(full_space, MeasureSpaceFcts):
            raise TypeError("full_space must be a MeasureSpaceFcts.")
        super().__init__(full_space, mask)
        # MaskedVectorSpace sets the flat coefficient layout and preserves the
        # full_space/mask pair. Make that layout explicit to MeasureSpaceFcts.
        self.shape_domain = self.shape
        self.shape_codomain = ()
        self.dtype = np.dtype(full_space.dtype)
        weights = np.broadcast_to(full_space.measure, full_space.shape)
        self.measure = np.array(weights[self.mask], copy=True)

    def vector_valued_space(self, shape_codomain):
        """Reject reshaping masked coefficients without adapting the full mask.

        Construct the vector-valued *full* space first and then mask it.
        A generic coefficient mask need not select complete point-value blocks.
        """
        raise NotImplementedError(
            "Construct a vector-valued full space before calling get_masked_space; "
            "masked coefficients need not form complete point-value blocks."
        )

def is_uniform(x: ReadOnlyVector) -> bool:
    r"""Checks whether the neighboring entries of a one-dimensional vector all have the same distance
    
    Parameters
    ----------
    x: ReadOnlyVector
        Input vector; its coefficients are only read.
    
    
    Returns
    -------
    bool
        Whether the stated condition is satisfied.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(Errors.value_error("To determine if a vector is uniform, the vector need to be one dimensional. Given vector has dimension {x.ndim}"))
    if(x.shape[0]==1):
        return True
    diffs = x[1:] - x[:-1]
    return np.allclose(diffs, diffs[0])


class GridFcts(MeasureSpaceFcts):
    r"""A vector space representing (possibly vector-valued) functions defined on a rectangular grid.
    
    Parameters
    ----------
    axisdata: None | tuple[np.ndarray] | list [default: None]
        If the axes represent indices into some auxiliary arrays, these can be passed via this
        parameter. If given, there must be one array for each dimension, the size of the first axis
        of which must match the respective dimension's length. Besides that, no further structure
        is imposed or assumed, this parameter exists solely to keep everything related to the
        vector space in one place.
    shape_codomain: int | tuple[int, ...] [default: ()]
        The shape of the codomain of the functions.
    dtype: type [default: float]
        The dtype of the vector space.
    use_cell_measure: bool [default: True]
        If true a measure is calculated using the volume of the grid cells. Else the measure is one for all cells.
    boundary_ext: str [default: 'sym']
        Defines how the measure is continued at the boundary. Possible modes are
        'sym' : The boundary coordinates are assumed to be in the center of their cell
        'const': The boundary cells are extended by a constant given in boundary_ext_const
        'zero': The boundary coordinates are assumed to be on the outer edge of their cell
        defaults to 'sym'
    ext_const: None | float | tuple[float] [default: None]
        Defines extension of cells at edges of each axis. Can be set to a constant for all axes, one constant for each axis
        or one constant for the start and one for the end of each axis. Is only used in combination with `boundary_ext='const'`
        in which case it needs to be defined.
    as_real_space: bool [default: True]
        Whether to use real scalars. With complex storage, True treats the
        space as real and counts real and imaginary coordinates separately.
    *coords: ArrayLike
        Coords.
    
    Notes
    -----
    If `axisdata` is given, the `coords` can be omitted.
    """

    def __init__(self, *coords: ArrayLike, 
                 axisdata : None | tuple[np.ndarray] | list = None, 
                 shape_codomain : int | tuple[int, ...] = (), 
                 dtype : type = float,
                 use_cell_measure : bool =True,
                 boundary_ext : str = 'sym',
                 ext_const : None | float | tuple[float] = None, as_real_space: bool = True) -> None:
        axes = []
        extents=[]
        if axisdata and not coords:
            coords = [d.shape[0] for d in axisdata]

        for n, c in enumerate(coords):
            if isinstance(c, int):
                v = np.arange(c)
            elif isinstance(c, tuple):
                if len(c) != 3 or any([not isinstance(c_i,(int,float)) for c_i in c[:2]]) or not isinstance(c[2],int):
                    raise ValueError(Errors.value_error(f"If giving coords a tuples, the tuples must be 2 real numbers and an integer to construct a np.linspace! You gave:"+"\n\t "+f"c = {c}"))
                v = np.linspace(*c)
            else:
                v = np.asarray(c).view()
                if is_complex_dtype(v):
                    raise ValueError(Errors.value_error(f"The given explicit coords are not real! You gave "+"\n\t "+f"c = {c}"))
            extents.append(abs(v[-1] - v[0]))
            v.flags.writeable = False
            axes.append(v)
        self.coords=np.meshgrid(*axes,indexing='ij',copy=False)
        r"""The coordinate arrays, broadcast to the shape of the grid. The shape will be
        `(len(self.shape),) + self.shape`. If the one specifies a codomain shape, the shape will be 
        `(len(self.shape_domain),) + self.shape_domain`. Disregarding the codomain shape."""
        self.axes = axes
        """The axes as 1d arrays"""
        self.extents = np.asarray(extents)
        r"""The lengths of the axes, i.e. `axis[-1] - axis[0]`, for each axis."""

        if(use_cell_measure):
            super().__init__(GridFcts._calc_cell_measure(self.axes,boundary_ext,ext_const),
                             shape=self.coords[0].shape,
                             shape_codomain=shape_codomain, 
                             dtype=dtype, as_real_space=as_real_space
                             )
        else:
            super().__init__(shape=self.coords[0].shape, 
                             shape_codomain=shape_codomain, 
                             dtype=dtype, as_real_space=as_real_space
                             )

        if axisdata is not None:
            axisdata = tuple(axisdata)
            if len(axisdata) != len(coords) and any(self.shape_domain[i] != ax.shape[0] for i,ax in enumerate(axisdata)):
                raise ValueError(
                    Errors._compose_message(
                        "Invalid axisdata",
                        "If axisdata is given, they must be one dimensional arrays matching the size of the respective domains length in that dimension.",
                    )
                )
        self.axisdata = axisdata
        """The axisdata, if given."""

    def _calc_cell_measure(axes: int | tuple[int, ...],boundary_ext: str,ext_const: float | tuple[Any, ...] | None=None) -> np.ndarray:
        if len(axes)> 26:
            raise ValueError(Errors.value_error(f"Computing the cell measure is only supported for less than 26 axes. You have {len(axes)} axes."))
        ext_axes=[]
        if(boundary_ext=="sym"):
            ext_axes=[np.pad(v,(1,1),mode='reflect',reflect_type='odd') for v in axes]
        elif(boundary_ext=="zero"):
            ext_axes=[np.pad(v,(1,1),mode='edge') for v in axes]           
        elif(boundary_ext=="const"):
            if ext_const is None:
                raise ValueError(Errors.value_error("When computing cell measure with constant boundary the constant have to be defined! Either a scalar or tuple of length of axes!"))
            if(np.isscalar(ext_const)):
                ext_const=len(axes)*(ext_const,)
            elif not isinstance(ext_const, tuple) or len(ext_const)!=len(axes):
                raise ValueError(Errors.value_error("When computing cell measure with constant boundary the constants have to be either a scalar or tuple of length of axes! Gave: "+"\n\t "+f"{ext_const}"))
            for i, v in enumerate(axes):
                if isinstance(ext_const[i],tuple) and len(ext_const[i]) == 2 and np.isscalar(ext_const[i][0]) and np.isscalar(ext_const[i][1]):
                    ext_axes.append(np.pad(v,(1,1),mode='constant',constant_values=(v[0]-ext_const[i][0], v[-1]+ext_const[i][1])))
                elif np.isscalar(ext_const[i]):
                    ext_axes.append(np.pad(v,(1,1),mode='constant',constant_values=(v[0]-ext_const[i], v[-1]+ext_const[i])))
                else:
                    raise ValueError(Errors.value_error(f"The extending constants need to be a tuple of a tuple (left_bnd_val,right_bnd_val) for left and right boundary values or a scalar both_bnd_val for both sides. You defined the {i}-th value by ext_const = {ext_const[i]}"))
        ax_widths=[0.5*(ext_v[2:]-ext_v[:-2]) for ext_v in ext_axes]
        ax_widths=[np.array([1.]) if len(aw)==1 and aw[0]==0. else aw for aw in ax_widths]#avoid all zero weights if there are axes of length 1
        ax_widths=[np.array([aw[0]]) if(np.allclose(aw[0],aw)) else aw for aw in ax_widths]#collapse constant width axis
        prod_string=','.join([chr(k) for k in range(ord('A'),ord('A')+len(axes))])
        return np.einsum(prod_string,*ax_widths)#computes product of entries from ax_widths
    
    def coord_distances(self,point: ReadOnlyVector | None=None,axes: int | tuple[int, ...] | None=None) -> list[np.ndarray]:
        r"""Computes the euclidean distances of all grid points to the given point. If the point is None, then the distance from the origin is returned.
        
        Parameters
        ----------
        point: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        axes: int | tuple[int, ...] | None [default: None]
            Axes.
        
        Returns
        -------
            np.ndarray: NumPy array with same shape as domain containing all the distances.
        """
        if point is None:
            point=np.zeros(self.ndim)
        if axes is None:
            axes=tuple(j for j in range(self.ndim))
        if(min(axes)<0 or max(axes)>=self.ndim):
            raise ValueError(Errors.value_error(f"Axes {axes} out of bounds for grid with {self.ndim} axes."))
        if(not isinstance(point,np.ndarray) or  point.shape[0]!=self.ndim or point.ndim!=1):
            raise ValueError(Errors.value_error(f"Point {point} not a NumPy array or not compatible with domain with dimension {self.ndim}."))
        res=self.zeros()
        for j in axes:
            res+=(self.coords[j]-point[j])**2
        return np.sqrt(res)    

            

class UniformGridFcts(GridFcts):
    r"""A vector space representing (possibly vector-valued) functions defined on a rectangular grid with equidistant axes.
    The measure is constant. Use `GridFcts` for grids with uniform axes and non-constant measures.
    
    All arguments are passed to the `GridFcts` constructor, but an error will be produced if any axis
    is not uniform.
    
    Parameters
    ----------
    axisdata: None | tuple[np.ndarray] [default: None]
        If the axes represent indices into some auxiliary arrays, these can be passed via this
        parameter. If given, there must be one array for each dimension, the size of the first axis
        of which must match the respective dimension's length. Besides that, no further structure
        is imposed or assumed, this parameter exists solely to keep everything related to the
        vector space in one place.
        
        If `axisdata` is given, the `coords` can be omitted.
    shape_codomain: int | tuple[int, ...] [default: ()]
        Shape codomain.
    dtype: type [default: float]
        The dtype of the vector space.
    periodic: bool [default: False]
        passed as arguments to numpy.linspace, the right boundaries (second elements of the triples)
        are reduced such that the difference of the second and first elements represents 
        periodicity lengths.
    as_real_space: bool [default: True]
        Whether to use real scalars. With complex storage, True treats the
        space as real and counts real and imaginary coordinates separately.
    *coords: int | tuple[float, float, int] | ArrayLike
        Coords.
    """

    def __init__(self, *coords: int | tuple[float, float, int] | ArrayLike, 
                 axisdata : None | tuple[np.ndarray] = None, 
                 shape_codomain : int | tuple[int, ...] = (),  
                 dtype : type =float, 
                 periodic : bool = False, as_real_space: bool = True) -> None:
        if periodic and all(isinstance(c,tuple) for c in coords):
            coords = tuple((l, (l+(n-1)*r)/n ,n) for (l,r,n) in coords)
        super().__init__(*coords, 
                         axisdata=axisdata,
                         shape_codomain=shape_codomain,
                         dtype=dtype,
                         use_cell_measure=False, as_real_space=as_real_space
                        )
        spacing = []
        for axis in self.axes:
            if not is_uniform(axis):
                raise ValueError(Errors.value_error("One of the axis is failed the uniformity test!\n\t "+f"axis = {axis} "))
            if(axis.shape[0]==1):
                spacing.append(1.0)
            else:
                spacing.append(axis[1] - axis[0])
        self.spacing = np.asarray(spacing)
        """The spacing along every axis, i.e. `axis[i+1] - axis[i]`"""
        self.volume_elem = np.prod(self.spacing)
        """The volume element, initialized as product of `spacing`"""
        self.measure = self.volume_elem
        """ Setting measure to be initialized by `volume_element`"""
    
    def update_measure(self, new_measure: Scalar | ArrayLike) -> None:
        r"""Update measure.
        
        Parameters
        ----------
        new_measure: Scalar | ArrayLike
            New measure.
        """
        broadcasted_measure=super().update_measure(new_measure)
        self.volume_elem=broadcasted_measure.flat[0]
        return broadcasted_measure



class Prod(NumPyVectorSpace):
    r"""The tensor product of an arbitrary number of vector spaces.
    
    Elements of the tensor product will always be arrays with in n-dim where n is number of factors. 
    Representing each coefficient to a basis tensor that are mad up be the tensor product of each 
    basis element from the factored spaces. Note, that spaces with possible multidimensional elements
    (e.g. `UniformGridFcts` with multiple dimensions) get flatted. 
    
    Prod instances can be indexed and iterated over, returning / yielding the component vector spaces.
    
    Parameters
    ----------
    flatten: bool [default: False]
        Whether factors that are themselves `Prod`\s should be merged into this instance. If False, Prod is not associative, but the product method behaves more predictably.
        Default: False
    *factors: NumPyVectorSpace
        The vector spaces to be factored.
    """

    def __init__(self, *factors: NumPyVectorSpace, 
                 flatten : bool = False) -> None:
        if any(not isinstance(s, VectorSpaceBase) for s in factors):
            raise TypeError(Errors.type_error("One of spaces is to factor is not a VectorSpace!"))
        if any(s.is_complex for s in factors) and any(not s.is_complex for s in factors):
            raise TypeError(Errors.type_error("There are both complex and non-complex Factors. This is not supported in Prod!"))
        if len({s.as_real_space for s in factors}) > 1:
            raise ValueError("Tensor factors must have the same scalar field.")
        self.factors = []
        """List of the `vector spaces` to be taken as Product."""
        shape = ()
        self.volume_elem = 1.
        """Product of the `volume_elem` of all factors that have defined this property. """
        if factors[0].is_complex:
            dt=np.complex128
        else:
            dt=np.float64
        for s in factors:
            if hasattr(s, 'volume_elem'):
                self.volume_elem *= s.volume_elem
            if flatten and isinstance(s, type(self)):
                self.factors.extend(s.factors)
                shape += s.shape
            else:
                self.factors.append(s)
                # Tensor storage counts native entries, not realified coordinates.
                shape += (int(np.prod(s.shape)),)
        characters=tuple(chr(k) for k in range(ord('A'),ord('A')+len(self.factors)))
        self._prod_trafo_string=f"{','.join(characters)}->{''.join(characters)}"
        """String to compute the outer product in einsum."""
        super().__init__(shape, dtype=dt, as_real_space=factors[0].as_real_space)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        return (
            isinstance(other, type(self)) and
            len(self.factors) == len(other.factors) and
            all(s == t for s, t in zip(self.factors, other.factors))
        )

    def product(self, *xs: np.ndarray) -> np.ndarray:
        r"""Transform a collection of elements of the factors into an element of the tensor product by an outer product.
        
        Parameters
        ----------
        *xs: np.ndarray
            The elements of the factors. The number should match the number of factors,
            and for all `i`, `xs[i]` should be an element of `self[i]`.
        
        Returns
        -------
        np.ndarray
            An element of the tensor product
        """
        if any(x not in s for s, x in zip(self.factors, xs)):
            raise ValueError(Errors.value_error("One of the vectors to be taken in the outer product is not in the corresponding factor."))
        return np.einsum(self._prod_trafo_string,*[x.flat for x in xs],optimize=True)


    def __getitem__(self, item: int) -> NumPyVectorSpace:
        return self.factors[item]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.factors)

    def __len__(self) -> int:
        return len(self.factors)
