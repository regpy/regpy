"""VectorSpaceBases on which operators are defined.

The classes in this module implement various vector spaces on which the
`regpy.operators.Operator` implementations are defined. The base class is `VectorSpaceBase`,
which represents plain numpy arrays of some shape and dtype. So far it is assumed that 
vectors are always represented by numpy arrays. 

VectorSpaceBases serve the following main purposes:

- Derived classes can contain additional data like grid coordinates, bundling metadata in one
place instead of having every operator generate linspaces / basis functions / whatever on their
own.

- Providing methods for generating elements of the proper shape and dtype, like zero arrays,
random arrays or iterators over a basis.

- Checking whether a given array is an element of the vector space. This is used for
consistency checks, e.g. when evaluating operators. The check is only based on shape and dtype,
elements do not need to carry additional structure. Real arrays are considered as elements of
complex vector spaces.

- Checking whether two vector spaces are considered equal. This is used in consistency checks
e.g. for operator compositions.

All vector spaces are considered as real vector spaces, even if the dtype is complex. This
affects iteration over a basis as well as functions returning the dimension or flattening arrays.
"""

from copy import copy,deepcopy
import numpy as np
from math import sqrt

from dataclasses import dataclass
from typing import List

from regpy import util, operators

class VectorStructureBase:

    log = util.classlogger

    def __init__(self,type,shape,complex = False) -> None:
        self.type = type
        """The underlying type for the vectors.
        """
        self.shape = shape
        """A tuple containing the shape of the vectors
        """
        self.is_complex = complex
        """The data type of the entries in the vectors.
        """

    @property
    def size(self):
        """The size of elements (as arrays) of this vector space."""
        return np.prod(self.shape)
    
    @property
    def real_size(self) -> int:
        """The size of elements (as arrays) of this vector space when casting complex to real."""
        if self.is_complex:
            return int(np.prod(self.shape)*2)
        else:
            return int(np.prod(self.shape))
    
    def zeros(self):
        """Should generate a zero vector.

        Raises
        ------
        NotImplemented
            If subclass does not provide this method.
        """
        raise NotImplementedError
    
    def ones(self):
        """Should generate a vector corresponding to the constant 1 function.

        Raises
        ------
        NotImplemented
            If subclass does not provide this method.
        """
        raise NotImplementedError

    
    def empty(self):
        """Provides an empty vector.

        Raises
        ------
        NotImplemented
            If subclass does not provide this method.
        """
        raise NotImplementedError
    
    def rand(self,random_generator = None):
        """Generate a random vector.

        Parameters
        ----------
        random_generator : callable, optional
            The random function to use. Should accept the shape as a tuple and return a real
            array of that shape. Numpy functions like `numpy.random.standard_normal` conform to
            this. Default: None.

        Raises
        ------
        NotImplemented
            If subclass does not provide this method.
        """
        raise NotImplementedError
    
    def rand(self,random_generator = None):
        """Generate a poisson vector.

        Parameters
        ----------
        x : self.type
            the distribution to be used.

        Raises
        ------
        NotImplemented
            If subclass does not provide this method.
        """
        raise NotImplementedError

    def is_vector(self,x):
        """Determine if provided vector x is a Vector of this type.

        Parameters
        ----------
        x : any
            The vector to be checked if its a vector of this type.

        Returns
        -------
        Boolean
            Returns True if the vector belongs to this class of vectors.

        Raises
        ------
        NotImplemented
            If subclass does not provide this method.
        """
        raise NotImplementedError

    def to_complex(self):
        """Returns the same vector class with complex dtype.

        Returns
        -------
        VectorStructureBase
            Same vectors with complex dtype. 
        
        Raises
        ------
        NotImplemented
            If subclass does not provide this method.
        """
        raise NotImplementedError

    def to_real(self):
        """Returns the same vector class with real dtype.

        Returns
        -------
        VectorStructureBase
            Same vectors with real dtype. 
        
        Raises
        ------
        NotImplemented
            If subclass does not provide this method.
        """
        raise NotImplementedError
    
    def flatten(self,x):
        """Transform the vector `x`, into a flattened array. Inverse
        to `fromflat`.

        Parameters
        ----------
        x : self.type
            The vector to transform.

        Returns
        -------
        array
            The flattened array. If memory layout allows, it will be a view into `x`.
        """
        raise NotImplementedError
    
    def fromflat(self,x):
        """Transform a flattened vector into an element of the vector space. Inverse to `flatten`.

        Parameters
        ----------
        x : self.type
            The flat vector to transform

        Returns
        -------
        array
            The reshaped array. If memory layout allows, this will be a view into `x`.
        """
        raise NotImplementedError
    
    def vdot(self,x,y):
        """Return the vector dot product as defined for these vectors. Note
        for complex vector it is supposed, that the second vector is conjugated.

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

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError
    
    def norm(self,x):
        return sqrt(self.vdot(x,x).real)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return (self.type == other.type and
                self.shape == other.shape and
                self.is_complex == other.is_complex)

    def __add__(self, other):
        if isinstance(other, VectorStructureBase):
            return VectorSum(self, other,flatten=True)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, VectorStructureBase,flatten=True):
            return VectorSum(other, self)
        else:
            return NotImplemented
        
    # def __mul__(self, other):
    #     if isinstance(other, VectorStructureBase):
    #         return VectorProd(self, other)
    #     else:
    #         return NotImplemented

    # def __rmul__(self, other):
    #     if isinstance(other, VectorStructureBase):
    #         return VectorProd(other, self)
    #     else:
    #         return NotImplemented

    def __pow__(self, power):
        assert isinstance(power, int)
        domain = self
        for i in range(power-1):
            domain = VectorSum(domain, self,flatten=True)
        return domain
    
    def __iter__(self):
        return self.iter_basis()

    def iter_basis(self):
        raise NotImplementedError

    @staticmethod
    def logical_and(x,y):
        return x & y 
    
    @staticmethod
    def logical_or(x,y):
        return x | y 
    
    def logical_not(self,x):
        return not x
    
    def logical_xor(self,x,y):
        return x ^ y

@dataclass 
class TupleVector:
    v: List

    def copy(self):
        return copy(self)

    def __post_init__(self):
        self.v = list(self.v)
        self.types = [type(v_i) for v_i in self.v]
        self.ndim = len(self.types)

    def conj(self):
        return TupleVector([v_i.conj() for v_i in self])
    
    @property
    def real(self):
        return TupleVector([v_i.real for v_i in self])
    
    def __len__(self):
        return self.ndim
    
    @property
    def real(self):
        return TupleVector([v_i.imag for v_i in self])

    def __eq__(self,other):
        if isinstance(other,type(self)) and self.ndim == other.ndim:
            return TupleVector([s_i == o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i == other for s_i in self.v])
    
    def __lt__(self, other):
        if isinstance(other,type(self)) and self.ndim == other.ndim:
            return TupleVector([s_i < o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i < other for s_i in self.v])

    def __le__(self, other):
        if isinstance(other,type(self)) and self.ndim == other.ndim:
            return TupleVector([s_i <= o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i <= other for s_i in self.v])

    def __ne__(self, other):
        return not (self == other)
    
    def __ge__(self, other):
        if isinstance(other,type(self)) and self.ndim == other.ndim:
            return TupleVector([s_i >= o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i >= other for s_i in self.v])
    
    def __gt__(self, other):
        if isinstance(other,type(self)) and self.ndim == other.ndim:
            return TupleVector([s_i > o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i > other for s_i in self.v])

    def all(self):
        return all((v_i.all() for v_i in self.v))

    def any(self):
        return any((v_i.any() for v_i in self.v))
    
    def __or__(self,other):
        assert isinstance(other,type(self)) and self.ndim == other.ndim 
        return TupleVector([s_i | o_i for s_i,o_i in zip(self.v,other.v)])
    
    def sum(self):
        z = []
        for v_i in self.v:
            if isinstance(v_i,np.ndarray):
                s = np.sum(v_i)
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
    
    def __iadd__(self,other):
        assert isinstance(other,TupleVector) and other.ndim == self.ndim and all([t_o==t_s for t_o,t_s in zip(other.types,self.types)])
        v = self.v
        for k,v_o in enumerate(other.v):
            v[k] += v_o
        return TupleVector(v)

    def __isub__(self,other):
        assert isinstance(other,TupleVector) and other.ndim == self.ndim and all([t_o==t_s for t_o,t_s in zip(other.types,self.types)])
        v = self.v
        for k,v_o in enumerate(other.v):
            v[k] -= v_o
        return TupleVector(v)
    
    def __add__(self,other):
        assert isinstance(other,TupleVector) and other.ndim == self.ndim and all([t_o==t_s for t_o,t_s in zip(other.types,self.types)])
        return TupleVector([s_k + o_k for s_k,o_k in zip(self,other)])
    
    def __radd__(self,other):
        return self + other
    
    def __sub__(self,other):
        return self + (-1*other)
    
    def __rsub__(self,other):
        return (-1*self) + other
    
    def __imul__(self,other):
        assert isinstance(other,float) or isinstance(other,int)
        v = self.v
        for k in range(self.ndim):
            v[k] *= other
        return TupleVector(v)
    
    def __itruediv__(self,other):
        assert isinstance(other,float) or isinstance(other,int)
        v = self.v
        for k in range(self.ndim):
            v[k] /= other
        return TupleVector(v)
    
    def __mul__(self,other):
        assert isinstance(other,float) or isinstance(other,int)
        return TupleVector([other * s_k for s_k in self])
        
    def __rmul__(self,other):
        return self * other

    def __truediv__(self,other):
        assert isinstance(other,float) or isinstance(other,int)
        return TupleVector([s_k / other for s_k in self])

    def __iter__(self):
        return iter(self.v)
    
    def __getitem__(self, key):
        if isinstance(key,slice) or isinstance(key,int):
            return self.v[key]
        elif (isinstance(key,TupleVector) and self.ndim == key.ndim):
            return TupleVector([v_i[k_i] for v_i,k_i in zip(self,key)])
        elif (isinstance(key,list) and len(key) == self.ndim): 
            return TupleVector([v_i[k_i] for v_i,k_i in zip(self,key)])
        else:
            raise KeyError("keys of type {} are not supported either int or list of length {}".format(type(key),self.ndim))
    
    def __setitem__(self, key, item):
        if isinstance(key,slice) or isinstance(key,int):
            self.v[key] = item
        elif (isinstance(key,list) and len(key) == self.ndim):
            if (isinstance(item,list) and len(item) == self.ndim):
                for k_i,item_i in zip(key,item):
                    self.v[k_i] = item_i
            elif np.isscalar(item):
                for k_i in key:
                    self.v[k_i] = item
            else:
                raise TypeError("items has to be a list of length {} not {} type".format(self.ndim,type(item)))                
        elif (isinstance(key,TupleVector) and self.ndim == key.ndim): 
            if (isinstance(item,TupleVector) and item.ndim == key.ndim):
                for v_i,k_i,item_i in zip(self.v,key,item):
                    v_i[k_i] = item_i
            elif np.isscalar(item):
                for v_i,k_i in zip(self.v,key):
                    v_i[k_i] = item
            else:
                raise TypeError("items has to be a TupleVector if key is TupleVector of ndim {} not {} type".format(self.ndim,type(item)))                
        else:
            raise KeyError("keys of type {} are not supported either int or list of length {} or TupleVector".format(type(key),self.ndim))
    
    def __copy__(self):
        return deepcopy(self)
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    
    def component_wise(self,method):
        assert callable(method)
        return TupleVector([method(s_k) for s_k in self])


class VectorSum(VectorStructureBase):

    log = util.classlogger

    def __init__(self, *summands : type[VectorStructureBase], flatten: bool = False) -> None:
        if flatten:
            self.summands = []
            for s in summands:
                if isinstance(s,VectorSum):
                    self.summands.extend(s.summands)
                else:
                    self.summands.append(s)
        else:
            self.summands = summands
        self.n_components = len(self.summands)
        super().__init__(TupleVector, (sum([s.size for s in summands]),), complex = any([s.is_complex for s in summands]))

    @property
    def real_size(self) -> int:
        return sum([s.real_size for s in self.summands]) 

    def zeros(self) -> TupleVector:
        return TupleVector([s.zeros() for s in self.summands])
    
    def ones(self)-> TupleVector:
        return TupleVector([s.ones() for s in self.summands])
    
    def empty(self)-> TupleVector:
        return TupleVector([s.empty() for s in self.summands])
    
    def rand(self,random_generator = None)-> TupleVector:
        return TupleVector([s.rand(random_generator=random_generator) for s in self.summands])
    
    def poisson(self,x)-> TupleVector:
        return TupleVector([s.poisson(x_k) for x_k,s in zip(x,self.summands)])

    def is_vector(self,x) -> bool:
        if isinstance(x,TupleVector) and x.ndim == self.n_components and all([isinstance(x_i,t_i.type) for x_i,t_i in zip(x,self.summands)]):
            return True
        else:
            print(self.summands)
            print(x.ndim, self.n_components)
            print([isinstance(x_i,t_i.type) for x_i,t_i in zip(x,self.summands)])
            return False
        
    def vdot(self, x : TupleVector, y : TupleVector) -> float | complex:
        assert self.is_vector(x), "x of type {} is not a vector".format(type(x)) 
        assert self.is_vector(y), "y of type {} is not a vector".format(type(y))
        return sum([s_i.vdot(x_i, y_i) for x_i,y_i,s_i in zip(x,y,self.summands) ])

    def to_complex(self):
        return VectorSum(*[s.to_complex for s in self.summands])

    def to_real(self):
        return VectorSum(*[s.to_real for s in self.summands])
    
    def flatten(self, x : TupleVector) -> np.ndarray:
        assert self.is_vector(x)
        return np.asarray([s.flatten(x_i) for x_i,s in zip(x.v,self.summands)])
    
    def fromflat(self, x : np.ndarray) -> TupleVector:
        if x.ndim == 1 and np.isreal(x) and x.size == self.real_size:
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
            raise ValueError("x has to be of type np.ndarray not {}".format(type(x)))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return all([s == o for s,o in zip(self.summands,other.summands)])
    
    def iter_basis(self):
        for i,s in enumerate(self.summands()):
            vec = self.zeros()
            for b in s.iter_basis():
                vec.v[i] = b
                yield vec
    
    def logical_not(self,x) -> TupleVector:
        return TupleVector([s.logical_not(x_i) for x_i,s in zip(x.v,self.summands)])
    
    def logical_xor(self,x,y) -> TupleVector:
        return TupleVector([s.logical_xor(x_i) for x_i,s in zip(x.v,self.summands)])


class NumPyVector(VectorStructureBase):

    log = util.classlogger

    def __init__(self, shape, dtype=float):
        # Upcast dtype to represent at least (single-precision) floats, no
        # bools or ints
        dtype = np.result_type(np.float32, dtype)
        # Allow only float and complexfloat, disallow objects, strings, times
        # or other fancy dtypes
        assert np.issubdtype(dtype, np.inexact)
        self.dtype = dtype
        try:
            shape = tuple(shape)
        except TypeError:
            shape = (shape,)
        super().__init__(np.ndarray, shape, util.is_complex_dtype(self.dtype))

    def zeros(self):
        return np.zeros(self.shape, dtype=self.dtype)
    
    def ones(self):
        return np.ones(self.shape, dtype=self.dtype)

    def empty(self):
        return np.empty(self.shape, dtype=self.dtype)
    
    def rand(self,random_generator = None):
        random_generator = random_generator or np.random.random_sample 
        r = random_generator(self.shape)
        if not np.can_cast(r.dtype, self.dtype):
            raise ValueError(
                'random generator {} can not produce values of dtype {}'.format(random_generator, self.dtype))
        if util.is_complex_dtype(self.dtype) and not util.is_complex_dtype(r.dtype):
            c = np.empty(self.shape, dtype=self.dtype)
            c.real = r
            c.imag = random_generator(self.shape)
            return c
        else:
            return np.asarray(r, dtype=self.dtype)
    
    def poisson(self,x):
        assert not self.is_complex
        return np.random.poisson(x)

    def is_vector(self,x):
        if not isinstance(x,np.ndarray):
            return False
        elif x.shape != self.shape:
            return False
        elif util.is_complex_dtype(x.dtype):
            return self.is_complex
        elif util.is_real_dtype(x.dtype):
            return True
        else:
            return False
        
    def vdot(self, x, y):
        return np.vdot(x, y).item()

    def to_complex(self):
        return NumPyVector(self.shape,np.result_type(1j, self.dtype))

    def to_real(self):
        return NumPyVector(self.shape,np.empty(0, dtype=self.dtype).real.dtype)
    
    def flatten(self, x : np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        assert self.shape == x.shape
        if self.is_complex:
            if util.is_complex_dtype(x.dtype):
                return util.complex2real(x).ravel()
            else:
                aux = self.empty()
                aux.real = x
                return util.complex2real(aux).ravel()
        elif util.is_complex_dtype(x.dtype):
            raise TypeError('Real vector space can not handle complex vectors')
        return x.ravel()

    def fromflat(self, x : np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        assert util.is_real_dtype(x.dtype)
        if self.is_complex:
            return util.real2complex(x.reshape(self.shape + (2,)))
        else:
            return x.reshape(self.shape)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other,type(self)) and self.dtype != other.dtype:
            return False
        return super().__eq__(other)

    def iter_basis(self):
        r"""Generator iterating over the standard basis of the vector space. For efficiency,
        the same array is returned in each step, and subsequently modified in-place. If you need
        the array longer than that, perform a copy. In case of complex a vector space after each
        each array modefied in its place with a real one it returns the same vector with \(1i\)
        in its place.   
        """
        elm = self.zeros()
        for idx in np.ndindex(self.shape):
            elm[idx] = 1
            yield elm
            if self.is_complex:
                elm[idx] = 1j
                yield elm
            elm[idx] = 0
    
    @staticmethod
    def logical_and(x,y):
        return np.logical_and(x, y) 
    
    @staticmethod
    def logical_or(x,y):
        return np.logical_or(x, y) 
    
    def logical_not(self,x):
        return np.logical_not(x) 
    
    def logical_xor(self,x,y):
        return np.logical_xor(x, y) 


class VectorSpaceBase:
    r"""Discrete space \(\mathbb{R}^\text{shape}\) or \(\mathbb{C}^\text{shape}\) (viewed as a real
    space) without any additional structure.

    VectorSpaceBases can be added, producing `DirectSum` instances.

    Parameters
    ----------
    vector_type : VectorStructureBase
        The class of vectors used.
    """

    log = util.classlogger

    def __init__(self, vector_type : VectorStructureBase):
        self.vec_type = vector_type
        """The vector type"""
        self.shape = self.vec_type.shape
        """The vector space's shape"""

    def zeros(self):
        """Return the zero vector of the space.
        """
        return self.vec_type.zeros()
    
    def ones(self):
        """Return the zero vector of the space.
        """
        return self.vec_type.ones()

    def empty(self):
        """Return an uninitalized element of the space.
        """
        return self.vec_type.empty()

    def iter_basis(self):
        r"""Generator iterating over the standard basis of the vector space. For efficiency,
        the same array is returned in each step, and subsequently modified in-place. If you need
        the array longer than that, perform a copy. In case of complex a vector space after each
        each array modefied in its place with a real one it returns the same vector with \(1i\)
        in its place.   
        """
        for idx in self.vec_type:
            yield idx

    def rand(self, rand=None):
        """Return a random element of the space.

        The random generator can be passed as argument. For complex dtypes, real and imaginary
        parts are generated independently.

        Parameters
        ----------
        rand : callable, optional
            The random function to use. Should accept the shape as a tuple and return a real
            array of that shape. Numpy functions like `numpy.random.standard_normal` conform to
            this. Default: uniform distribution on `[0, 1)` (`numpy.random.random_sample`).
        """
        return self.vec_type.rand(random_generator=rand)
    
    def poisson(self, x):
        """Return a poisson distributed vector given the distribution x.

        The random generator can be passed as argument. For complex dtypes, real and imaginary
        parts are generated independently.

        Parameters
        ----------
        rand : callable, optional
            The random function to use. Should accept the shape as a tuple and return a real
            array of that shape. Numpy functions like `numpy.random.standard_normal` conform to
            this. Default: uniform distribution on `[0, 1)` (`numpy.random.random_sample`).
        """
        assert x in self
        return self.vec_type.poisson(x)

    def randn(self):
        """Like `rand`, but using a standard normal distribution."""
        return self.rand(rand=np.random.standard_normal)

    @property
    def is_complex(self):
        """Boolean indicating whether the dtype is complex"""
        return self.vec_type.is_complex

    @property
    def size(self):
        """The size of elements (as arrays) of this vector space."""
        return np.prod(self.shape)

    @property
    def realsize(self):
        """The dimension of the vector space as a real vector space. For complex dtypes,
        this is twice the number of array elements. """
        if self.is_complex:
            return 2 * np.prod(self.shape)
        else:
            return np.prod(self.shape)

    @property
    def ndim(self):
        """The number of array dimensions, i.e. the length of the shape. """
        return len(self.shape)

    @util.memoized_property
    def identity(self):
        """The `regpy.operators.Identity` operator on this vector space. """
        return operators.Identity(self)

    def __contains__(self, x):
        return self.vec_type.is_vector(x=x)

    def flatten(self, x):
        """Transform the vector `x`, an element of the vector space, into a flattened vector. Inverse
        to `fromflat`.

        Parameters
        ----------
        x : self.vec_type.type
            The vector to transform.

        Returns
        -------
        array
            The flattened array. If memory layout allows, it will be a view into `x`.
        """
        assert x in self
        return self.vec_type.flatten(x)

    def fromflat(self, x):
        """Transform a flattened vector into an element of the vector space. Inverse to `flatten`.

        Parameters
        ----------
        x : array-like
            The flat vector to transform

        Returns
        -------
        array
            The reshaped array.
        """
        return self.vec_type.fromflat(x)

    def complex_space(self):
        """Compute the corresponding complex vector space.

        Returns
        -------
        VectorSpaceBase
            The complex space corresponding to this vector space.
        """
        return VectorSpaceBase(self.vec_type.to_complex)

    def real_space(self):
        """Compute the corresponding real vector space.

        Returns
        -------
        VectorSpaceBase
            The real space corresponding to this vector space.
        """
        return VectorSpaceBase(self.vec_type.to_real)
    
    def masked_space(self,mask):
        """Gives a masked space given a mask.

        Parameters
        ----------
        mask : boolean
            mask for masking the vector space
        
        Returns
        -------
        VectorSpaceBase
            The masked Space depending on the vector space.
        """
        raise NotImplementedError
    
    def vdot(self,x,y):
        return self.vec_type.vdot(x,y)
    
    def norm(self,x):
        return self.vec_type.norm(x)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (
                self.vec_type == other.vec_type
            )
        else:
            return False
        
    def __iadd__(self, other):
        if isinstance(other, VectorSpaceBase):
            return DirectSum(self, other, flatten=True)
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, VectorSpaceBase):
            return DirectSum(self, other, flatten=True)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, VectorSpaceBase):
            return DirectSum(other, self, flatten=True)
        else:
            return NotImplemented
        
    def __pow__(self, power):
        assert isinstance(power, int)
        domain = self
        for i in range(power-1):
            domain = DirectSum(domain, self, flatten=True)
        return domain


class NumPyVectorSpace(VectorSpaceBase):
    r"""Discrete space \(\mathbb{R}^\text{shape}\) or \(\mathbb{C}^\text{shape}\) (viewed as a real
    space) without any additional structure.

    VectorSpaceBases can be added, producing `DirectSum` instances.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the arrays representing elements of this vector space.
    dtype : data-type, optional
        The elements' dtype. Should usually be either `float` or `complex`. Default: `float`.
    """

    log = util.classlogger

    def __init__(self, shape:tuple, dtype=float):
        super().__init__(NumPyVector(shape=shape,dtype=dtype))
        self.dtype = self.vec_type.dtype

    def complex_space(self):
        """Compute the corresponding complex vector space.

        Returns
        -------
        VectorSpaceBase
            The complex space corresponding to this vector space as a shallow copy with modified
            dtype.
        """
        other = copy(self)
        other.dtype = np.result_type(1j, self.dtype)
        other.vec_type = NumPyVector(shape=self.shape,dtype=other.dtype)
        return other

    def real_space(self):
        """Compute the corresponding real vector space.

        Returns
        -------
        VectorSpaceBase
            The real space corresponding to this vector space as a shallow copy with modified
            dtype.
        """
        other = copy(self)
        other.dtype = np.empty(0, dtype=self.dtype).real.dtype
        other.vec_type = NumPyVector(shape=self.shape,dtype=other.dtype)
        return other
    
    def masked_space(self, mask):
        mask = np.broadcast_to(mask, self.shape)
        assert mask.dtype == bool
        res = NumPyVectorSpace(np.sum(mask), dtype=self.dtype)
        res.mask = mask
        return res

    def __mul__(self, other):
        if isinstance(other, NumPyVectorSpace):
            return Prod(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, NumPyVectorSpace):
            return Prod(other, self)
        else:
            return NotImplemented


class MeasureSpaceFcts(NumPyVectorSpace):
    r"""Discrete space \(\mathbb{R}^N\) or \(\mathbb{C}^N\) (viewed as a real
    space) with an additional measure that is given via a non-negative weight for each element of the space.
    Either the measure or the shape have to be specified. The measure defaults to the constant 1 measure for each point if it is not given.


    Parameters
    ----------
    measure : np.ndarray, optional
        The non negative array representing the point measures. If it is not given the measures are set to 1 for each point. Default: None
    shape : int or tuple of ints, optional
        The shape of the arrays representing elements of this vector space. If it is not given the shape is taken from measure. Default: None
    dtype : data-type, optional
        The elements' dtype. Should usually be either `float` or `complex`. Default: `float`.

    """

    def __init__(self,measure=None,shape=None,dtype=float):
        assert measure is not None or shape is not None
        if(isinstance(measure, np.ndarray)):
            assert np.issubdtype(measure.dtype, np.floating)
            assert np.min(measure)>=0
            shape = measure.shape
        elif(np.isscalar(measure)):
            assert isinstance(measure, int) or isinstance(measure,float)
            assert measure>=0
            assert shape!=None
        elif(measure==None):
            measure=1
        #TODO: Make a default case            
        super().__init__(shape,dtype)
        self.measure=measure
        r""" Stores values of point measures """

    @property
    def measure(self):
        return self._measure
    
    @measure.setter
    def measure(self,new_measure):
        if np.isscalar(new_measure):
            assert isinstance(new_measure, int) or isinstance(new_measure,float) or (np.issubdtype(new_measure.dtype,np.number) and np.isrealobj(new_measure))
        else:
            assert  new_measure.shape==self.shape and np.issubdtype(new_measure.dtype, np.number) and np.isrealobj(new_measure)
        assert np.min(new_measure)>=0
        self._measure=new_measure

    def __eq__(self, other):
        if(not super().__eq__(other)):
            return False
        return np.all(self.measure==other.measure)
        



class GridFcts(MeasureSpaceFcts):
    r"""A vector space representing functions defined on a rectangular grid.

    Parameters
    ----------
    *coords
         Axis specifications, one for each dimension. Each can be either

         - an integer `n`, making the axis range from `0` to `n-1`,
         - a tuple that is passed as arguments to `numpy.linspace`, or
         - an array-like containing the axis coordinates.
    axisdata : tuple of arrays, optional
         If the axes represent indices into some auxiliary arrays, these can be passed via this
         parameter. If given, there must be one array for each dimension, the size of the first axis
         of which must match the respective dimension's length. Besides that, no further structure
         is imposed or assumed, this parameter exists solely to keep everything related to the
         vector space in one place.

         If `axisdata` is given, the `coords` can be omitted.
    dtype : data-type, optional
        The dtype of the vector space.
    use_cell_measure : bool, optional
        If true a measure is calculated using the volume of the grid cells. Else the measure is one for all cells. Defaults to True.
    boundary_ext : string {‘sym’, ‘const’, ‘zero’}, optional
        Defines how the measure is continued at the boundary. Possible modes are
        'sym' : The boundary coordinates are assumed to be in the center of their cell
        'const': The boundary cells are extended by a constant given in boundary_ext_const
        'zero': The boundary coordinates are assumed to be on the outer edge of their cell
        defaults to 'sym'
    boundary_ext_const: float or tuple of floats, optional
        Defines extension of cells at edges of each axis. Can be set to a constant for all axes, one constant for each axis
        or one constant for the start and one for the end of each axis. 
    """

    def __init__(self, *coords, axisdata=None, dtype=float,use_cell_measure=True,boundary_ext='sym',ext_const=None):
        views = []
        if axisdata and not coords:
            coords = [d.shape[0] for d in axisdata]

        for n, c in enumerate(coords):
            if isinstance(c, int):
                v = np.arange(c)
            elif isinstance(c, tuple):
                assert len(c) == 3, "Tuple must be of length 3"
                assert all([isinstance(c_i, int) or isinstance(c_i,float) for c_i in c]) and isinstance(c[2], int), "The axis must be real"
                v = np.linspace(*c)
            else:
                v = np.asarray(c).view()
                assert np.issubdtype(v.dtype, np.number) and np.isrealobj(v), "axis must be real"
            if 1 == v.ndim < len(coords):
                s = [1] * len(coords)
                s[n] = -1
                v = v.reshape(s)
            v.flags.writeable = False
            #assert np.all(v[:-1] <= v[1:])    # ensure coords are ascending
            views.append(v)
        self.coords = np.asarray(np.broadcast_arrays(*views))
        """The coordinate arrays, broadcast to the shape of the grid. The shape will be
        `(len(self.shape),) + self.shape`."""
        assert self.coords[0].ndim == len(self.coords)

        axes = []
        extents = []
        for i in range(self.coords.shape[0]):
            slc = [0] * self.coords.shape[0]
            slc[i] = slice(None)
            axis = self.coords[i][tuple(slc)]
            axes.append(np.asarray(axis))
            extents.append(abs(axis[-1] - axis[0]))
        self.axes = axes
        """The axes as 1d arrays"""
        self.extents = np.asarray(extents)
        """The lengths of the axes, i.e. `axis[-1] - axis[0]`, for each axis."""

        if(use_cell_measure):
            super().__init__(GridFcts._calc_cell_measure(axes,boundary_ext,ext_const), dtype=dtype)
        else:
            super().__init__(shape=self.coords[0].shape, dtype=dtype)

        if axisdata is not None:
            axisdata = tuple(axisdata)
            assert len(axisdata) == len(coords)
            for i in range(len(axisdata)):
                assert self.shape[i] == axisdata[i].shape[0]
        self.axisdata = axisdata
        """The axisdata, if given."""

    def _calc_cell_measure(axes,boundary_ext,ext_const=None):
        ext_axes=[]
        if(boundary_ext=="sym"):
            ext_axes=[np.pad(v,(1,1),mode='reflect',reflect_type='odd') for v in axes]
        elif(boundary_ext=="zero"):
            ext_axes=[np.pad(v,(1,1),mode='edge') for v in axes]           
        elif(boundary_ext=="const"):
            ext_arr=np.zeros((len(axes),2))
            if(np.isscalar(ext_const)):
                ext_const=len(axes)*(ext_const,)
            assert isinstance(ext_const, tuple)
            assert len(ext_const)==len(axes)
            for i, v in enumerate(axes):
                
                if isinstance(ext_const[i],tuple):
                    assert np.isscalar(ext_const[i][0]) and np.isscalar(ext_const[i][1])
                    ext_axes.append(np.pad(v,(1,1),mode='constant',constant_values=(v[0]-ext_const[i][0], v[-1]+ext_const[i][1])))
                else:
                    assert np.isscalar(ext_const[i])
                    ext_axes.append(np.pad(v,(1,1),mode='constant',constant_values=(v[0]-ext_const[i], v[-1]+ext_const[i])))
        ax_widths=[0.5*(ext_v[2:]-ext_v[:-2]) for ext_v in ext_axes]
        assert len(axes)<=26
        prod_string=','.join([chr(k) for k in range(65,65+len(axes))])
        return np.einsum(prod_string,*ax_widths)#computes product of entries from ax_widths
            

class UniformGridFcts(GridFcts):
    """A vector space representing functions defined on a rectangular grid with equidistant axes.
    The measure is constant. Use `GridFcts` for grids with uniform axes and non-constant measures.

    All arguments are passed to the `GridFcts` constructor, but an error will be produced if any axis
    is not uniform.

    Parameters
    ----------
    *coords
         Axis specifications, one for each dimension. Each can be either

         - an integer `n`, making the axis range from `0` to `n-1`,
         - a tuple that is passed as arguments to `numpy.linspace`, or
         - an array-like containing the axis coordinates.
    axisdata : tuple of arrays, optional
         If the axes represent indices into some auxiliary arrays, these can be passed via this
         parameter. If given, there must be one array for each dimension, the size of the first axis
         of which must match the respective dimension's length. Besides that, no further structure
         is imposed or assumed, this parameter exists solely to keep everything related to the
         vector space in one place.

         If `axisdata` is given, the `coords` can be omitted.
    dtype : data-type, optional
        The dtype of the vector space.
    periodic: If true, the grid is assumed to be periodic. If coords is a tuple of triples 
        passed as arguments to numpy.linspace, the right boundaries (second elements of the triples)
        are reduced such that the difference of the second and first elements represents 
        periodicity lengths. 
    """

    def __init__(self, *coords, axisdata=None, dtype=float, periodic = False):
        if periodic and all(isinstance(c,tuple) for c in coords):
            coords = tuple((l, (l+(n-1)*r)/n ,n) for (l,r,n) in coords)
        super().__init__(*coords, axisdata=axisdata,dtype=dtype,use_cell_measure=False)
        spacing = []
        for axis in self.axes:
            assert util.is_uniform(axis)
            spacing.append(axis[1] - axis[0])
        self.spacing = np.asarray(spacing)
        """The spacing along every axis, i.e. `axis[i+1] - axis[i]`"""
        self.volume_elem = np.prod(self.spacing)
        """The volumen element, initialized as product of `spacing`"""
        self.measure = self.volume_elem
        """ Setting measure to be initialzed by `volume_element`"""
    
    @MeasureSpaceFcts.measure.setter
    def measure(self,new_measure):
        if np.isscalar(new_measure):
            assert isinstance(new_measure, int) or isinstance(new_measure,float) or np.issubdtype(new_measure.dtype,np.number)
            assert new_measure>0
            super(UniformGridFcts, self.__class__).measure.fset(self, new_measure)
        elif(isinstance(new_measure,np.ndarray)):
            assert np.all(new_measure == new_measure.flat[0])
            super(UniformGridFcts, self.__class__).measure.fset(self, new_measure.flat[0])
        self.volume_elem=self.measure
        

class DirectSum(VectorSpaceBase):
    """The direct sum of an arbitrary number of vector spaces.

    Elements of the direct sum will always be 1d real arrays.

    Note that constructing DirectSum instances can be done more comfortably simply by adding
    `VectorSpaceBase` instances. However, for generic code, when it's not known whether the summands
    are themselves direct sums, it's better to avoid the `+` overload due the `flatten` parameter
    (see below), since otherwise the number of summands is not fixed.

    DirectSum instances can be indexed and iterated over, returning / yielding the component
    vector spaces.

    Parameters
    ----------
    *summands : tuple of VectorSpaceBase instances
        The vector spaces to be summed.
    flatten : bool, optional
        Whether summands that are themselves `DirectSum`s should be merged into this instance. If
        False, DirectSum is not associative, but the join and split methods behave more
        predictably. Default: False, but will be set to True when constructing the DirectSum via
        VectorSpaceBase.__add__, i.e. when using the `+` operator, in order to make repeated sums
        like `A + B + C` unambiguous.
    """

    def __init__(self, *summands, flatten=False):
        assert all(isinstance(s, VectorSpaceBase) for s in summands)
        if flatten:
            self.summands = []
            for s in summands:
                if isinstance(s,DirectSum):
                    self.summands.extend(s.summands)
                else:
                    self.summands.append(s)
        else:
            self.summands = summands
        super().__init__(VectorSum(*[s.vec_type for s in summands],flatten=flatten))

    def masked_space(self, mask):
        if isinstance(mask,int):
            self.summands[mask]
        x = self.rand()
        _ = x[mask]
        x[mask] = self.ones()[mask]
        return DirectSum(*[s_i.masked_space(m_i) for s_i,m_i in zip(self.summands,mask)])

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (
                len(self.summands) == len(other.summands) and
                all(s == t for s, t in zip(self.summands, other.summands))
            )
        else:
            return NotImplemented

    def join(self, *xs):
        """Transform a collection of elements of the summands to an element of the direct sum.

        Parameters
        ----------
        *xs : tuple of array-like
            The elements of the summands. The number should match the number of summands,
            and for all `i`, `xs[i]` should be an element of `self[i]`.

        Returns
        -------
        1d array
            An element of the direct sum
        """
        assert all(x in s for s, x in zip(self.summands, xs))
        return TupleVector(list(xs))

    def split(self, x):
        """Split an element of the direct sum into a tuple of elements of the summands.

        The result arrays may be views into `x`, if memory layout allows it. For complex
        summands, a necessary condition is that the elements' real and imaginary parts are
        contiguous in memory.

        Parameters
        ----------
        x : array
            An array representing an element of the direct sum.

        Returns
        -------
        tuple of arrays
            The components of x for the summands.
        """
        assert x in self
        return tuple(x.v)

    def __getitem__(self, item):
        return self.summands[item]

    def __iter__(self):
        return iter(self.summands)

    def __len__(self):
        return len(self.summands)


class Prod(NumPyVectorSpace):
    """The tensor product of an arbitrary number of vector spaces.

    Elements of the tensor product will always be arrays with in n-dim where n is number of factors. 
    Representing each coefficient to a basis tensor that are mad up be the tensor product of each 
    basis element from teh factored spaces. Note, that spaces with possible multidimensional elements
    (e.g. `UniformGridFcts` with multiple dimensions) get flatted. 

    Prod instances can be indexed and iterated over, returning / yielding the component vector spaces.

    Parameters
    ----------
    *factors : tuple of VectorSpaceBase instances
        The vector spaces to be factored.
    flatten : bool, optional
        Whether factors that are themselves `Prod`s should be merged into this instance. If False, Prod is not associative, but the product method behaves more predictably.
        Default: False
    """

    def __init__(self, *factors, flatten=False):
        assert all(isinstance(s, VectorSpaceBase) for s in factors)
        assert all(s.is_complex for s in factors) or all(not s.is_complex for s in factors)
        self.factors = []
        """List of the `VectorSpaceBases` to be taken as Product."""
        shape = ()
        self.volume_elem = 1
        """Poduct of the `volume_elem` of all factors that have defined this property. """
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
                shape += (s.size,)
        super().__init__(shape,dtype=dt)

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and
            len(self.factors) == len(other.factors) and
            all(s == t for s, t in zip(self.factors, other.factors))
        )

    def product(self, *xs):
        """Transform a collection of elements of the factors into an element of the tensor product by an outer product.

        Parameters
        ----------
        *xs : tuple of array-like
            The elements of the factors. The number should match the number of factors,
            and for all `i`, `xs[i]` should be an element of `self[i]`.

        Returns
        -------
        n-dim array
            An element of the tensor product
        """
        assert all(x in s for s, x in zip(self.factors, xs))
        elm = 1
        for s, x in zip(self.factors, xs):
            elm = np.ma.outer(elm,x)
        return elm

    def __getitem__(self, item):
        return self.factors[item]

    def __iter__(self):
        return iter(self.factors)

    def __len__(self):
        return len(self.factors)
