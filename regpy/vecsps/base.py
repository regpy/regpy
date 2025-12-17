from copy import copy,deepcopy
from math import sqrt
from dataclasses import dataclass
from typing import List, overload


import numpy as np

from regpy.util import ClassLogger,Errors,make_repr,memoized_property,get_rng, set_rng_seed

__all__ = ["TupleVector", "VectorSpaceBase", "DirectSum"]

@dataclass 
class TupleVector:
    v: List

    __array_ufunc__ = None

    log = ClassLogger()

    def copy(self):
        return copy(self)

    def __post_init__(self):
        self.v = list(self.v)
        self.types = [type(v_i) for v_i in self.v]
        self.ndim = len(self.types)

    def __len__(self):
        return self.ndim
    
    def conj(self):
        return TupleVector([v_i.conj() for v_i in self])

    @property
    def real(self):
        return TupleVector([v_i.real for v_i in self])
    
    @property
    def imag(self):
        return TupleVector([v_i.imag for v_i in self])

    def __eq__(self,other):
        if isinstance(other,TupleVector) and self.ndim == other.ndim:
            return TupleVector([s_i == o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i == other for s_i in self.v])
    
    def __lt__(self, other):
        if isinstance(other,TupleVector) and self.ndim == other.ndim:
            return TupleVector([s_i < o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i < other for s_i in self.v])

    def __le__(self, other):
        if isinstance(other,TupleVector) and self.ndim == other.ndim:
            return TupleVector([s_i <= o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i <= other for s_i in self.v])

    def __ne__(self, other):
        return not (self == other)
    
    def __ge__(self, other):
        if isinstance(other,TupleVector) and self.ndim == other.ndim:
            return TupleVector([s_i >= o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i >= other for s_i in self.v])
    
    def __gt__(self, other):
        if isinstance(other,TupleVector) and self.ndim == other.ndim:
            return TupleVector([s_i > o_i for s_i,o_i in zip(self.v,other.v)])
        else:
            return TupleVector([s_i > other for s_i in self.v])

    def all(self):
        return all((v_i.all() for v_i in self.v))

    def any(self):
        return any((v_i.any() for v_i in self.v))
    
    def __or__(self,other):
        if not isinstance(other,TupleVector) or self.ndim != other.ndim:
            raise ValueError(Errors.value_error("Comparing with TupleVector only supported for TupleVectors of identical dimension!"))
        return TupleVector([s_i | o_i for s_i,o_i in zip(self.v,other.v)])
    
    def sum(self,**kwargs):
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
    
    def __iadd__(self,other):
        if not isinstance(other,TupleVector) or other.ndim != self.ndim or any([t_o!=t_s for t_o,t_s in zip(other.types,self.types)]):
            raise ValueError(Errors.value_error("Adding TupleVector only supported for TupleVectors of identical dimension and identical type in each component!"))
        for k,v_o in enumerate(other.v):
            self.v[k] += v_o
        return self

    def __isub__(self,other):
        if not isinstance(other,TupleVector) or other.ndim != self.ndim or any([t_o!=t_s for t_o,t_s in zip(other.types,self.types)]):
            raise ValueError(Errors.value_error("Subtracting TupleVector only supported for TupleVectors of identical dimension and identical type in each component!"))
        for k,v_o in enumerate(other.v):
            self.v[k] -= v_o
        return self
    
    def __neg__(self):
        return TupleVector([-v_i for v_i in self])
    
    def __add__(self,other):
        if not isinstance(other,TupleVector) or other.ndim != self.ndim or any([t_o!=t_s for t_o,t_s in zip(other.types,self.types)]):
            raise ValueError(Errors.value_error("Adding TupleVector only supported for TupleVectors of identical dimension and identical type in each component!"))
        return TupleVector([s_k + o_k for s_k,o_k in zip(self,other)])
    
    def __radd__(self,other):
        return self + other
    
    def __sub__(self,other):
        return self + (-1*other)
    
    def __rsub__(self,other):
        return (-1*self) + other
    
    def __imul__(self,other):
        if not isinstance(other,(float,int,complex)):
            raise ValueError(Errors.value_error("Multiplying TupleVector only supported for scalars (int, float, or complex)!"))
        for k in range(self.ndim):
            self.v[k] *= other
        return self
    
    def __itruediv__(self,other):
        if not isinstance(other,(float,int,complex)):
            raise ValueError(Errors.value_error("Division TupleVector only supported for scalars (int, float, or complex)!"))
        for k in range(self.ndim):
            self.v[k] /= other
            return self
    
    def __mul__(self,other):
        from regpy.operators.base import Operator,PtwMultiplication
        if isinstance(other,float) or isinstance(other,int) or isinstance(other,complex):
            return TupleVector([other * s_k for s_k in self])
        elif isinstance(other,Operator):
            return PtwMultiplication(other.codomain, self) * other
        else:
            raise NotImplementedError(Errors.generic_message(f"Multiplication of TupleVector with {type(other)} is not defined. It has to be either a number eg float, int or complex or an Operator."))
        
    def __rmul__(self,other):
        return self * other

    def __truediv__(self,other):
        if not isinstance(other,(float,int,complex)):
            raise ValueError(Errors.value_error("Division TupleVector only supported for scalars (int, float, or complex)!"))
        return TupleVector([s_k / other for s_k in self])

    def __iter__(self):
        return iter(self.v)
    
    def __getitem__(self, key):
        if isinstance(key,slice) or isinstance(key,int):
            return self.v[key]
        elif (isinstance(key,TupleVector) and self.ndim == key.ndim):
            return TupleVector([v_i[k_i] for v_i,k_i in zip(self,key)])
        elif isinstance(key,(list,tuple)) and len(key) == self.ndim: 
            return TupleVector([v_i[k_i] for v_i,k_i in zip(self,key)])
        else:
            raise KeyError(Errors.indexation(key,self,add_info=f"keys of type {type(key)} are not supported either int or list of length {self.ndim}"))
    
    def __setitem__(self, key, item):
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
                raise TypeError(Errors.type_error(f"If keys are lists or tuples of length smaller then the length of the Vector  the items can be list or TupleVectors of length same length. Got key list of length {len(key)} and item of length {len(item)}"))
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
        if not callable(method):
            raise TypeError(Errors.type_error(f"To apply a method component wise to a TupleVector the method has to be a callable. {method} is not callable"))
        return TupleVector([method(s_k) for s_k in self])

class VectorSpaceBase:
    r"""Discrete space :math:`\mathbb{R}^\text{shape}` or :math:`\mathbb{C}^\text{shape}` (viewed as a real
    space) without any additional structure.

    VectorSpaceBases can be added, producing `DirectSum` instances.

    The type if given can be used to implement the methods with the same name
    given each can deal with the following input the following methods:

     - zeros(shape : tuple)
     - ones(shape : tuple)
     - empty(shape : tuple)
     - rand(shape : tuple,random_generator : method)
     - poisson(x : Vector of the Space)
     - vdot(x : Vector of the Space, y : Vector of the Space)
     - logical_and(x : boolean Vector of the Space, y : boolean Vector of the Space)
     - logical_or(x : boolean Vector of the Space, y : boolean Vector of the Space)
     - logical_not(x : boolean Vector of the Space)
     - logical_xor(x : boolean Vector of the Space, y : boolean Vector of the Space)
     
    for convenience these methods will be linked by the same name in this
    class.

    Parameters
    ----------
    vec_type : object
        The class of vectors used.
    shape : tuple(int) or int
        The general shape of the vectors.
    complex : boolean, optional
        Determines if the vectors have complex coefficients. Default, False.
    type : {None,object}, optional
        A class, module or library that the vec_type belongs to and implements the 
        above methods. Default, None.
    """

    log = ClassLogger()

    def __init__(self, 
                 vec_type : object, 
                 shape : tuple, 
                 complex : bool = False, 
                 type = None):
        self.vec_type = vec_type
        """The vector type"""
        self.shape = (shape,) if isinstance(shape,int) else shape
        """The vector space's shape"""
        self.is_complex = complex
        self.type = type 
        """Type of the vectors if different"""
        self._no_pickle = {'type', 'vec_type'}
        """A dictionary containing modules kept extra in the copy"""

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

    def zeros(self):
        """Return the zero vector of the space.
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.zeros(shape = self.shape)
    
    def ones(self):
        """Return the zero vector of the space.
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.ones(shape = self.shape)

    def empty(self):
        """Return an uninitialized element of the space.
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.empty(shape = self.shape)
    
    def set_rng_seed(self,seed):
        set_rng_seed(seed)
    
    def _draw_sample(self, distribution : str , size = None, **kwargs):
        """Draws samples of the shape of the space from the given distribution. The distribution
        has to be given as a string representing a method associated with a distribution of
        the NumPy Generator.

        Parameters
        ----------
        distribution : str
            Name of the distribution to be used.
        size : None | int | tuple(int), optional
            The size/shape that is passed to the distribution. Default: self.realsize
        kwargs : dict
            These keyword arguments are passed to the distribution sampler. Note that we have
            `size = ` as an extra argument so it will be always handled.

        Returns
        -------
        ndarray or scalar
            A numpy array or scalar by default of the realsize of the space. Which can by 
            the `fromflat` method be directly transformed into a vector of the space.
        """
        if size is None:
            size = self.realsize
        elif isinstance(size,int) and size != self.realsize:
            self.log.debug(f"You are sampling on a size specified as an integer that is different from the realsize. This might lead to vectors that are not in the space.")
        elif isinstance(size,tuple) and np.prod(size) != self.realsize:
            self.log.debug(f"You are sampling on a size specified as an tuple of integers that taken as a product is different from the realsize. This might lead to vectors that are not in the space.")
        try:
            dist = getattr(get_rng(),distribution)
            return dist(size = size, **kwargs)
        except AttributeError:
            raise AttributeError(Errors.generic_message(f"The given distribution {distribution} is unknown to numpy maybe you misspelled, please check the documentation."))
        except TypeError:
            raise TypeError(Errors.type_error(f"The chosen distribution {distribution} might not have an argument size or one of your additional keyword arguments is unknown. Please, only use distributions listed in NumPy that have specified a size Argument."))

    def rand(self, distribution = "uniform", **kwargs):
        """Return a random vector of the vector space. Note, this might not correspond to a
        random element seen in the Hilbert space. For that we refer to the random sampling in Hilbert 
        spaces.

        For complex vectors it generates, real and imaginary parts are generated independently.

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
        array_like
            A vector in the space. The vector is created as a flat numpy array and then cast to the actual 
            vector and type by the `fromflat` method
        """
        return self.fromflat(self._draw_sample(distribution=distribution))
    
    def poisson(self, x):
        """Return a poisson distributed vector given the distribution x.

        Parameters
        ----------
        x : self.vec_type
            The lambda to be used.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="poisson sampling requires the x to be in the vector space!"))
        return self.rand(distribution="poisson", lam = self.flatten(x))
    
    def randn(self):
        """Like `rand`, but using a standard normal distribution."""
        return self.rand(distribution="standard_normal")
    
    def vdot(self,x,y):
        r"""Return the vector dot product as defined for these vectors. Note
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
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.vdot(x,y)

    def logical_and(self,x,y):
        """Logical and of two boolean vectors
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.logical_and(x,y) 

    def logical_or(self,x,y):
        """Logical or of two boolean vectors
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.logical_or(x,y) 
    
    def logical_not(self,x):
        """Logical not of a boolean vectors
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.logical_not(x) 
    
    def logical_xor(self,x,y):
        """Logical xor of two boolean vectors
        """
        if self.type is None:
            raise NotImplementedError
        return self.type.logical_xor(x,y) 

    def iter_basis(self):
        r"""Generator iterating over the standard basis of the vector space. For efficiency,
        the same array should returned in each step, and subsequently modified in-place. If you need
        the array longer than that, perform a copy. In case of a complex vector space after each
        each array modified in its place with a real one it should return the same vector with \(1i\)
        in its place.
        """
        raise NotImplementedError
    
    def __iter__(self):
        return self.iter_basis()
    
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

    @memoized_property
    def identity(self):
        """The `regpy.operators.Identity` operator on this vector space. """
        from regpy.operators import Identity
        return Identity(self)

    def __contains__(self, x):
        return isinstance(x,self.vec_type) and all(xis == si for xis,si in zip(x.shape, self.shape))

    def flatten(self, x):
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
        raise NotImplementedError

    def complex_space(self):
        """Compute the corresponding complex vector space.

        Returns
        -------
        regpy.vecsps.VectorSpaceBase
            The complex space corresponding to this vector space.
        """
        raise NotImplementedError

    def real_space(self):
        """Compute the corresponding real vector space.

        Returns
        -------
        regpy.vecsps.VectorSpaceBase
            The real space corresponding to this vector space.
        """
        raise NotImplementedError
    
    def masked_space(self,mask):
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
    
    def IfPos(self, x):
        """Analyses which components contribute to the positive part of the 
        function corresponding to the vector.

        Parameters
        ----------
        x : array-like
            The vector to analyse.

        Returns
        -------
        mask : array-like
            Mask for the vector components contributing to the positive part of a function.
        """
        raise NotImplementedError
    
    def norm(self,x):
        return sqrt(self.vdot(x,x).real)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (self.type == other.type and
                self.shape == other.shape and
                self.is_complex == other.is_complex and
                self.vec_type == other.vec_type)
    
    def compatible(self,other):
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
        if not isinstance(power, int):
            raise TypeError(Errors.not_instance(power,int,add_info="Construction of powers of a vector space only supported for integer!"))
        domain = self
        for i in range(power-1):
            domain = DirectSum(domain, self, flatten=True)
        return domain
    
    def __repr__(self):
        return make_repr(self,self.shape,self.is_complex)


class DirectSum(VectorSpaceBase):
    r"""The direct sum of an arbitrary number of vector spaces.

    Elements of the direct sum will always be 1d real arrays.

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

    def __init__(self, *summands, flatten=False):
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
    
    def rand(self, distribution = "uniform", **kwargs)-> TupleVector:
        return TupleVector([s.rand(distribution=distribution, **kwargs) for s in self.summands])
    
    def poisson(self,x)-> TupleVector:
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="Argument in poisson not in the VectorSpace."))
        return TupleVector([s.poisson(x_k) for x_k,s in zip(x,self.summands)])

    def vdot(self, x : TupleVector, y : TupleVector) -> float | complex:
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="First argument in vdot not in the VectorSpace."))
        if y not in self:
            raise ValueError(Errors.not_in_vecsp(y,self,add_info="Second argument in vdot not in the VectorSpace."))
        return sum([s_i.vdot(x_i, y_i) for x_i,y_i,s_i in zip(x,y,self.summands) ])

    def logical_and(self,x,y) -> TupleVector:
        return TupleVector([s.logical_and(x_i,y_i) for x_i,y_i,s in zip(x.v,y.v,self.summands)])
    
    def logical_or(self,x,y) -> TupleVector:
        return TupleVector([s.logical_or(x_i,y_i) for x_i,y_i,s in zip(x.v,y.v,self.summands)])
    
    def logical_not(self,x) -> TupleVector:
        return TupleVector([s.logical_not(x_i) for x_i,s in zip(x.v,self.summands)])
    
    def logical_xor(self,x,y) -> TupleVector:
        return TupleVector([s.logical_xor(x_i,y_i) for x_i,y_i,s in zip(x.v,y.v,self.summands)])

    def complex_space(self):
        return DirectSum(*[s.complex_space() for s in self.summands])

    def real_space(self):
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
    
    def iter_basis(self):
        for i,s in enumerate(self.summands()):
            vec = self.zeros()
            for b in s.iter_basis():
                vec.v[i] = b
                yield vec

    def masked_space(self, mask):
        if isinstance(mask,int):
            self.summands[mask]
        x = self.rand()
        _ = x[mask]
        x[mask] = self.ones()[mask]
        return DirectSum(*[s_i.masked_space(m_i) for s_i,m_i in zip(self.summands,mask)])
    
    def IfPos(self, x):
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

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (
                len(self.summands) == len(other.summands) and
                all(s == t for s, t in zip(self.summands, other.summands))
            )
        else:
            return NotImplemented
        
    def __contains__(self, x):
        return isinstance(x,TupleVector) and x.ndim == self.n_components and all([x_i in s_i for x_i,s_i in zip(x,self.summands)])
        

    def join(self, *xs):
        r"""Transform a collection of elements of the summands to an element of the direct sum.

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
        if any(x not in s for s, x in zip(self.summands, xs)):
            raise ValueError(Errors.value_error(f"One of the given vectors to join does not belong to the corresponding summand: "+"\n\t "+f"xs = {xs},"+"\n\t "+f"summands = {self.summands} "))
        return TupleVector(list(xs))

    def split(self, x):
        r"""Split an element of the direct sum into a tuple of elements of the summands.

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
        if not x in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="split only defined for vectors in the space!"))
        return tuple(x.v)

    def __getitem__(self, item):
        return self.summands[item]

    def __iter__(self):
        return iter(self.summands)

    def __len__(self):
        return len(self.summands)
