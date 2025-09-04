from copy import copy

import numpy as np

from regpy import util
from .base import VectorSpaceBase

__all__ = ["NumPyVectorSpace", "MeasureSpaceFcts", "GridFcts", "UniformGridFcts","Prod"]

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

    def __init__(self, shape:tuple, dtype=float):
        super().__init__(vec_type=np.ndarray,shape=shape, complex = util.is_complex_dtype(np.array([],dtype=dtype)),type = np)
        self.dtype = dtype

    def zeros(self):
        """Return the zero vector of the space.
        """
        return np.zeros(shape = self.shape,dtype=self.dtype)
    
    def ones(self):
        """Return the zero vector of the space.
        """
        return np.ones(shape = self.shape,dtype=self.dtype)

    def empty(self):
        """Return an uninitalized element of the space.
        """
        return np.empty(shape = self.shape,dtype=self.dtype)

    def rand(self,random_generator = None):
        random_generator = random_generator or np.random.random_sample 
        r = random_generator(self.shape)
        if not np.can_cast(r.dtype, self.dtype):
            raise ValueError(
                'random generator {} can not produce values of dtype {}'.format(random_generator, self.dtype))
        if util.is_complex_dtype(np.array([],dtype=self.dtype)) and not util.is_complex_dtype(r.dtype):
            c = np.empty(self.shape, dtype=self.dtype)
            c.real = r
            c.imag = random_generator(self.shape)
            return c
        else:
            return np.asarray(r, dtype=self.dtype)

    def poisson(self, x):
        return np.random.poisson(x)
    
    def __contains__(self, x):
        if not super().__contains__(x):
            return False
        elif util.is_complex_dtype(x.dtype):
            return self.is_complex
        elif util.is_real_dtype(x.dtype):
            return True
        else:
            return False
        
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

    def complex_space(self):
        r"""Compute the corresponding complex vector space.

        Returns
        -------
        VectorSpaceBase
            The complex space corresponding to this vector space as a shallow copy with modified
            dtype.
        """
        other = copy(self)
        other.dtype = np.result_type(1j, self.dtype)
        other.is_complex = True
        return other

    def real_space(self):
        r"""Compute the corresponding real vector space.

        Returns
        -------
        VectorSpaceBase
            The real space corresponding to this vector space as a shallow copy with modified
            dtype.
        """
        other = copy(self)
        other.dtype = np.empty(0, dtype=self.dtype).real.dtype
        other.is_complex = False
        return other
    
    def masked_space(self, mask):
        mask = np.broadcast_to(mask, self.shape)
        assert mask.dtype == bool
        res = NumPyVectorSpace(np.sum(mask).item(), dtype=self.dtype)
        res.mask = mask
        return res
    
    def IfPos(self, x):
        """Analyses which components contribute to the positive part of the 
        function corresponding to the vector.

        Parameters
        ----------
        x : ndarray
            The vector to analyse.

        Returns
        -------
        mask : BitArray
            Mask for the vector components contributing to the positive part of a function.
        """
        if not x in self:
            raise ValueError("The vector {} is not an element of the vector space {}".format(x,self))
        if not self.is_complex: 
            return x > 0
        else:
            return TypeError("The vector space {} is complex, use IfPos only works for real valued functions.".format(self))

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

    def __eq__(self,other):
        if hasattr(self,"mask") and hasattr(other,"mask"):
            if (self.mask == other.mask).all():
                return super().__eq__(other)
            else:
                False
        elif hasattr(self,"mask") ^ hasattr(other,"mask"):
            return False
        else:
            return super().__eq__(other)

    def __repr__(self):
        if hasattr(self,"mask"):
            return util.make_repr(self,self.shape,self.is_complex,self.mask)
        else: 
            return util.make_repr(self,self.shape,self.is_complex)

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
    r"""Discrete space :math:`\mathbb{R}^N` or :math:`\mathbb{C}^N` (viewed as a real
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
            assert np.issubdtype(measure.dtype, np.floating) or np.issubdtype(measure.dtype, np.integer)
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

    @property
    def measure(self):
        r""" Stores values of point measures """
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

    Notes
    -----
    If `axisdata` is given, the `coords` can be omitted.
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
        r"""The coordinate arrays, broadcast to the shape of the grid. The shape will be
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
        r"""The lengths of the axes, i.e. `axis[-1] - axis[0]`, for each axis."""

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
    r"""A vector space representing functions defined on a rectangular grid with equidistant axes.
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
            if(axis.shape[0]==1):
                spacing.append(1.0)
            else:
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


class Prod(NumPyVectorSpace):
    r"""The tensor product of an arbitrary number of vector spaces.

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
        Whether factors that are themselves `Prod`\s should be merged into this instance. If False, Prod is not associative, but the product method behaves more predictably.
        Default: False
    """

    def __init__(self, *factors, flatten=False):
        assert all(isinstance(s, VectorSpaceBase) for s in factors)
        assert all(s.is_complex for s in factors) or all(not s.is_complex for s in factors)
        self.factors = []
        """List of the `VectorSpaceBases` to be taken as Product."""
        shape = ()
        self.volume_elem = 1
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
                shape += (s.size,)
        super().__init__(shape,dtype=dt)

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and
            len(self.factors) == len(other.factors) and
            all(s == t for s, t in zip(self.factors, other.factors))
        )

    def product(self, *xs):
        r"""Transform a collection of elements of the factors into an element of the tensor product by an outer product.

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
