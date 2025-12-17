from copy import copy, deepcopy
from warnings import warn
from typing import *

import numpy as np

from regpy.util import is_complex_dtype,is_real_dtype, Errors, complex2real,real2complex, make_repr, is_uniform
from .base import VectorSpaceBase

__all__ = ["NumPyVectorSpace", "MeasureSpaceFcts", "GridFcts", "UniformGridFcts","Prod"]

class NumPyVectorSpace(VectorSpaceBase):
    r"""Discrete space \(\mathbb{R}^\text{shape}\) or \(\mathbb{C}^\text{shape}\) (viewed as a real
    space) without any additional structure.

    `regpy.vecsps.VectorSpaceBase`s can be added, producing `DirectSum` instances.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the arrays representing elements of this vector space.
    dtype : data-type, optional
        The elements' dtype. Should usually be either `float` or `complex`. Default: `float`.
    random_seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator, RandomState}, optional
        The random seed to be used by the `numpy.random.default_rng` to construct the random generator used 
        to generate pseudo random vectors. For possible details how the argument is handled we refer to the 
        numpy documentation.
    """

    def __init__(self, shape : tuple, dtype : type = float):
        super().__init__(vec_type=np.ndarray,shape=shape, complex = is_complex_dtype(np.array([],dtype=dtype)),type = np)
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
        """Return an uninitialized element of the space.
        """
        return np.empty(shape = self.shape,dtype=self.dtype)

    def rand(self, distribution = "uniform", **kwargs):
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

    def poisson(self, x):
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="poisson sampling requires the x to be in the vector space!"))
        return self.rand(distribution="poisson", lam = x)
    
    def __contains__(self, x):
        if not super().__contains__(x):
            return False
        elif is_complex_dtype(x.dtype):
            return self.is_complex
        elif is_real_dtype(x.dtype):
            return True
        else:
            return False
        
    def flatten(self, x : np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if self.shape != x.shape:
            raise ValueError(Errors.value_error(f"After casting the input {x} to an ndarray it is not of the shape of the vector space and thus cannot be flatted to a flatted vector of the space!"))
        if self.is_complex:
            if is_complex_dtype(x.dtype):
                return complex2real(x).ravel()
            else:
                aux = self.empty()
                aux.real = x
                return complex2real(aux).ravel()
        elif is_complex_dtype(x.dtype):
            raise TypeError(Errors.type_error('Real vector space can not handle complex vectors to flatten!'))
        return x.ravel()

    def fromflat(self, x : np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if not is_real_dtype(x.dtype):
            raise TypeError(Errors.type_error(f"Flatted vectors need to be of real dtype. The given vector has different dtype"+"\n\t "+f"x = {x}"))
        if self.is_complex:
            return real2complex(x.reshape(self.shape + (2,)))
        else:
            return x.reshape(self.shape)

    def complex_space(self):
        r"""Compute the corresponding complex vector space.

        Returns
        -------
        regpy.vecsps.VectorSpaceBase
            The complex space corresponding to this vector space as a shallow copy with modified
            dtype.
        """
        other = deepcopy(self)
        other.dtype = np.result_type(1j, self.dtype)
        other.is_complex = True
        return other

    def real_space(self):
        r"""Compute the corresponding real vector space.

        Returns
        -------
        regpy.vecsps.VectorSpaceBase
            The real space corresponding to this vector space as a shallow copy with modified
            dtype.
        """
        other = deepcopy(self)
        other.dtype = np.empty(0, dtype=self.dtype).real.dtype
        other.is_complex = False
        return other
    
    def masked_space(self, mask):
        """Gives a masked space given a mask.

        Parameters
        ----------
        mask : np.ndarray
            mask for masking the vector space can be anything broadcastable to the shape of the vector space
        
        Returns
        -------
        regpy.vecsps.NumPyVectorSpace
            The masked Space depending on the vector space.
        """
        mask = np.broadcast_to(mask, self.shape)
        if mask.dtype != bool:
            raise TypeError(Errors.type_error(f"The dtype of a mask need to be boolean! was given "+"\n\t"+f"mask = {mask}"))
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
            raise ValueError(Errors.not_in_vecsp(x,self,add_info="IfPos not supported for vectors outside the space!"))
        if not self.is_complex: 
            return x > 0
        else:
            return TypeError("The vector space {} is complex, use IfPos only works for real valued functions.".format(self))

    def iter_basis(self):
        r"""Generator iterating over the standard basis of the vector space. For efficiency,
        the same array is returned in each step, and subsequently modified in-place. If you need
        the array longer than that, perform a copy. In case of complex a vector space after each
        each array modified in its place with a real one it returns the same vector with \(1i\)
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
            return make_repr(self,self.shape,self.is_complex,f"mask = {self.mask}")
        else: 
            return make_repr(self,self.shape,self.is_complex)

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
    r"""Space of functions :math:`\mathbb{R}^N\to \mathbb{R}^M` or :math:`\mathbb{R}^N\to \mathbb{C}^M` (viewed as a real
    space) with an additional measure on `\mathbb{R}^N` given by `N` non-negative weights.
    Both `N` and `M` can be multi-dimensional, i.e., tuples of integers. By default, `M` is  `()` (scalar functions). 
    Either the measure or the shape have to be specified. The measure defaults to the constant 1 measure for each point if it is not given.

    Notes
    -----

    The measure is stored as a numpy array of shape `shape + (1,)*len(shape_codomain)`, i.e., the measure is broadcasted

    Parameters
    ----------
    shape : int or tuple of positive ints, optional
        The shape of the domain of the functions (`N`). If it is not given the shape is taken from measure. Default: None
    shape_codomain : int or tuple of ints, optional
        The shape of the codomain of the functions (`M`). Default: `()`. 
    measure : np.ndarray, optional:  Default: None
        The non negative array representing the point measures. If it is not given the measures are set to 1 for each point. The shape of the measure has to be shape+(1,)*len(shape_codomain)
    dtype : data-type, optional
        The elements' dtype. Should usually be either `float` or `complex`. Default: `float`.
    random_seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator, RandomState}, optional
        The random seed to be used by the `numpy.random.default_rng` to construct the random generator used 
        to generate pseudo random vectors. For possible details how the argument is handled we refer to the 
        numpy documentation.
    """
    @overload
    def __init__(self,measure : None, shape : Tuple[int] | int, shape_codomain : Tuple[int | None] | int = (), dtype : type = float) -> None: ...

    @overload
    def __init__(self,measure : np.ndarray, shape : None, shape_codomain : Tuple[int | None] | int = (), dtype : type = float) -> None: ...

    def __init__(self,
                 measure : np.ndarray | None = None, 
                 shape : Tuple[int] | int | None = None, 
                 shape_codomain : Tuple[int | None] | int = (), 
                 dtype : type = float) -> None:
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
        super().__init__(shape = shape + shape_codomain, dtype = dtype)
        self.shape_domain = shape
        r"""The shape of the domain of the functions (`N`)."""
        self.shape_codomain = shape_codomain
        r"""The shape of the codomain of the functions (`M`)."""
        self.measure=measure
        r"""The measure on the domain of the functions (`N`)."""

    @property
    def ndim_domain(self):
        r"""The number of domain dimensions."""
        return len(self.shape_domain)
    
    @property
    def ndim_codomain(self):
        r"""The number of codomain dimensions."""
        return len(self.shape_codomain)

    def scalar_space(self):
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
        res.measure = np.squeeze(self.measure, axis = tuple(range(-self.ndim+self.ndim_domain,0)))
        del res.identity
        return res

    def vector_valued_space(self,shape_codomain):
        r"""    Returns a corresponding vector-valued function space with given shape_codomain.

        That is given a space of functions :math:`\mathbb{R}^N\to \mathbb{R}` or :math:`\mathbb{R}^N\to \mathbb{C}`,
        it returns the space of functions :math:`\mathbb{R}^N\to \mathbb{R}^M` or :math:`\mathbb{R}^N\to \mathbb{C}^M`.

        Parameters
        ----------
        shape_codomain : int or tuple of ints
        
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
        res.measure = np.squeeze(self.measure, axis = tuple(range(-self.ndim+self.ndim_domain,0)))
        del res.identity
        return res

    @property
    def measure(self):
        r""" Stores values of point measures """
        return self._measure
    
    @measure.setter
    def measure(self,new_measure):
        broadcasted_measure=self.update_measure(new_measure)
        self._measure=broadcasted_measure

    def update_measure(self,new_measure):
        try:
            broadcasted=np.broadcast_to(new_measure,self.shape_domain)
        except ValueError as e:
            raise ValueError(Errors._compose_message("Invalid Measure",f"The measure with shape {new_measure.shape} can not be broadcasted to the domain shape of the domain {self.shape_domain}. Note that the shape of the space is decomposed into shape_domain + shape_codomain given by {self.shape_domain} + {self.shape_codomain} and the measure has to be at broadcastable to the shape_domain!")) from e
        broadcasted = np.expand_dims(broadcasted, axis=tuple(range(len(self.shape_domain), len(self.shape))))
        if(not (np.issubdtype(broadcasted.dtype, np.floating) or np.issubdtype(broadcasted.dtype, np.integer))):
            raise ValueError(Errors._compose_message("Mismatch of dtype", f'Type {broadcasted.dtype} is invalid type for measure.'))
        if(np.min(broadcasted)<0):
            raise ValueError(Errors._compose_message("Not a Measure"),'Negative values are not allowed in measure.')
        return broadcasted
    
    def __eq__(self, other):
        if(not super().__eq__(other)):
            return False
        return np.allclose(self.measure,other.measure) and self.shape_codomain==other.shape_codomain
    
    def __repr__(self):
        if hasattr(self,"mask"):
            return make_repr(self,self.shape,self.is_complex,f"mask = {self.mask}",f"measure= ({self.measure})",f"shape_codomain= {self.shape_codomain}")
        else: 
            return make_repr(self,self.shape,self.is_complex,f"measure= ({self.measure})",f"shape_codomain= {self.shape_codomain}")


class GridFcts(MeasureSpaceFcts):
    r"""A vector space representing (possibly vector-valued) functions defined on a rectangular grid.

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
    shape_codomain : int or tuple of ints, optional
        The shape of the codomain of the functions. Default: `()`.
    dtype : data-type, optional
        The dtype of the vector space.
    use_cell_measure : bool, optional
        If true a measure is calculated using the volume of the grid cells. Else the measure is one for all cells. Defaults to True.
    boundary_ext : string {`sym`, `const`, `zero`}, optional
        Defines how the measure is continued at the boundary. Possible modes are
        'sym' : The boundary coordinates are assumed to be in the center of their cell
        'const': The boundary cells are extended by a constant given in boundary_ext_const
        'zero': The boundary coordinates are assumed to be on the outer edge of their cell
        defaults to 'sym'
    ext_const: float or tuple of floats, optional
        Defines extension of cells at edges of each axis. Can be set to a constant for all axes, one constant for each axis
        or one constant for the start and one for the end of each axis. Is only used in combination with `boundary_ext='const'`
        in which case it needs to be defined.
    
    Notes
    -----
    If `axisdata` is given, the `coords` can be omitted.
    """

    def __init__(self, *coords, 
                 axisdata : None | tuple[np.ndarray] | list = None, 
                 shape_codomain : int | tuple[int] = (), 
                 dtype : type = float,
                 use_cell_measure : bool =True,
                 boundary_ext : str = 'sym',
                 ext_const : None | float | tuple[float] = None):
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
                             dtype=dtype
                             )
        else:
            super().__init__(shape=self.coords[0].shape, 
                             shape_codomain=shape_codomain, 
                             dtype=dtype
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

    def _calc_cell_measure(axes,boundary_ext,ext_const=None):
        if len(axes)> 26:
            raise ValueError(Errors.value_error(f"Computing the cell measure is only supported for less then 26 axes you have {len(axes)} axes."))
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
        ax_widths=[np.array([aw[0]]) if(np.allclose(aw[0],aw)) else aw for aw in ax_widths]#collapse constant width axis
        prod_string=','.join([chr(k) for k in range(ord('A'),ord('A')+len(axes))])
        return np.einsum(prod_string,*ax_widths)#computes product of entries from ax_widths
    
    def coord_distances(self,point=None,axes=None):
        r"""Computes the euclidean distances of all grid points to the given point. If the point is None, then the distance from the origin is returned.

        Parameters
        ----------
            point : np.ndarray, optional
              Point to which the distances are computed. Defaults to None.
            axes : iterable, optional
              Axes over which the distances are computed. If None the distance is computed over all axes. Defaults to None.


        Returns
        -------
            np.ndarray: Numpy array with same shape as domain containing all the distances.
        """
        if point is None:
            point=np.zeros(self.ndim)
        if axes is None:
            axes=tuple(j for j in range(self.ndim))
        if(min(axes)<0 or max(axes)>=self.ndim):
            raise ValueError(Errors.value_error(f"Axes {axes} out of bounds for grid with {self.ndim} axes."))
        if(not isinstance(point,np.ndarray) or  point.shape[0]!=self.ndim or point.ndim!=1):
            raise ValueError(Errors.value_error(f"Point {point} not a numpy array or not compatible with domain with dimension {self.ndim}."))
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
    shape_codomain : int or tuple of ints, optional
    dtype : data-type, optional
        The dtype of the vector space.
    periodic: If true, the grid is assumed to be periodic. If coords is a tuple of triples 
        passed as arguments to numpy.linspace, the right boundaries (second elements of the triples)
        are reduced such that the difference of the second and first elements represents 
        periodicity lengths.
    """

    def __init__(self, *coords, 
                 axisdata : None | tuple[np.ndarray] = None, 
                 shape_codomain : int | tuple[int] = (),  
                 dtype : type =float, 
                 periodic : bool = False):
        if periodic and all(isinstance(c,tuple) for c in coords):
            coords = tuple((l, (l+(n-1)*r)/n ,n) for (l,r,n) in coords)
        super().__init__(*coords, 
                         axisdata=axisdata,
                         shape_codomain=shape_codomain,
                         dtype=dtype,
                         use_cell_measure=False
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
    
    def update_measure(self, new_measure):
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
    *factors : tuple of regpy.vecsps.VectorSpaceBase instances
        The vector spaces to be factored.
    flatten : bool, optional
        Whether factors that are themselves `Prod`\s should be merged into this instance. If False, Prod is not associative, but the product method behaves more predictably.
        Default: False

    """

    def __init__(self, *factors, 
                 flatten : bool = False):
        if any(not isinstance(s, VectorSpaceBase) for s in factors):
            raise TypeError(Errors.type_error("One of spaces is to factor is not a VectorSpace!"))
        if any(s.is_complex for s in factors) and any(not s.is_complex for s in factors):
            raise TypeError(Errors.type_error("There are both complex and non-complex Factors. This is not supported in Prod!"))
        self.factors = []
        """List of the `VectorSpaceBases` to be taken as Product."""
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
                shape += (s.size,)
        characters=tuple(chr(k) for k in range(ord('A'),ord('A')+len(self.factors)))
        self._prod_trafo_string=f"{','.join(characters)}->{''.join(characters)}"
        """String to compute the outer product in einsum."""
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
        if any(x not in s for s, x in zip(self.factors, xs)):
            raise ValueError(Errors.value_error("One of the vectors to be taken in the outer product is not in the corresponding factor."))
        return np.einsum(self._prod_trafo_string,*[x.flat for x in xs],optimize=True)


    def __getitem__(self, item):
        return self.factors[item]

    def __iter__(self):
        return iter(self.factors)

    def __len__(self):
        return len(self.factors)
