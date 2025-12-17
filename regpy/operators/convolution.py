import numpy as np

from regpy import util
from regpy.util import Errors
from regpy.vecsps import UniformGridFcts, GridFcts

from .base import PtwMultiplication, Operator, Composition, LinearCombination,OuterShift
from .numpy import FourierTransform, AddSingletonVectorDimension, PtwMatrixVectorMultiplication

__all__ = ["PaddingOperator","TruncationOperator","ConvolutionOperator","GaussianBlur", "ExponentialConvolution","FourierInterpolationOperator","FresnelPropagator",
           "gradient","curl","divergence", "Laplacian",
           "PeriodicShift"]

class PaddingOperator(Operator):
    r"""Operator that implements zero-padding for numpy arrays.

    Parameters
    ----------
    grid : regpy.vecsps.UniformGridFcts
        The domain on which the operator is defined.
    pad_amount: integer or sequence n non-negative integers determining the amount of padding
        where n is the dimension of the domain grid. E.g., for n=2,  
        pad_amount = [pad_bottom_top, pad_right_left]
        If pad_amount is an integer, this value is used for the amount of padding in each axis of the domain.
        No padding is performed in the codomain dimensions.

    Notes
    -----
    A wrapper of the np.pad function
    """

    def __init__(self,grid, pad_amount = None, pad_value =0.):
        if not isinstance(grid, UniformGridFcts):
            raise TypeError(Errors.type_error(f"First argument has to be of type UniformGridFcts. Was given {grid}"))
        self.ndim = grid.ndim
        self.ndim_domain = len(grid.shape_domain)
        if pad_amount is None:
            self.pad_amount = ((0,0),)*self.ndim
        elif isinstance(pad_amount,int):
            self.pad_amount = ((pad_amount,pad_amount),)*self.ndim_domain + ((0,0),)*(self.ndim - self.ndim_domain)
            if not pad_amount>=0:
                raise ValueError(Errors.value_error("pad_amount must be non-negative."))
        else: 
            try:
                pad_amount = np.array(pad_amount)
            except:
                raise TypeError(Errors.type_error(f'pad_amount must be None, int, or convertible to a numpy array. Got {pad_amount}'))
            if not pad_amount.dtype==int or np.any(pad_amount<0):
                raise ValueError(Errors.value_error("pad_amount must be non-negative."))
            if not pad_amount.shape == (self.ndim,):
                if not pad_amount.shape == (len(grid.shape_domain),):
                    raise ValueError(Errors.value_error(f"length of pad_amount must be (grid.ndim,) or (len(grid.shape_domain),). Got pad_amount.shape = {pad_amount.shape}, grid.ndim={self.ndim}, len(grid.shape_domain)={len(grid.shape_domain)}"))
                pad_amount = np.concatenate((pad_amount, np.zeros((self.ndim - len(grid.shape_domain),),dtype=int)))
            self.pad_amount = tuple((val,val) for val in pad_amount)
        padded_grid = UniformGridFcts(
            *[np.arange(N+pad[0]+pad[1])*spc + ax[0] - pad[0]*spc for (N,pad,spc,ax) in zip(grid.shape[:self.ndim_domain],self.pad_amount[:self.ndim_domain],grid.spacing,grid.axes)],
            dtype = grid. dtype,
            shape_codomain=grid.shape_codomain
            )
        self.pad_value = pad_value
        super().__init__(domain=grid,codomain=padded_grid,linear = (pad_value ==0))

    def _eval(self,x,differentiate=False):    
        return np.pad(x,self.pad_amount,'constant',constant_values=self.pad_value)
    
    def _derivative(self,x):    
        return np.pad(x,self.pad_amount,'constant')

    def _adjoint(self,y):
        ind = tuple(slice(pad[0],None if pad[1]==0 else -pad[1]) for pad in self.pad_amount)
        return y[ind]

def TruncationOperator(grid, truncation_amount):
    r"""Operator that implements truncation of numpy arrays.

    Parameters
    ----------
    grid : regpy.vecsps.UniformGridFcts
        The domain on which the operator is defined.
    truncation_amount: integer or (n,) np.array of non-negative integer determining the amount of truncation
        where n is the dimension of grid.
        If truncation_amount is an integer, this value is used for the amount of truncation in each axis of the domain.
        No truncation is performed in the codomain dimensions.

    Notes
    -----
    Returns the adjoint of a PaddingOperator
    """
    ndim = grid.ndim
    ndim_domain = ndim - len(grid.shape_codomain)
    if not isinstance(grid, UniformGridFcts):
        raise TypeError(Errors.type_error(f'grid must be a UniformGridFcts. Got {grid}'))
    if isinstance(truncation_amount,int):
            truncation_amount = truncation_amount * np.ones((grid.ndim-len(grid.shape_codomain),),dtype =int)
    elif isinstance(truncation_amount, np.ndarray):
        if not truncation_amount.dtype==int or not np.all(truncation_amount>=0):
            raise ValueError(Errors.value_error('truncation_amount must be non-negative integers.'))
        if not truncation_amount.shape == (ndim,):
            if truncation_amount.shape == (ndim_domain,):
                truncation_amount = np.concatenate((truncation_amount, np.zeros((ndim - ndim_domain,),dtype=int)))
            else:
                raise ValueError(Errors.value_error(f"shape of truncation_amount must be of size {ndim} or {ndim_domain}. Got {truncation_amount.shape}, {truncation_amount.dtype}"))
    else:
        raise TypeError(Errors.type_error(f"truncation_amount must be int or np.array of ints. Got {truncation_amount}"))
    if not np.all(np.array(grid.shape)>2*truncation_amount):
        raise ValueError(Errors.value_error(f'Condition grid.shape>2*truncation_amount violated: {grid.shape}, {truncation_amount}'))
    if not np.all(truncation_amount>=0):
        raise ValueError(Errors.value_error(f'Condition truncation_amount>=0 violated: Got {truncation_amount}'))
    truncated_grid = UniformGridFcts(
            *[np.arange(N-2*trunc)*spc + ax[0] + trunc*spc for (N,trunc,spc,ax) in zip(grid.shape[:ndim_domain],truncation_amount[:ndim_domain],grid.spacing,grid.axes)],
            dtype = grid.dtype,
            shape_codomain = grid.shape_codomain
            )
    pad_op = PaddingOperator(truncated_grid,truncation_amount)
    return pad_op.adjoint

class ConvolutionOperator(Composition):
    r"""Periodic convolution operator on a periodic UniformGridFcts space. 
    This is a discrete approximation of the operator

    .. math::
        (Kf)(x) = \int_D k(x-y)f(y) dy
    
    where D is the domain of the grid, and k and f are assumed to be periodic functions with periodicity cell D. 
    The implementation is based on the Fourier convolution formula 

    .. math::
        Kf = F^\ast(F(k)^\ast F(f))

    with the Fourier transform f. 
    If grid is a real vector space, the convolution kernel k must be real-valued --  
    or equivalently, :math:`F(f)` must be symmetric w.r.t. the origin. 
    The kernel k may be matrix-valued. 
    
    Parameters
    ----------
    grid : regpy.vecsps.UniformGridFcts
        The space on which the operator is defined.
        If it real, real-valued fft will be used, otherwise complex fft   
    fourier_multiplier: (:math:`F(k)`) 
        - Either a d-dimensional numpy array (d=grid.ndim_domain), the Fourier transform of the convolution kernel 
          If grid is real, the size of the last dimension is about half of that of grid
          The dimensions of the axes, which are not convolution axes should be one, the dimensions of the others 
          should correspond to the corresponding dimensions of grid 
          (If grid is real, the last axis is about half the dimension of the last grid axis corresponding to frequencies in rfft.)
        - or a function taking d real values (frequencies) and returning a real or complex number.
          (In other words, the function will be evaluated on a grid that is reciprocal to the input grid.)           
    pad_amount: [optional, default:None] None or integer or (d,) np.array of integers 
        Zero-padding should be used if periodic convolution operators are employed to approximate convolution operators on R^d.
        If pad_amount is too small or 0, aliasing artifacts can appear due to periodization. 
        Each integer specifies the number of pixels to be added on both sides in the corresponding dimension. 
        If an integer is given, this is used as pad amount in each direction. If None, no padding is performed.
        If Fourier_truncation_amount is None, the convolution restricted to the original domain is returned, otherwise
        the convolution on the padded domain is returned
    pad_value: [optional, default:0]
        The values inserted in the padded domain. If not 0, the convolution operator is not linear, but only affinely linear.
    Fourier_truncation_amount: [optional, default:None] None or integer or d-tuple of integers 
        Specifies a truncation of the Fourier domain, leading to a subsampling of the padded spatial domain.
        In particular, if Fourier_truncation_amount=0, the convolution on the full padded domain is returned. 
    convolution_axes: [optional, default: None] None or 1d numpy array integer of integers 
        Specifies a subset of the axes of grid along which convolution is performed. 
        If None, np.arange(grid.ndim_domain) is used, i.e. convolution is performed w.r.t. to all domain axes.
    kernel_matrix_shape: [optional, default: None] None or pair of positive integers
        The case where kernel_matrix_shape is not None treats the situation that the convolution kernel k is matrix-value 
        with shape kernel_matrix_shape. 
        In this case, the input space is assumed to be vector-valued with values of shape (kernel_matrix_shape[1],),
        If fourier_multiplier is given by a function, the output of this function must be matrix-valued, i.e. the codomain must be two-dimensional. 
        
                    
    Notes
    ----- 
    **functional_calculus:** 
    *Input:* A scalar function :math:`phi`.
    *Output:* The functional calculus of the convolution operator at :math:`phi`, :math:`f\mapsto F^*(F(\vaphi(k))F(f))`
    
    **composition:**
    *Input:* Another convolution operator L with kernel l
    *Output:* The composition K L, a convolution operator with Fourier multiplier :math:`F(k)*F(l)`
    *Note:* If zero-padding or Fourier truncation are used, this is not the composition K*L (implemented in Composition), 
    but it is a valid and faster approximation of the composition of the underlying convolution operators in R^d.
    
    **conv_inverse:**
    *Output:* Inverse operator, the convolution operator with Fourier multiplier :math:`F(1/k)`
    *Note:*  If zero-padding or Fourier truncation are used, this is not the exact inverse, 
    but an approximation of the inverse of the underlying convolution operators in R^d. 
    
    **conv_adjoint:**
    Output: Adjoint operator as a convolution operator with Fourier multiplier given by the pointwise Hermitian matrices
        
    **Linear combinations:**
    
    .. math::
         \alpha * K + \beta * L

    with scalars :math:`\alpha,\beta` yield convolution operators (implemented by only two Fourier transforms)
    """

    def __init__(self, grid, fourier_multiplier, pad_amount=None,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None,kernel_matrix_shape=None):
        if not isinstance(grid,UniformGridFcts):
            raise TypeError(Errors.type_error(f'grid must be of type UniformGridFcts. Got {grid}'))
        if not kernel_matrix_shape is None and not grid.shape_codomain[0] ==  kernel_matrix_shape[1]:
            raise ValueError(Errors.value_error(f'Last dimension of grid must equal last matrix kernel dimension. Got {grid.shape}, {kernel_matrix_shape}'))
        self.grid = grid
        ndim = grid.ndim_domain
        if not convolution_axes is None:
            if not np.all([0 <= ax < ndim for ax in convolution_axes]):
                raise ValueError(Errors.value_error(f"Invalid axis specified: {convolution_axes}. Must be within [0, {ndim})"))
            if not len(convolution_axes) == len(set(convolution_axes)):
                raise ValueError(Errors.value_error(f"Axes contain duplicates: {convolution_axes}"))
            self.convolution_axes =  np.array(convolution_axes)
        else:
            self.convolution_axes = np.arange(ndim)
        self.kernel_matrix_shape = kernel_matrix_shape
        if not (isinstance(self.convolution_axes,np.ndarray) and self.convolution_axes.dtype==int
                and np.max(self.convolution_axes)<grid.ndim_domain and np.min(self.convolution_axes)>=0):
            raise TypeError(Errors.type_error(f'convolution_axes must be a numpy array of integers between 0 and d. Got {self.convolution_axes}'))
        # array containing the numbers of the axes that are not convolution axes

        self.stackaxes = np.array(list(set(np.arange(grid.ndim_domain)) - set(self.convolution_axes)))

        if not callable(fourier_multiplier) and  not isinstance(fourier_multiplier,np.ndarray):
            raise TypeError(Errors.type_error('fourier_multiplier must be callable or a numpy array'))
        if not pad_amount is None:
            if isinstance(pad_amount,int):
                self.pad_amount =np.array([(pad_amount if i in self.convolution_axes else 0) for i in np.arange(grid.ndim)])
            else:
                try:
                    pad_amount = np.array(pad_amount)
                except:
                    raise TypeError(Errors.type_error('pad_amount must be None, integer or convertible to a numpy array.') )
                if not len(self.stackaxes) == 0 and not np.all(pad_amount[self.stackaxes]==0):
                    raise ValueError(Errors.value_error(f'pad_amount should be 0 for non-convolution axes. Got {pad_amount}. Non-convolution axes are {self.stackaxes}'))
                else:
                    self.pad_amount = pad_amount
        if not Fourier_truncation_amount is None:
            if isinstance(Fourier_truncation_amount,int):
                self.Fourier_truncation_amount = [(Fourier_truncation_amount if i in self.convolution_axes else 0) for i in np.arange(grid.ndim)]
            else:
                if not np.all(Fourier_truncation_amount[self.stackaxes]==0):
                    raise ValueError(Errors.value_error(f'Fourier_truncation_amount should be 0 for non-convolution axes. Got {Fourier_truncation_amount}'))
                else:
                    self.Fourier_truncation_amount = Fourier_truncation_amount

        codomain = grid if kernel_matrix_shape is None else grid.vector_valued_space((kernel_matrix_shape[0],))

        self.kwargs = {'pad_amount' : pad_amount,
                       'pad_value' : pad_value,
                       'Fourier_truncation_amount' : Fourier_truncation_amount,
                       'convolution_axes' : self.convolution_axes,
                       'kernel_matrix_shape' : self.kernel_matrix_shape}

        freq_slice = (self.convolution_axes,  *tuple(slice(None) if i in self.convolution_axes else slice(0, 1) for i in np.arange(ndim)))
        if pad_amount is None or np.all(pad_amount ==0):
            ft = FourierTransform(grid,axes=self.convolution_axes)
            if kernel_matrix_shape is None:
                ft_codomain = ft
            else:
                ft_codomain = FourierTransform(codomain,axes=self.convolution_axes)
            self._frqs = np.asarray(ft.codomain.coords)
            if callable(fourier_multiplier):
                self._otf = fourier_multiplier(*self._frqs[freq_slice])
            else:
                self._otf = fourier_multiplier
            if self.kernel_matrix_shape is None:
                multiplier = PtwMultiplication(ft.codomain, np.broadcast_to(self._otf,ft.codomain.shape))
            else:
                extended_shape = ft.codomain.shape_domain+self.kernel_matrix_shape
                multiplier = PtwMatrixVectorMultiplication(ft.codomain,  np.broadcast_to(self._otf,extended_shape))
            super().__init__(ft_codomain.adjoint, multiplier, ft)
        elif Fourier_truncation_amount is None: 
            pad_op = PaddingOperator(grid,self.pad_amount,pad_value=pad_value)
            pad_op_codomain = PaddingOperator(codomain,self.pad_amount,pad_value=pad_value)
            ft = FourierTransform(pad_op.codomain,axes=self.convolution_axes)
            if self.kernel_matrix_shape is None:
                ft_codomain = ft
            else:
                ft_codomain = FourierTransform(pad_op_codomain.codomain,axes=self.convolution_axes)
            self._frqs = np.asarray(ft.codomain.coords)
            if callable(fourier_multiplier):
                self._otf = fourier_multiplier(*self._frqs[freq_slice])
            else:
                self._otf = fourier_multiplier
            if self.kernel_matrix_shape is None:
                multiplier = PtwMultiplication(ft.codomain, np.broadcast_to(self._otf,ft.codomain.shape))
            else:
                extended_shape = ft.codomain.shape_domain+self.kernel_matrix_shape
                multiplier = PtwMatrixVectorMultiplication(ft.codomain, np.broadcast_to(self._otf,extended_shape))
            trunc_op = TruncationOperator(ft_codomain.domain, self.pad_amount)

            super().__init__(trunc_op, ft_codomain.adjoint, multiplier, ft, pad_op)
        else:
            pad_op = PaddingOperator(grid,self.self.pad_amount,pad_value=pad_value) 
            ft = FourierTransform(pad_op.codomain,axes=self.convolution_axes,centered=True)
            if not grid.dtype == complex:
                raise NotImplementedError
            trunc_op = TruncationOperator(ft.codomain,Fourier_truncation_amount)
            self._frqs = np.asarray(trunc_op.codomain.coords)
            if callable(fourier_multiplier):
                self._otf = fourier_multiplier(*self._frqs[freq_slice])
            else:
                self._otf = fourier_multiplier
            fac = np.sqrt(np.prod(trunc_op.codomain.shape)/np.prod(trunc_op.domain.shape))
            if self.kernel_matrix_shape is None:
                multiplier = PtwMultiplication(trunc_op.codomain, np.broadcast_to(fac*self._otf,trunc_op.codomain.shape))
            else:
                multiplier = PtwMatrixVectorMultiplication(trunc_op.codomain, np.broadcast_to(self._otf,trunc_op.codomain.shape)) 
            new_coords = FourierTransform.frequencies(trunc_op.codomain,centered=True, axes=self.convolution_axes)
            codomain = UniformGridFcts(*new_coords, dtype=complex)
            if kernel_matrix_shape is not None:
                codomain = codomain.vector_valued_space(kernel_matrix_shape[0])
            ft_codomain = FourierTransform(codomain,axes=self.convolution_axes,centered=True)
            if ft_codomain.codomain != multiplier.codomain:
                raise RuntimeError(f'The codomain of the multiplier and the codomain of the Fourier Transform do not match!. {ft_codomain.shape},{multiplier.codomain.shape}')
            super().__init__(ft_codomain.adjoint,multiplier,trunc_op,ft,pad_op)

    @property
    def freqs(self):
        """coordinates in Fourier space"""
        return self._freqs    
    
    @property
    def fourier_multiplier(self):
        """Fourier transform of the convolution kernel"""
        return self._otf

    def functional_calculus(self,f):
        """Implements functional calculus for the convolution operator.

        Parameters
        ----------
        f : callable
            The function implemented as a callable

        Returns
        -------
        regpy.operators.ConvolutionOperator
            The convolution operator by taking functional calculus of the Fourier multiplier.

        Raises
        ------
        TypeError
            If not f is not a callable
        NotImplementedError
            If the kernel_matrix is not empty
        """
        if not callable(f):
            raise TypeError(Errors.type_error("To use functional calculus of need to be a callable!",self,"functional_calculus"))
        if not self.kernel_matrix_shape is None:
            raise NotImplementedError
        return ConvolutionOperator(self.grid,f(self.fourier_multiplier),**self.kwargs)

    def composition(self,L):
        if not isinstance(L,ConvolutionOperator):
            raise TypeError(Errors.type_error(f'Argument must be a convolution operator. Got {L}'))
        if not (self.grid == L.grid or 
                (not self.kernel_matrix_shape is None and self.grid.shape[:-1]==L.grid.shape[:-1] and self.grid.dtype==L.grid.dtype)):
            raise ValueError(Errors.value_error(f'Compositions only possible on same grid. Got {self.grid}, {L.grid}'))
        if not self._parameters_equal(self.kwargs, L.kwargs,ignore_kernel_matrix_shape=True):
            raise ValueError(Errors.value_error(f'Keyword arguments must agree. Own: {self.kwargs} Got {L.kwargs}'))
        if self.kernel_matrix_shape is None:
            if L.kernel_matrix_shape is not None:
                raise ValueError(Errors.value_error(f'Kernel matrix shape is none but L has a defined kernel shape.'))
        else:
            if L.kernel_matrix_shape is None or not self.kernel_matrix_shape[1] == L.kernel_matrix_shape[0]:
                raise ValueError(Errors.value_error(f'Kernel matrices cannot be multiplied. Given shapes are {self.kernel_matrix_shape} and {L.kernel_matrix_shape}'))
        if self.kernel_matrix_shape is None:
            new_otf = self._otf*L._otf
        else:
            new_otf = np.einsum('...ij,...jk->...ik',self._otf,L._otf)
        if self.kernel_matrix_shape is None:
            return ConvolutionOperator(L.grid,new_otf,**self.kwargs)
        else:
            kwargs_comp = self.kwargs.copy()
            kwargs_comp["kernel_matrix_shape"] = (self.kernel_matrix_shape[0],L.kernel_matrix_shape[1])
            return ConvolutionOperator(L.grid,new_otf, **kwargs_comp) 

    def conv_inverse(self):
        """Inverse convolution operator, with Fourier multiplier 1/F(k).
        Only valid for scalar convolution kernels which are non-zero everywhere in Fourier space.
        """
        if self.kernel_matrix_shape is None:
            return ConvolutionOperator(self.grid, 1/self._otf,**self.kwargs)
        else:
            raise NotImplementedError('Inverse convolution only implemented for scalar convolution kernels.')           

    def conv_adjoint(self):
        """Adjoint convolution operator, with Fourier multiplier F(k)^*.
        In contrast to the adjoint operator implemented in Operator, this is again a convolution operator.
        """
        if self.kernel_matrix_shape is None:
            return ConvolutionOperator(self.grid, self._otf.conj(),**self.kwargs)
        else:       
            kernel_matrix_shape = (self.kernel_matrix_shape[1],self.kernel_matrix_shape[0])
            hermitian_otf = np.conj(np.swapaxes(self._otf,-2,-1))
            kwargs_adj = self.kwargs.copy()
            kwargs_adj["kernel_matrix_shape"] = kernel_matrix_shape
            return ConvolutionOperator(self.grid.vector_valued_space(kernel_matrix_shape[1]), 
                                       hermitian_otf, 
                                       **kwargs_adj
                                       )           

    def _parameters_equal(self,p,q,ignore_kernel_matrix_shape=False):
        return (np.all(p['pad_amount']==q['pad_amount']) 
                and p['pad_value'] == q['pad_value']
                and ((p['Fourier_truncation_amount'] is None and q['Fourier_truncation_amount'] is None)
                    or np.all(p['Fourier_truncation_amount']==q['Fourier_truncation_amount']))
                and np.all(p['convolution_axes']==q['convolution_axes'])
                and (p['kernel_matrix_shape']==q['kernel_matrix_shape'] or ignore_kernel_matrix_shape)
        ) 

    def __rmul__(self, other):
        if np.isscalar(other):
            if other == 1:
                return self
            else:
                return ConvolutionOperator(self.grid,
                                           other*self._otf,
                                           **self.kwargs
                                           )         
        elif other in self.codomain:
            return PtwMultiplication(self.codomain, other) * self
        elif isinstance(other, Operator):
            return Composition(other, self) 
        else:
            return NotImplemented

    def __add__(self, other):
        if np.isscalar(other) and other == 0:
            return self
        elif isinstance(other, ConvolutionOperator):
            if self.grid != other.grid:
                raise ValueError(Errors.value_error("Cannot add two Convolution operators with different grids!"))
            if not self._parameters_equal(self.kwargs,other.kwargs,ignore_kernel_matrix_shape=True):
                raise ValueError(Errors.value_error("Cannot add two Convolution operators with different parameters ignoring the kernel matrix!"))
            return ConvolutionOperator(self.grid,self._otf + other._otf,**self.kwargs)
        else:
            return super().__add__(other)

    def __repr__(self):
        return util.make_repr(self, self._otf)

def _Lap_in_FD(*freq,dim_codomain=None):
    shape_domain = freq[0].shape
    toret = np.zeros(shape_domain+(dim_codomain,dim_codomain),dtype=complex)
    scal_Lap = -sum((2*np.pi*y)**2 for y in freq)
    for j in range(dim_codomain):
        toret[...,j,j] = scal_Lap
    return toret

class Laplacian(ConvolutionOperator):
    """Laplace operator with periodic boundary conditions, implemented as convolution operator. 
    The second derivatives are computed with respect to the coordinates of the given grid. 
    If vector-valued spaces, this is the vector-Laplacian.

    Parameters
    ----------
    grid: regpy.vecsps.UniformGridFcts
        The Grid.
    pad_amount
        See Convolution Operator.
    pad_value
        See Convolution Operator.
    Fourier_truncation_amount
        See Convolution Operator.
    convolution_axes
        See Convolution Operator
    """
    def __init__(self,grid, pad_amount=None,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None):
        if grid.shape_codomain == ():
            super().__init__(grid,
                            lambda *x : -sum((2*np.pi*y)**2 for y in x),
                            pad_amount=pad_amount,pad_value=pad_value,
                            Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes
                            )
        else:
            super().__init__(grid,
                            lambda *x : _Lap_in_FD(*x,dim_codomain=grid.shape_codomain[0]),
                            kernel_matrix_shape=(grid.shape_codomain[0],)*2,
                            pad_amount=pad_amount,pad_value=pad_value,
                            Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes
                            )

def gradient(grid,pad_amount=None,pad_value=0.,Fourier_truncation_amount=None,convolution_axes=None):
    """Gradient operator with periodic boundary conditions, implemented as convolution operator.
    The first derivatives are computed with respect to the coordinates of the given grid.   
    
    Parameters
    ----------
    grid: regpy.vecsps.UniformGridFcts
       The codomain shape of grid is ignored. The codomain of the gradient operator is vector-valued with shape (grid.ndim_domain,), and the domain has codomain shape (1,).  
    pad_amount
        See Convolution Operator. 
    pad_value
        See Convolution Operator.
    Fourier_truncation_amount
        See Convolution Operator.
    convolution_axes
        See Convolution Operator.
    """
    grad = ConvolutionOperator(grid.vector_valued_space((1,)) if grid.shape_codomain==() else grid,
                               lambda *x : 2j*np.pi* np.stack(list(y[...,np.newaxis] for y in x),axis=-2),
                               kernel_matrix_shape=(len(grid.shape_domain),1),
                               pad_amount=pad_amount,pad_value=pad_value,
                               Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes
                            )
    return grad

def divergence(grid,pad_amount=None,pad_value=0.,Fourier_truncation_amount=None,convolution_axes=None):
    """Divergence operator with periodic boundary conditions, implemented as convolution operator.
    The first derivatives are computed with respect to the coordinates of the given grid.
    
    Parameters
    ----------
    grid: regpy.vecsps.UniformGridFcts
         The codomain of grid is ignored. The domain of the divergence operator is vector-valued with shape (grid.ndim_domain,), and the codomain has codomain shape (1,).
    pad_amount
        See Convolution Operator.
    pad_value
        See Convolution Operator.
    Fourier_truncation_amount
        See Convolution Operator.
    convolution_axes
        See Convolution Operator.
    """
    grad = gradient(grid.vector_valued_space(1),pad_amount=pad_amount,pad_value=pad_value,
                                Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes)
    return - grad.conv_adjoint()

def curl(grid,pad_amount=None,pad_value=0.,Fourier_truncation_amount=None,convolution_axes=None):
    """Curl operator with periodic boundary conditions, implemented as convolution operator.
    The first derivatives are computed with respect to the coordinates of the given grid.
    
    Parameters
    ----------
    grid: regpy.vecsps.UniformGridFcts
         grid must be three-dimensional. The codomain of grid is ignored. The domain and the codomain of the curl operator are both vector-valued with shape (3,). 
    pad_amount 
        See Convolution Operator.
    pad_value
        See Convolution Operator.
    Fourier_truncation_amount
        See Convolution Operator.
    convolution_axes
        See Convolution Operator.
    """
    if not ((convolution_axes is None and len(grid.shape_domain)==3) or (convolution_axes is not None and len(convolution_axes)==3)):
        raise ValueError(Errors.value_error('grid must be three-dimensional'))
    return ConvolutionOperator(grid.vector_valued_space(3),
                               _curl_in_FD,kernel_matrix_shape=(3,3),
                               pad_amount=pad_amount,pad_value=pad_value,Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes
                               )

def _curl_in_FD(Dx,Dy,Dz):
    if not(Dx.shape == Dy.shape == Dz.shape):
        raise ValueError(Errors.value_error("To compute the curl the single vectors need to have identical shape",meth="_curl_in_FD")) 
    toret = np.zeros(Dx.shape+(3,3),dtype=complex)
    toret[...,0,1] = - 2j*np.pi*Dz
    toret[...,0,2] =   2j*np.pi*Dy
    toret[...,1,0] =   2j*np.pi*Dz
    toret[...,1,2] = - 2j*np.pi*Dx
    toret[...,2,0] = - 2j*np.pi*Dy
    toret[...,2,1] =   2j*np.pi*Dx
    return toret

class PeriodicShift(ConvolutionOperator):
    """Periodic shift operator on a given uniform grid, implemented as convolution operator. 
    
    Parameters
    ----------
    grid: regpy.vecsps.UniformGridFcts 
    shift: array or tuple of length grid.ndim_domain
       Amount by which grid functions are shifted (in units of grid)
    pad_amount
        See Convolution Operator.
    pad_value
        See Convolution Operator.
    Fourier_truncation_amount
        See Convolution Operator.
    convolution_axes
        See Convolution Operator.
    """
    def __init__(self,grid, shift,pad_amount=None,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None):
        super().__init__(grid,
                        lambda *x : np.exp(sum(2j*np.pi*sh*y for y,sh in zip(x,shift))),
                        pad_amount=pad_amount,pad_value=pad_value,
                        Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes
                        )  

def GaussianBlur(grid,sigma=1.,pad_amount=None,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None):
    """Convolution with a Gaussian kernel
    
    Parameters
    ----------
    grid: regpy.vecsps.UniformGridFcts 
    sigma: scalar, default:1 
        width of the Gaussian kernel
    pad_amount
        See Convolution Operator.
    pad_value
        See Convolution Operator.
    Fourier_truncation_amount
        See Convolution Operator.
    convolution_axes
        See Convolution Operator.
    """
    if not np.isscalar(sigma):
        raise TypeError(Errors.type_error("To define a Gaussian blur the width of the gaussian kernel has to be a scalar!"))
    Lap = Laplacian(grid,pad_amount=pad_amount,pad_value=pad_value,
                        Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes)
    return Lap.functional_calculus(lambda t: np.exp((sigma/2)**2 * t))

                                   
class ExponentialConvolution(ConvolutionOperator):
    r"""Convolution with an exponential function :math:`exp(-|x|_1/a)`.
    """
    def __init__(self,grid,a,pad_amount=None,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None):
        super().__init__(grid,
                        lambda *x : np.prod([1/(1 + (2*np.pi*a*y)**2) for y in x],axis=0),
                        pad_amount=pad_amount,pad_value=pad_value,
                        Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes
                        )
        
class FourierInterpolationOperator(ConvolutionOperator):
    r"""Interpolation operator implemented as Fourier multiplier with the constant 1 function, 
    using Fourier_truncation_amount to change the grid in the spatial domain.
    """
    def __init__(self,grid,pad_amount=None,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None):
        super().__init__(grid,np.ones(grid.shape),pad_amount=pad_amount,pad_value=pad_value,
                        Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes)

def FresnelPropagator(grid,fresnel_number, pad_amount=None,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None):
    r"""Time evolution operator over the unit a interval for the Schrödinger equation
    
    .. math::
        \frac{\partial u}{\partial t} = \frac{i}{4\pi F} \Delta u
    
    i.e.  :math:`u(t=0,\cdot)\mapsto u(t=1,\cdot)`.

    This operator coincides with Fresnel-propagation, and in particular, in 2D this models near-field 
    diffraction in the regime of the free-space paraxial Helmholtz equation.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The domain on which the operator is defined.
    fresnel_number : float
        Fresnel number of the imaging setup, defined with respect to the lengthscale
        that corresponds to length 1 in domain.coords. Governs the strength of the
        diffractive effects modeled by the Fresnel-propagator
    pad_amount : [optional: Default None]: None or integer or (dim,) np.array of integers]
        amount of padding to avoid aliasing artifacts, see ConvolutionOperator for details
    Fourier_domain_truncation: [optional: Default None]: None or integer or tuple of pairs of integers]
        amount of truncation of Fourier domain, see ConvolutionOperator for details

    Notes
    -----
    This operator approximates the free-space Fresnel-propagator :math:`D_F`, which is the unitary 
    Fourier-multiplier defined by

    .. math::
        D_F(f) = FT^{-1}(m_F \cdot FT(f))


    where :math:`FT(f)(\nu) = \int_{\mathbb{R}^2} \exp(-i\xi \cdot x) f(x) Dx`
    denotes the Fourier transform and the factor :math:`m_F` is defined by
    :math:`m_F(\xi) := \exp(-i \pi |\nu|^2 / F)` with the Fresnel-number :math:`F`.
    
    It should be noted that if the grid is not dimensionless, 
    the frequency vector (here defined in units of :math:`1/\text{length}` instead of :math:`2\pi/\text{length}` 
    is not dimensionless either. 
    In this case, the Fresnel number is :math:`F = 1 / (\lambda d)`  
    with wavelength  :math:`lambda` and propagation distance :math:`d`.
    """
    if not grid.is_complex:
        raise ValueError(Errors.value_error("The grid for a Fresnel propagation need to be complex!"))
    Lap = Laplacian(grid,pad_amount=pad_amount,pad_value=pad_value,
                        Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes)
    return Lap.functional_calculus(lambda t: np.exp(1j / (4*np.pi* fresnel_number) * t))

from scipy.special import hankel1, jv as besselj
class PeriodizedHelmholtzVolumePotential(ConvolutionOperator):
    """Implements the convolution with a periodized version of the outgoing fundamental solution to the Helmholtz equation. 
    The fundamental solution is multiplied by the characteristic function of a maximal circle (2D) or ball (3D) in the 
    periodicity cell (which is assumed to be quadratic or cubic, respectively). 
    For sources for which the diameter of the support is smaller than half of the length of the periodicity interval, 
    the values of the potential coincide with the convolution of the fundamental solution in free space. 
    (Note that this is in sharp contrast to the behavior of the periodic Helmholtz volume potential 
    Laplace.functional_calculus(lambda t:1/(t+kappa**2))!)
    Analytic expressions for the Fourier coefficients of the convolution kernel were computed in 
    Vainikko, Gennadi M. "Fast Solvers of the Lippmann-Schwinger equation" 2000 
    Gilbert, R. P. / Kajiwara, J. / Xu, Y. S. (Eds.) Direct and Inverse Problems of Mathematical Physics Kluwer Acad. Publ.: Dordrecht
    """
    def __init__(self,grid, kappa):
        if not grid.is_complex:
            raise ValueError(Errors.value_error("The grid for a periodized Helmholtz volume potential need to be complex!"))
        if not (grid.ndim_domain ==2 or grid.ndim_domain==3):
            raise ValueError(Errors.value_error('PeriodicHelmholtzVolumePotential only implemented for dimensions 2 and 3.'))
        self.kappa = kappa      
        self.N = grid.shape[0]
        if grid.ndim_domain==2:
            if grid.shape != (self.N,self.N):
                raise ValueError(Errors.value_error("The shape of a two dimensional grid needs to be square!"))
            compute_kernel = self._compute_kernel_2d
        else:
            if grid.shape != (self.N,)*3:
                raise ValueError(Errors.value_error("The shape of a non two dimensional grid needs to be a cube!"))
            compute_kernel = self._compute_kernel_3d   
        if self.N%2 != 0:
            raise ValueError(Errors.value_error("The number of points in each dimension need to be even!"))
        if not np.all(grid.extents == grid.extents[0]):
            raise ValueError(Errors.value_error('grid must be quadratic.'))
        self.a = self.N * grid.spacing[0]/2.  # half of the periodicity length of the grid

        super().__init__(grid,
                        compute_kernel(self.kappa * self.a, grid.shape))

    # noinspection PyPep8Naming 
    @staticmethod
    def _compute_kernel_2d(R, shape):
        J = np.mgrid[[slice(-(s//2), (s+1)//2) for s in shape]]
        piabsJ = np.pi * np.linalg.norm(J, axis=0)
        Jzero = tuple(s//2 for s in shape)

        K_hat =  R**2 / (piabsJ**2 - R**2) * (
            1 + 1j*np.pi/2 * (
                piabsJ * besselj(1, piabsJ) * hankel1(0, R) -
                R * besselj(0, piabsJ) * hankel1(1, R)
            )
        )
        K_hat[Jzero] = -1/(2*R) + 1j*np.pi/4 * hankel1(1, R)
        K_hat[piabsJ == R] = 1j*np.pi*R/8 * (
            besselj(0, R) * hankel1(0, R) + besselj(1, R) * hankel1(1, R)
        )
        return 2 * R * np.fft.fftshift(K_hat)

    @staticmethod
    def _compute_kernel_3d(R, shape):
        J = np.mgrid[[slice(-(s//2), (s+1)//2) for s in shape]]
        piabsJ = np.pi * np.linalg.norm(J, axis=0)
        Jzero = tuple(s//2 for s in shape)

        K_hat =  R**2 / (piabsJ**2 - R**2) * (
            1 - np.exp(1j*R) * (np.cos(piabsJ) - 1j*R * np.sin(piabsJ) / piabsJ)
        )
        K_hat[Jzero] = -(1 - np.exp(1j*R) * (1 - 1j*R))
        K_hat[piabsJ == R] = -1j/4 * (2*R)**(-1/2) * (1 - np.exp(1j*R) * np.sin(R) / R)
        return (2*R)**(3/2) * np.fftshift(K_hat)

    def __repr__(self):
        return util.make_repr(self, self.kappa, self.a, self.N)
