import numpy as np
from math import prod
#from collections.abc import Sequence
from numpy.typing import ArrayLike

from regpy import util
from regpy.util import Errors
from regpy.vecsps import UniformGridFcts, GridFcts

from .base import PtwMultiplication, Operator, Composition
from .numpy import FourierTransform, PtwMatrixVectorMultiplication

__all__ = ["PaddingOperator","TruncationOperator","ConvolutionOperator","GaussianBlur", "ExponentialConvolution","FourierInterpolationOperator","FresnelPropagator",
           "Derivative","gradient","curl","divergence", "Laplacian",
           "PeriodicShift","BesselPotential","PeriodizedHelmholtzVolumePotential"]

########################### padding and truncation #################################################

class PaddingOperator(Operator):
    r"""Operator that implements zero-padding for numpy arrays.

    Parameters
    ----------
    grid : regpy.vecsps.UniformGridFcts
        The domain on which the operator is defined.
    pad_amount: int | ArrayLike[int]
        sequence n non-negative integers determining the amount of padding
        where n is the dimension of the domain grid. E.g., for n=2,  
        pad_amount = [pad_bottom_top, pad_right_left]
        If pad_amount is an integer, this value is used for the amount of padding in each axis of the domain.
        No padding is performed in the codomain dimensions.
    pad_value: grid.dtype, optional
        value with which the padded region is filled. Defaults to 0.. 
        If a value different from 0. is specified, PaddingOperator becomes nonlinear!
    last_axis_positive: bool, optional
        Backward-compatible shorthand for ``positive_axis=grid.ndim_domain-1``.
    positive_axis: int or None, optional
        Axis containing only non-negative frequencies. Padding is applied only on its right-hand side.

    Notes
    -----
    A wrapper of the np.pad function
    """

    def __init__(self,grid:UniformGridFcts, 
                 pad_amount: int | list[int] | tuple[int]= 0, 
                 pad_value:float =0.,
                 last_axis_positive:bool = False,
                 positive_axis:int|None = None):
        if not isinstance(grid, UniformGridFcts):
            raise TypeError(Errors.type_error(f"First argument has to be of type UniformGridFcts. Was given {grid}"))
        self.ndim = grid.ndim
        self.ndim_domain = grid.ndim_domain
        if isinstance(pad_amount,(int,np.integer)):
            if pad_amount < 0:
                raise ValueError(Errors.value_error("pad_amount must be non-negative."))
            pad_amount = np.full(self.ndim_domain, pad_amount, dtype=int)
        else: 
            try:
                pad_amount = np.asarray(pad_amount)
            except (TypeError, ValueError) as error:
                raise TypeError(Errors.type_error(f'pad_amount must be int, or array-like. Got {pad_amount}')) from error
            if pad_amount.ndim != 1 or not np.issubdtype(pad_amount.dtype, np.integer) or np.any(pad_amount < 0):
                raise ValueError(Errors.value_error(f"pad_amount must be a one-dimensional array of non-negative integers. Got {pad_amount}"))
            if pad_amount.shape == (self.ndim_domain,):
                pass
            elif pad_amount.shape == (self.ndim,):
                if np.any(pad_amount[self.ndim_domain:] != 0):
                    raise ValueError(Errors.value_error("No padding may be applied to codomain dimensions."))
                pad_amount = pad_amount[:self.ndim_domain]
            else:
                raise ValueError(Errors.value_error(
                    f"length of pad_amount must be grid.ndim_domain={self.ndim_domain} or grid.ndim={self.ndim}. "
                    f"Got pad_amount.shape={pad_amount.shape}."))
            pad_amount = pad_amount.astype(int, copy=False)

        if positive_axis is None and last_axis_positive:
            positive_axis = self.ndim_domain - 1
        if positive_axis is not None:
            if not isinstance(positive_axis,(int,np.integer)) or not 0 <= positive_axis < self.ndim_domain:
                raise ValueError(Errors.value_error(
                    f"positive_axis must be a domain axis in [0, {self.ndim_domain}). Got {positive_axis}."))
            positive_axis = int(positive_axis)
        self.positive_axis = positive_axis

        pad_pairs = [(int(val),int(val)) for val in pad_amount]
        if self.positive_axis is not None:
            pad_pairs[self.positive_axis] = (0,pad_pairs[self.positive_axis][1])
        self.pad_amount = tuple(pad_pairs) + ((0,0),)*grid.ndim_codomain
        self._inner_slice = tuple(
            slice(pad[0], None if pad[1] == 0 else -pad[1])
            for pad in self.pad_amount
        )
        self._has_padding = any(left or right for left,right in self.pad_amount)

        padded_grid = UniformGridFcts(
            *[np.arange(N+pad[0]+pad[1])*spc + ax[0] - pad[0]*spc
              for N,pad,spc,ax in zip(grid.shape_domain,self.pad_amount[:self.ndim_domain],grid.spacing,grid.axes)],
            dtype=grid.dtype,
            shape_codomain=grid.shape_codomain
            )
        self.pad_value = pad_value
        super().__init__(domain=grid,codomain=padded_grid,linear=(pad_value == 0))

    def _eval(self,x,differentiate=False):    
        if not self._has_padding:
            return x.copy()
        return np.pad(x,self.pad_amount,'constant',constant_values=self.pad_value)

    def _ieval(self,x,out,differentiate=False):
        if not self._has_padding:
            if out is x:
                return out
            if np.shares_memory(out,x):
                x = x.copy()
            np.copyto(out,x)
            return out
        if np.shares_memory(out,x):
            x = x.copy()
        out.fill(self.pad_value)
        np.copyto(out[self._inner_slice],x)
        return out
    
    def _derivative(self,x):    
        if not self._has_padding:
            return x.copy()
        return np.pad(x,self.pad_amount,'constant')

    def _iderivative(self,x,out):
        if not self._has_padding:
            if out is x:
                return out
            if np.shares_memory(out,x):
                x = x.copy()
            np.copyto(out,x)
            return out
        if np.shares_memory(out,x):
            x = x.copy()
        out.fill(0)
        np.copyto(out[self._inner_slice],x)
        return out

    def _adjoint(self,y):
        # Return an independent vector, as required by the allocation-returning Operator API.
        return y[self._inner_slice].copy()

    def _iadjoint(self,y,out):
        source = y[self._inner_slice]
        if not self._has_padding and out is y:
            return out
        if np.shares_memory(out,source):
            source = source.copy()
        np.copyto(out,source)
        return out


def TruncationOperator(grid:UniformGridFcts, 
                       truncation_amount:int|list[int]|tuple[int],
                       last_axis_positive:bool=False,
                       positive_axis:int|None=None)->Operator:
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
    if not isinstance(grid, UniformGridFcts):
        raise TypeError(Errors.type_error(f'grid must be a UniformGridFcts. Got {grid}'))
    ndim = grid.ndim
    ndim_domain = grid.ndim_domain
    if isinstance(truncation_amount,(int,np.integer)):
        truncation_amount = np.full(ndim_domain,truncation_amount,dtype=int)
    else:
        try:
            truncation_amount = np.asarray(truncation_amount)
        except (TypeError, ValueError) as error:
            raise TypeError(Errors.type_error(
                f"truncation_amount must be int or array-like. Got {truncation_amount}.")) from error
        if truncation_amount.ndim != 1 or not np.issubdtype(truncation_amount.dtype,np.integer):
            raise TypeError(Errors.type_error(
                f"truncation_amount must contain integers. Got {truncation_amount}."))
        if truncation_amount.shape == (ndim_domain,):
            pass
        elif truncation_amount.shape == (ndim,):
            if np.any(truncation_amount[ndim_domain:] != 0):
                raise ValueError(Errors.value_error("Codomain dimensions cannot be truncated."))
            truncation_amount = truncation_amount[:ndim_domain]
        else:
            raise ValueError(Errors.value_error(
                f"shape of truncation_amount must be ({ndim_domain},) or ({ndim},). "
                f"Got {truncation_amount.shape}."))
        truncation_amount = truncation_amount.astype(int,copy=False)
    if np.any(truncation_amount < 0):
        raise ValueError(Errors.value_error(
            f'truncation_amount must be non-negative. Got {truncation_amount}'))

    if positive_axis is None and last_axis_positive:
        positive_axis = ndim_domain-1
    if positive_axis is not None:
        if not isinstance(positive_axis,(int,np.integer)) or not 0 <= positive_axis < ndim_domain:
            raise ValueError(Errors.value_error(
                f"positive_axis must be a domain axis in [0, {ndim_domain}). Got {positive_axis}."))
        positive_axis = int(positive_axis)
    left = truncation_amount.copy()
    right = truncation_amount.copy()
    if positive_axis is not None:
        left[positive_axis] = 0
    remaining_shape = np.asarray(grid.shape_domain) - left - right
    if np.any(remaining_shape <= 0):
        raise ValueError(Errors.value_error(
            f"Truncation must leave at least one point on every axis. grid.shape_domain={grid.shape_domain}, "
            f"truncation_amount={truncation_amount}, positive_axis={positive_axis}."))
    coord_list = [
        np.arange(size)*spc + ax[0] + amount_left*spc
        for size,spc,ax,amount_left in zip(remaining_shape,grid.spacing,grid.axes,left)
    ]
    truncated_grid = UniformGridFcts(*coord_list,dtype=grid.dtype,shape_codomain=grid.shape_codomain)
    pad_op = PaddingOperator(
        truncated_grid,truncation_amount,
        positive_axis=positive_axis)
    return pad_op.adjoint

##################### implementation of the convolution operator #######################################

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
        - Either: An numpy array, the Fourier transform of the convolution kernel.
          Case of scalar domain, i.e., domain.shape_codomain=(): 
            The number of dimensions should be d=len(domain.shape_domain). 
            If grid is real, the size of the last dimension is about half of that of grid corresponding to the frequency grid in rfft.
            The dimensions of the axes, which are not convolution axes should be 1, the dimensions of the others 
            should correspond to the corresponding dimensions of grid 
          Case of vector-valued domains and kernel_matrix_shape is None: 
            The convolution is applied component-wise, i.e. the vector axes are treated as stack axes.
            fourier_multiplier should have d+1 dimensions, the last one being 1. 
          Case of vector-valued domains and kernel_matrix_shape is not None:
            The number of dimensions has to be d+2, and the last two dimensions must agree with 
            kernel_matrix_shape. 
        - Or: a function taking d real values (frequencies) and returning a real or complex number.
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

    def __init__(self, grid, fourier_multiplier, pad_amount=0,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None,kernel_matrix_shape=None):
        if not isinstance(grid,UniformGridFcts):
            raise TypeError(Errors.type_error(f'grid must be of type UniformGridFcts. Got {grid}'))
        if kernel_matrix_shape is not None:
            if (not isinstance(kernel_matrix_shape,(tuple,list)) or len(kernel_matrix_shape) != 2
                    or not all(isinstance(size,(int,np.integer)) and size > 0 for size in kernel_matrix_shape)):
                raise ValueError(Errors.value_error(
                    f"kernel_matrix_shape must be a pair of positive integers. Got {kernel_matrix_shape}."))
            kernel_matrix_shape = tuple(int(size) for size in kernel_matrix_shape)
            if len(grid.shape_codomain) != 1 or grid.shape_codomain[0] != kernel_matrix_shape[1]:
                raise ValueError(Errors.value_error(
                    f'grid.shape_codomain must be ({kernel_matrix_shape[1]},) for kernel_matrix_shape '
                    f'{kernel_matrix_shape}. Got {grid.shape_codomain}.'))
        self.grid = grid
        ndim = grid.ndim_domain
        if convolution_axes is None:
            self.convolution_axes = np.arange(ndim,dtype=int)
        else:
            convolution_axes = np.asarray(convolution_axes)
            if convolution_axes.ndim != 1 or convolution_axes.size == 0 or not np.issubdtype(convolution_axes.dtype,np.integer):
                raise TypeError(Errors.type_error(
                    f'convolution_axes must be a non-empty one-dimensional array of integers. Got {convolution_axes}.'))
            self.convolution_axes = convolution_axes.astype(int,copy=False)
            if np.any(self.convolution_axes < 0) or np.any(self.convolution_axes >= ndim):
                raise ValueError(Errors.value_error(
                    f"Invalid axis specified: {self.convolution_axes}. Must be within [0, {ndim})."))
            if len(self.convolution_axes) != len(set(self.convolution_axes.tolist())):
                raise ValueError(Errors.value_error(f"Axes contain duplicates: {self.convolution_axes}"))
        self.kernel_matrix_shape = kernel_matrix_shape

        # Array containing the numbers of axes that are not convolution axes.
        stack_ndim = grid.ndim_domain if grid.shape_codomain == () or kernel_matrix_shape is not None else grid.ndim
        self.stackaxes = np.asarray(
            sorted(set(range(stack_ndim)) - set(self.convolution_axes.tolist())),dtype=int)
        if not callable(fourier_multiplier) and not isinstance(fourier_multiplier,np.ndarray):
            raise TypeError(Errors.type_error('fourier_multiplier must be callable or a numpy array'))

        def normalize_amount(amount,name,allow_negative=False,none_allowed=False):
            if amount is None:
                if none_allowed:
                    return None
                amount = 0
            if isinstance(amount,(int,np.integer)):
                values = np.zeros(grid.ndim,dtype=int)
                values[self.convolution_axes] = int(amount)
            else:
                try:
                    values = np.asarray(amount)
                except (TypeError,ValueError) as error:
                    raise TypeError(Errors.type_error(
                        f'{name} must be an integer or one-dimensional array-like. Got {amount}.')) from error
                if values.ndim != 1 or not np.issubdtype(values.dtype,np.integer):
                    raise TypeError(Errors.type_error(
                        f'{name} must contain integers. Got {amount}.'))
                if values.shape == (grid.ndim_domain,):
                    values = np.concatenate((values,np.zeros(grid.ndim_codomain,dtype=int)))
                elif values.shape != (grid.ndim,):
                    raise ValueError(Errors.value_error(
                        f'{name} must have length grid.ndim_domain={grid.ndim_domain} or grid.ndim={grid.ndim}. '
                        f'Got shape {values.shape}.'))
                values = values.astype(int,copy=False)
            if not allow_negative and np.any(values < 0):
                raise ValueError(Errors.value_error(f'{name} must be non-negative. Got {values}.'))
            non_convolution_domain_axes = np.asarray(
                sorted(set(range(grid.ndim_domain))-set(self.convolution_axes.tolist())),dtype=int)
            if (non_convolution_domain_axes.size and np.any(values[non_convolution_domain_axes] != 0)):
                raise ValueError(Errors.value_error(
                    f'{name} must be 0 for non-convolution axes {non_convolution_domain_axes}. Got {values}.'))
            if grid.ndim_codomain and np.any(values[grid.ndim_domain:] != 0):
                raise ValueError(Errors.value_error(f'{name} must be 0 on codomain dimensions. Got {values}.'))
            return values

        self.pad_amount = normalize_amount(pad_amount,'pad_amount')
        self.Fourier_truncation_amount = normalize_amount(
            Fourier_truncation_amount,'Fourier_truncation_amount',allow_negative=True,none_allowed=True)

        codomain = grid if kernel_matrix_shape is None else grid.vector_valued_space((kernel_matrix_shape[0],))

        self.kwargs = {'pad_amount' : self.pad_amount.copy(),
                       'pad_value' : pad_value,
                       'Fourier_truncation_amount' : (None if self.Fourier_truncation_amount is None
                                                       else self.Fourier_truncation_amount.copy()),
                       'convolution_axes' : self.convolution_axes.copy(),
                       'kernel_matrix_shape' : self.kernel_matrix_shape}

        freq_slice = (self.convolution_axes,  *tuple(slice(None) if i in self.convolution_axes else slice(0, 1) for i in np.arange(ndim)))
        if np.all(self.pad_amount ==0) and self.Fourier_truncation_amount is None:
            ft = FourierTransform(grid,axes=self.convolution_axes)
            if kernel_matrix_shape is None:
                ft_codomain = ft
            else:
                ft_codomain = FourierTransform(codomain,axes=self.convolution_axes)
            self._frqs = np.asarray(ft.codomain.coords)
            if grid.shape_codomain!=() and kernel_matrix_shape is None:
                self._frqs = self._frqs[...,np.newaxis]
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
        elif self.Fourier_truncation_amount is None: 
            pad_op = PaddingOperator(grid,self.pad_amount,pad_value=pad_value)
            pad_op_codomain = PaddingOperator(codomain,self.pad_amount,pad_value=pad_value)
            ft = FourierTransform(pad_op.codomain,axes=self.convolution_axes)
            if self.kernel_matrix_shape is None:
                ft_codomain = ft
            else:
                ft_codomain = FourierTransform(pad_op_codomain.codomain,axes=self.convolution_axes)
            self._frqs = np.asarray(ft.codomain.coords)
            if grid.shape_codomain!=() and kernel_matrix_shape is None:
                self._frqs = self._frqs[...,np.newaxis]
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
            pad_op = PaddingOperator(grid,self.pad_amount,pad_value=pad_value) 
            ft = FourierTransform(pad_op.codomain,axes=self.convolution_axes,centered=True)
            trunc_op = TruncationOperator(ft.codomain,
                                          np.maximum(0,self.Fourier_truncation_amount),
                                          positive_axis=(self.convolution_axes[-1]
                                                         if np.issubdtype(grid.dtype,np.floating) else None))
            if np.min(self.Fourier_truncation_amount)<0:
                pad_op2 = PaddingOperator(trunc_op.codomain,
                                         pad_amount=-np.minimum(0,self.Fourier_truncation_amount),
                                         positive_axis=(self.convolution_axes[-1]
                                                        if np.issubdtype(grid.dtype,np.floating) else None))
                trunc_op = pad_op2 * trunc_op
            self._frqs = np.asarray(trunc_op.codomain.coords)
            if grid.shape_codomain!=() and kernel_matrix_shape is None:
                self._frqs = self._frqs[...,np.newaxis]            
            if callable(fourier_multiplier):
                self._otf = fourier_multiplier(*self._frqs[freq_slice])
            else:
                self._otf = fourier_multiplier
            shape_out,shape_in = list(trunc_op.codomain.shape), list(trunc_op.domain.shape)
            if np.issubdtype(grid.dtype,np.floating):
                last_conv_axis = self.convolution_axes[-1]
                if grid.shape[last_conv_axis]%2==0:
                    shape_out[last_conv_axis] = 2*(shape_out[last_conv_axis]-1)
                    shape_in[last_conv_axis] = 2*(shape_in[last_conv_axis]-1)
                else:
                    shape_out[last_conv_axis] = 2*shape_out[last_conv_axis]-1
                    shape_in[last_conv_axis] = 2*shape_in[last_conv_axis]-1
            fac = np.sqrt(np.prod(shape_out)/np.prod(shape_in))
            if self.kernel_matrix_shape is None:
                multiplier = PtwMultiplication(trunc_op.codomain, np.broadcast_to(fac*self._otf,trunc_op.codomain.shape))
            else:
                extended_shape = trunc_op.codomain.shape_domain+self.kernel_matrix_shape                
                multiplier = PtwMatrixVectorMultiplication(trunc_op.codomain, np.broadcast_to(fac*self._otf,extended_shape)) 
            Fourier_codomain =trunc_op.codomain
            if np.issubdtype(grid.dtype,np.floating):
                last_conv_axis = self.convolution_axes[-1]
                axes = [np.array(axis,copy=True) for axis in Fourier_codomain.axes]
                N = pad_op.codomain.shape_domain[last_conv_axis]
                last_axis = axes[last_conv_axis]
                if N%2==0:
                    axes[last_conv_axis] = np.append(-last_axis[::-1],last_axis[1:-1])
                else:
                    axes[last_conv_axis] = np.append(-last_axis[-1:0:-1],last_axis)
                Fourier_codomain = UniformGridFcts(
                    *axes,dtype=complex,shape_codomain=Fourier_codomain.shape_codomain)
            new_coords = FourierTransform.frequencies(Fourier_codomain,centered=True, axes=self.convolution_axes)
            codomain = UniformGridFcts(*new_coords, 
                                       dtype=grid.dtype,
                                       shape_codomain=grid.shape_codomain)
            if kernel_matrix_shape is not None:
                codomain = codomain.vector_valued_space((kernel_matrix_shape[0],))
            ft_codomain = FourierTransform(codomain,axes=self.convolution_axes,centered=True)
            if ft_codomain.codomain != multiplier.codomain:
                raise RuntimeError(f'The codomain of the multiplier and the codomain of the Fourier Transform do not match!. {ft_codomain.codomain.shape},{multiplier.codomain.shape}')
            super().__init__(ft_codomain.adjoint,multiplier,trunc_op,ft,pad_op)

    @property
    def freqs(self):
        """coordinates in Fourier space"""
        return self._frqs    
    
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

##################################### Differential operators #######################################

class Derivative(ConvolutionOperator):
    """
    Partial derivative of arbitrary order implemented by trigonometric differentiation.

    Parameters
    ----------
    grid: regpy.vecsps.UniformGridFcts
        The Grid.
    order: tuple of integer
        The order of the differential operator, e.g. for a 2D grid (1,1) for the mixed x-y derivative 
        and (2,0) for the second derivative w.r.t. x
    pad_amount
        See Convolution Operator.
    pad_value
        See Convolution Operator.
    Fourier_truncation_amount
        See Convolution Operator.
    convolution_axes
        See Convolution Operator
    """
    def __init__(self,grid, order, pad_amount=0,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None):
        if not (isinstance(order,tuple) and (len(order)==grid.ndim_domain if convolution_axes is None else len(convolution_axes)) and all(isinstance(val,int) for val in order)):
            raise ValueError(Errors.value_error(f'order must be a d tuple of ints. Got {order}.'))
        super().__init__(grid,
                            lambda *x : prod((2j*np.pi*y)**k for y,k in zip(x,order)),
                            pad_amount=pad_amount,pad_value=pad_value,
                            Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes
                            )

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

    For vector-valued spaces, this is the vector-Laplacian. The vector axes may either be treated
    as stack axes by setting kernel_matrix_shape =None or as vector_axis by setting kernel_matrix_shape =(d,d) 
    if grid.codomain_shape = (d,). In the latter case the convolution kernel is a diagonal matrix. The difference is only relevant for operators like composition or functional_calculus. 

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
    kernel_matrix_shape:  [optional, default: None] None or pair of integers
        See description above
    """
    def __init__(self,grid, pad_amount=0,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None,kernel_matrix_shape=None):
        if grid.shape_codomain == () or kernel_matrix_shape is None:
            if  grid.shape_codomain == () and kernel_matrix_shape is not None:
                raise ValueError(Errors.value_error("For scalar domain kernel_matrix_shape must be None"))
            super().__init__(grid,
                            lambda *x : -sum((2*np.pi*y)**2 for y in x),
                            pad_amount=pad_amount,pad_value=pad_value,
                            Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes
                            )
        else:
            d = grid.shape_codomain[0]
            if len(grid.shape_codomain)!=1:
                raise ValueError(Errors.value_error('Can only treat codomain shapes of length 1.'))
            if kernel_matrix_shape != (d,d):
                raise ValueError(Errors.value_error('kernel_matrix_shape must be None or (d,d) with d the size of the codomain.'))
            super().__init__(grid,
                            lambda *x : _Lap_in_FD(*x,dim_codomain=grid.shape_codomain[0]),
                            kernel_matrix_shape=kernel_matrix_shape,
                            pad_amount=pad_amount,pad_value=pad_value,
                            Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes
                            )

def gradient(grid,pad_amount=0,pad_value=0.,Fourier_truncation_amount=None,convolution_axes=None):
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

def divergence(grid,pad_amount=0,pad_value=0.,Fourier_truncation_amount=None,convolution_axes=None):
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

def curl(grid,pad_amount=0,pad_value=0.,Fourier_truncation_amount=None,convolution_axes=None):
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

def BesselPotential(grid,s, convolution_axes=None):
    r"""Differential operator :math:`(1-\Delta)^{s/2}`.
    """
    Lap = Laplacian(grid,convolution_axes=convolution_axes)
    return Lap.functional_calculus(lambda t: (1-t)**(s/2.))

###################### other convolution operators ##########################################

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
    def __init__(self,grid, shift,pad_amount=0,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None):
        super().__init__(grid,
                        lambda *x : np.exp(sum(2j*np.pi*sh*y for y,sh in zip(x,shift))),
                        pad_amount=pad_amount,pad_value=pad_value,
                        Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes
                        )  

def GaussianBlur(grid,sigma=1.,pad_amount=0,pad_value=0.,
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
    def __init__(self,grid,a,pad_amount=0,pad_value=0.,
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
    def __init__(self,grid,pad_amount=0,pad_value=0.,
                 Fourier_truncation_amount=None,convolution_axes=None):
        super().__init__(grid,lambda *freqs: 1.,pad_amount=pad_amount,pad_value=pad_value,
                        Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes)

def FresnelPropagator(grid,fresnel_number, pad_amount=0,pad_value=0.,
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
        raise ValueError(Errors.value_error("The grid for a Fresnel propagation needs to be complex!"))
    Lap = Laplacian(grid,pad_amount=pad_amount,pad_value=pad_value,
                        Fourier_truncation_amount=Fourier_truncation_amount,convolution_axes=convolution_axes)
    return Lap.functional_calculus(lambda t: np.exp(1j / (4*np.pi* fresnel_number) * t))

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
        from scipy.special import hankel1, jv as besselj         
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
