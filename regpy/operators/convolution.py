import numpy as np

from regpy.operators import FourierTransform, PtwMultiplication, Operator, Composition
from regpy.vecsps import UniformGridFcts

class PaddingOperator(Operator):
    r"""Operator that implements zero-padding for numpy arrays.

    Parameters
    ----------
    grid : regpy.vecsps.UniformGridFcts
        The domain on which the operator is defined.
    pad_amount: n-tuple of pairs of non-negative integer determining the amount of padding
        where n is the dimension of grid. E.g., for n=2,  
        pad_amont = ((pad_top,pad_bottom),(pad_left,pad_right))

    Notes
    -----
    A wrapper of the np.pad function
    """

    def __init__(self,grid, pad_amount = None):
        assert isinstance(grid, UniformGridFcts)
        s = grid.shape
        self.ndim = grid.ndim
        if pad_amount is None:
            pad_amount = ((0,0),)*self.ndim
        self.pad_amount = pad_amount
        padded_grid = UniformGridFcts(
            *[np.arange(N+pad[0]+pad[1])*spc + ax[0] - pad[0]*spc for (N,pad,spc,ax) in zip(grid.shape,pad_amount,grid.spacing,grid.axes)],
            dtype = grid. dtype
            )
        super().__init__(domain=grid,codomain=padded_grid,linear =True)

    def _eval(self,x):    
        return np.pad(x,self.pad_amount,'constant')
    
    def _adjoint(self,y):
        ind = tuple(slice(pad[0],None if pad[1]==0 else -pad[1]) for pad in self.pad_amount)
        return y[ind]
    
class ConvolutionOperator(Composition):
    r"""Periodic convolution operator on UniformGridFcts

    If the UniformGridFcts is a real vector space, the convolution kernel must be real-valued 
    or equivalently its Fourier transform, the Fourier multiplier must be symmetric w.r.t. the origin. 
    
    Parameters
    ----------
    grid : regpy.vecsps.UniformGridFcts of arbitrary dimension d
        The space on which the operator is defined. If it real, real-valued fft will be used, 
        otherwise complex fft   
    fourier_multiplier: 
        - Either a d-dimensional numpy array, the Fourier transform of the convolution kernel 
          (If grid is real, the size of the last dimension is about half of that of grid)         
        - of a function taking d real values and returning a real or complex number
           In this case, the function is evaluated on a grid that is reciprocal to the input grid           
    pad_amount: d-tuple of pairs of integers 
        To model non-periodic convolutions, zero padding is often needed to avoid aliasing artifacts
        by periodization. Each pair of integers specifies the number of pixels to be added in each dimension. 
    first_conv_axis: integer, default:0
        If first_conv_axis>0, then convolution is only performed along the last (grid.ndim-first_conv_axis) axes.
    """

    def __init__(self, grid, fourier_multiplier, pad_amount=None,first_conv_axis=0):
        ndim = grid.ndim
        if pad_amount is None or pad_amount == ((0,0),)*ndim:
            ft = FourierTransform(grid,axes=tuple(range(first_conv_axis,ndim)))
            self._frqs = ft.codomain.coords
            if callable(fourier_multiplier):
                self._otf = fourier_multiplier(*self._frqs)
            else:
                self._otf = fourier_multiplier
            multiplier = PtwMultiplication(ft.codomain, np.broadcast_to(self._otf,ft.codomain.shape))

            super().__init__(ft.adjoint, multiplier, ft)
        else:
            pad_op = PaddingOperator(grid,pad_amount)
            ft = FourierTransform(pad_op.codomain,axes=tuple(range(first_conv_axis,ndim)))
            self._frqs = ft.codomain.coords
            if callable(fourier_multiplier):
                self._otf = fourier_multiplier(*self._frqs)
            else:
                self._otf = fourier_multiplier
            multiplier = PtwMultiplication(ft.codomain, np.broadcast_to(self._otf,ft.codomain.shape))
        
            super().__init__(pad_op.adjoint, ft.adjoint, multiplier, ft, pad_op)

    @property
    def freqs(self):
        """coordinates in Fourier space"""
        return self._freqs    
    
    @property
    def fourier_multiplier(self):
        """Fourier transform of the convolution kernel"""
        return self._otf

class GaussianBlur(ConvolutionOperator):
    r"""Convolution with the shifted Gaussian kernel .math:`exp(-((x-shift)/kernel_width)^2)`.
    For :math:`shift=0` it also represents the forward operator for the backward heat equation if 
    :math:`kernel_width= 2\sqrt{t}`.
    """
    def __init__(self,grid,kernel_width,shift=None,pad_amount= None,first_conv_axis=0):
        if shift==None:
            super().__init__(grid,
                             lambda *x : np.exp(-(np.pi*kernel_width)**2 * sum(y**2 for y in x)),
                             pad_amount=pad_amount,
                             first_conv_axis=first_conv_axis
                             )
        else:
            super().__init__(grid,
                             lambda *x : np.exp(sum(-(np.pi*kernel_width)**2*y**2 + 2*np.pi*1j*sh*y
                                                         for y,sh in zip(x,shift))),
                             pad_amount=pad_amount,
                             first_conv_axis=first_conv_axis
                            )
            
class ExponentialConvolution(ConvolutionOperator):
    r"""Convolution with an exponential function :math:`exp(-|x|_1/a)`.
    """
    def __init__(self,grid,a,pad_amount= None,first_conv_axis=0):
        super().__init__(grid,
                        lambda *x : np.prod([1/(1 + (2*np.pi*a*y)**2) for y in x],axis=0),
                        pad_amount=pad_amount,
                        first_conv_axis=first_conv_axis
                        )
            
class FresnelPropagator(ConvolutionOperator):
    r"""Operator that implements Fresnel-propagation of arrays of arbitrary dimension. 
    In 2D this models near-field diffraction in the regime of the free-space paraxial 
    Helmholtz equation.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The domain on which the operator is defined.
    fresnel_number : float
        Fresnel number of the imaging setup, defined with respect to the lengthscale
        that corresponds to length 1 in domain.coords. Governs the strength of the
        diffractive effects modeled by the Fresnel-propagator
    pad_amount = ((pad_top,pad_bottom),(pad_left,pad_right)): amount of padding to avoid aliasing artifacts

    Notes
    -----
    The Fresnel-propagator :math:`D_F` is a unitary Fourier-multiplier defined by

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

    def __init__(self,grid, fresnel_number, pad_amount=None,first_conv_axis=0):
        assert grid.is_complex
        super().__init__(grid,
                        lambda *x : np.exp((-1j * np.pi / fresnel_number) * sum(y**2 for y in x)),
                        pad_amount=pad_amount,
                        first_conv_axis=first_conv_axis
                        )
 