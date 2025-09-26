import numpy as np

from regpy import util
from regpy.vecsps import UniformGridFcts

from .base import PtwMultiplication, Operator, Composition, LinearCombination,OuterShift
from .numpy import FourierTransform

__all__ = ["PaddingOperator","TruncationOperator","ConvolutionOperator","GaussianBlur","ExponentialConvolution","FourierInterpolationOperator","FresnelPropagator"]

class PaddingOperator(Operator):
    r"""Operator that implements zero-padding for numpy arrays.

    Parameters
    ----------
    grid : regpy.vecsps.UniformGridFcts
        The domain on which the operator is defined.
    pad_amount: integer or n-tuple (n=grid.ndim) of pairs of non-negative integer determining the amount of padding
        where n is the dimension of grid. E.g., for n=2,  
        pad_amont = ((pad_top,pad_bottom),(pad_left,pad_right))
        If pad_amount is an integer, this value is used for the amount of padding in each direction

    Notes
    -----
    A wrapper of the np.pad function
    """

    def __init__(self,grid, pad_amount = None, pad_value =0.):
        if not isinstance(grid, UniformGridFcts):
            raise TypeError(f"First argument has to be of type UniformGridFcts. Was given {grid}")
        s = grid.shape
        self.ndim = grid.ndim
        if pad_amount is None:
            self.pad_amount = ((0,0),)*self.ndim
        elif isinstance(pad_amount,int):
            self.pad_amount = ((pad_amount,pad_amount),)*self.ndim
            if not pad_amount>=0:
                raise ValueError("pad_amount must be non-negative.")
        elif isinstance(pad_amount, np.ndarray):
            if not pad_amount.shape == (self.ndim,) or not pad_amount.dtype==int:
                raise ValueError(f"shape of pad_amount must be (grid.ndim,) array of ints. Got {pad_amount.shape}, {pad_amount.dtype}")
            if not np.all(pad_amount>=0):
                raise ValueError("pad_amount must be non-negative.")
            self.pad_amount = tuple((val,val) for val in pad_amount)
        else:
            raise TypeError(f"pad_amount must be None or int or np.array of ints. Got {pad_amount}")
        padded_grid = UniformGridFcts(
            *[np.arange(N+pad[0]+pad[1])*spc + ax[0] - pad[0]*spc for (N,pad,spc,ax) in zip(grid.shape,self.pad_amount,grid.spacing,grid.axes)],
            dtype = grid. dtype
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
        If truncation_amount is an integer, this value is used for the amount of truncation in each direction

    Notes
    -----
    Returns the adjoint of a PaddingOperator
    """
    if not isinstance(grid, UniformGridFcts):
        raise TypeError(f'grid must be a UniformGridFcts. Got {grid}')
    if isinstance(truncation_amount,int):
            truncation_amount = truncation_amount * np.ones((grid.ndim,),dtype =int)
    elif isinstance(truncation_amount, np.parray):
        if not truncation_amount.shape == (grid.ndim,) or not truncation_amount.dtype==int:
            raise ValueError(f"shape of truncation_amount must be (grid.ndim,) array of ints. Got {truncation_amount.shape}, {truncation_amount.dtype}")
    else:
        raise TypeError(f"truncation_amount must be int or np.array of ints. Got {truncation_amount}")
    if not np.all(np.array(grid.shape)>2*truncation_amount):
        raise ValueError(f'Condition grid.shape>2*truncation_amount violated: {grid.shape}, {truncation_amount}')
    if not np.all(truncation_amount>=0):
        raise ValueError(f'Condition truncation_amount>=0 violated: Got {truncation_amount}')
    truncated_grid = UniformGridFcts(
            *[np.arange(N-2*trunc)*spc + ax[0] + trunc*spc for (N,trunc,spc,ax) in zip(grid.shape,truncation_amount,grid.spacing,grid.axes)],
            dtype = grid. dtype
            )
    pad_op = PaddingOperator(truncated_grid,truncation_amount)
    return pad_op.adjoint
    
class ConvolutionOperator(Composition):
    r"""Periodic convolution operator on a periodic UniformGridFcts space. 

    .. math::
        (Kf)(x) = \int_D k(x-y)f(y) dy
    
    Here D is the domain of the grid, and k and f are assumed to be periodic functions with periodicity cell D. 
    The implementation is based on the Fourier convolution formula 

    .. math::
        Kf = F^*(F(k)* F(f))

    with the Fourier transform f. 
    If grid is a real vector space, the convolution kernel k must be real-valued --  
    or equivalently, :math:`F(f)` must be symmetric w.r.t. the origin. 
    
    Parameters
    ----------
    grid : regpy.vecsps.UniformGridFcts
        The space on which the operator is defined. If it real, real-valued fft will be used, 
        otherwise complex fft   
    fourier_multiplier: (:math:`F(k)`) 
        - Either a d-dimensional numpy array, the Fourier transform of the convolution kernel 
          (If grid is real, the size of the last dimension is about half of that of grid)         
        - of a function taking d real values and returning a real or complex number
           In this case, the function is evaluated on a grid that is reciprocal to the input grid           
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
    first_conv_axis: integer, default:0
        If first_conv_axis>0, then convolution is only performed along the last (grid.ndim-first_conv_axis) axes.

    Methods: 
    functional_calculus: 
        Input: A scalar function :math:`phi`.
        Output: The functional calculus of the convolution operator at :math:`phi`, :math:`f\mapsto F^*(F(\vaphi(k))F(f))`
    composition:
        Input: Another convolution operator L with kenel l
        Output: The composition K L, a convolution operator with Fourier multiplier :math:`F(k)*F(l)`
        Note: If zero-padding or Fourier truncation are used, this is not the composition K*L (implmented in Composition), 
        but it is a valid and faster approximation of the composition of the underlying convolution operators in R^d.
    inverse:
        Output: Inverse operator, the convolution operator with Fourier multiplier :math:`F(1/k)'
        Note:  If zero-padding or Fourier truncation are used, this is not the exact inverse, 
        but an approximation of the inverse of the underlying convolution operators in R^d. 
    Linear combinations: 
    ::math::

        \alpha * K + \beta * L

    with scalars :math:`\alpha,\beta` yield convolution operators (implemented by only two Fourier transforms)
    """

    def __init__(self, grid, fourier_multiplier, pad_amount=None,pad_value=0.,
                 Fourier_truncation_amount=None,first_conv_axis=0):
        self.grid = grid
        self.kwargs = {'pad_amount' : pad_amount,
                       'pad_value' : pad_value,
                       'Fourier_truncation_amount' : Fourier_truncation_amount,
                       'first_conv_axis' : first_conv_axis}
        if not isinstance(grid,UniformGridFcts):
            raise ValueError(f"The given grid has to be a `UniformGirdFcts`, was given {grid} ")
        ndim = grid.ndim
        if pad_amount is None or np.all(pad_amount ==0):
            ft = FourierTransform(grid,axes=tuple(range(first_conv_axis,ndim)))
            self._frqs = ft.codomain.coords
            if callable(fourier_multiplier):
                self._otf = fourier_multiplier(*self._frqs)
            else:
                self._otf = fourier_multiplier
            multiplier = PtwMultiplication(ft.codomain, np.broadcast_to(self._otf,ft.codomain.shape))

            super().__init__(ft.adjoint, multiplier, ft)
        elif Fourier_truncation_amount is None: 
            pad_op = PaddingOperator(grid,pad_amount,pad_value=pad_value)
            ft = FourierTransform(pad_op.codomain,axes=tuple(range(first_conv_axis,ndim)))
            self._frqs = ft.codomain.coords
            if callable(fourier_multiplier):
                self._otf = fourier_multiplier(*self._frqs)
            else:
                self._otf = fourier_multiplier
            multiplier = PtwMultiplication(ft.codomain, np.broadcast_to(self._otf,ft.codomain.shape))
            trunc_op = TruncationOperator(ft.domain, pad_amount)

            super().__init__(trunc_op, ft.adjoint, multiplier, ft, pad_op)
        else:
            pad_op = PaddingOperator(grid,pad_amount,pad_value=pad_value) 
            ft = FourierTransform(pad_op.codomain,axes=tuple(range(first_conv_axis,ndim)),centered=True)
            trunc_op = TruncationOperator(ft.codomain,Fourier_truncation_amount)
            self._frqs = trunc_op.codomain.coords
            if callable(fourier_multiplier):
                self._otf = fourier_multiplier(*self._frqs)
            else:
                self._otf = fourier_multiplier
            fac = np.sqrt(np.prod(trunc_op.codomain.shape)/np.prod(trunc_op.domain.shape))
            multiplier = PtwMultiplication(trunc_op.codomain, np.broadcast_to(fac*self._otf,trunc_op.codomain.shape))
            frqs = FourierTransform.frequencies(trunc_op.codomain,centered=True, axes=tuple(range(first_conv_axis,ndim)))
            cd = UniformGridFcts(*frqs, dtype=complex)
            ft2 = FourierTransform(cd,axes=tuple(range(first_conv_axis,ndim)),centered=True)
            if ft2.codomain != multiplier.codomain:
                self.log.error(f"The codomain of the multiplier and the codomain of the Fourier Transform do not match! \n Please have a closer look!")
            super().__init__(ft2.adjoint,multiplier,trunc_op,ft,pad_op)

    @property
    def freqs(self):
        """coordinates in Fourier space"""
        return self._freqs    
    
    @property
    def fourier_multiplier(self):
        """Fourier transform of the convolution kernel"""
        return self._otf

    def functional_calculus(self,f):
        assert callable(f)
        return ConvolutionOperator(self.grid,f(self.fourier_multiplier),**self.kwargs)

    def composition(self,L):
        if not isinstance(L,ConvolutionOperator):
            raise TypeError(f'Argument must be a convolution operator. Got {L}')
        if not self.grid == L.grid:
            raise ValueError(f'Compositions only possible on same grid. Got {self.grid}, {L.grid}')
        if not self.kwargs == L.kwargs:
            raise ValueError(f'Keyword arguments must agree. Own: {self.kwargs} Got {L.kwargs}')    
        return ConvolutionOperator(self.grid, self._otf*L._otf)

    def inverse(self):
        return ConvolutionOperator(self.grid, 1/self._otf)


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
            assert self.grid == other.grid
            assert self.kwargs ==  other.kwargs
            return ConvolutionOperator(self.grid,self._otf + other._otf,**self.kwargs)
        elif isinstance(other, Operator):
            return LinearCombination(self, other)
        elif np.isscalar(other) or other in self.codomain:
            return OuterShift(self, other)
        else:
            return NotImplemented

    def __repr__(self):
        return util.make_repr(self, self._otf)


class Laplacian(ConvolutionOperator):
    """Laplace operator with periodic boundary conditions, implemented as convolution operator. 
    The second derivatives are computed with respect to the coordinates of the given grid. 

    Parameters:
    grid: UniformGridFcts
    pad_amount, pad_value, Fourier_truncation_amount, and first_conv_axis as in ConvolutionOperator
    """
    def __init__(self,grid, **kwargs):
        super().__init__(grid,
                        lambda *x : -sum((2*np.pi*y)**2 for y in x),
                        **kwargs
                        )  

class PeriodicShift(ConvolutionOperator):
    """Periodic shift operator on a given uniform grid, implemented as convolution operator. 
    Parameters:
    grid: UniformGridFcts 
    shift: array or tuple of length grid.ndim
       Amount by which grid functions are shifted (in units of grid)
    pad_amount, pad_value, Fourier_truncation_amount, and first_conv_axis as in ConvolutionOperator
    """
    def __init__(self,grid, shift,**kwargs):
        super().__init__(grid,
                        lambda *x : np.exp(sum(2j*np.pi*sh*y for y,sh in zip(x,shift))),
                        **kwargs
                        )  

def GaussianBlur(grid,sigma=1.,**kwargs):
    """Convolution with a Gaussian kernel
    Parameters: 
        grid: UniformGridFcts 
        sigma: scalar, default:1 
           width of the Gaussian kernel
        pad_amount, pad_value, Fourier_truncation_amount, and first_conv_axis as in ConvolutionOperator
    """
    assert np.isscalar(sigma)
    Lap = Laplacian(grid,**kwargs)
    return Lap.functional_calculus(lambda t: np.exp((sigma/2)**2 * t))

                                   
class ExponentialConvolution(ConvolutionOperator):
    r"""Convolution with an exponential function :math:`exp(-|x|_1/a)`.
    """
    def __init__(self,grid,a,**kwargs):
        super().__init__(grid,
                        lambda *x : np.prod([1/(1 + (2*np.pi*a*y)**2) for y in x],axis=0),
                        **kwargs
                        )
        
class FourierInterpolationOperator(ConvolutionOperator):
    r"""Interpolation operator implemented as Fourier multiplier with the constant 1 function, 
    using Fourier_truncation_amount to change the grid in the spatial domain.
    """
    def __init__(self,grid,**kwargs):
        super().__init__(grid,np.ones(grid.shape),**kwargs)

def FresnelPropagator(grid,fresnel_number, **kwargs):
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
    assert grid.is_complex
    Lap = Laplacian(grid,**kwargs)
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
    def __init__(self,grid, kappa, first_conv_axis=0):
        assert grid.is_complex
        if not (grid.ndim ==2 or grid.ndim==3):
            raise ValueError('PeriodicHelmholtzVolumePotential only implemented for dimensions 2 and 3.')
        self.kappa = kappa      
        self.N = grid.shape[0]
        if grid.ndim==2:
            assert grid.shape == (self.N,self.N)
            compute_kernel = self._compute_kernel_2d
        else:
            assert grid.shape == (self.N,)*3
            compute_kernel = self._compute_kernel_3d   
        assert self.N%2 == 0
        if not np.all(grid.extents == grid.extents[0]):
            raise ValueError('grid must be quadratic.')
        self.a = self.N * grid.spacing[0]/2.  # half of the periodicity length of the grid

        super().__init__(grid,
                        compute_kernel(self.kappa * self.a, grid.shape),
                        first_conv_axis=first_conv_axis
                        )

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
