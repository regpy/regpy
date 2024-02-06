import numpy as np

from regpy.operators import Exponential, FourierTransform, Ptw_Multiplication, RealPart, SquaredModulus, Operator
from regpy.vecsp import UniformGrid

class PaddingOperator2D(Operator):
    r"""Operator that implements zero-padding for 2-dimensional numpy arrays.

    Parameters
    ----------
    grid : regpy.vecsp.UniformGrid
        The domain on which the operator is defined.
    pad_top=,pad_bottom,pad_left,pad_right: amount of padding in different directions

    Returns
    -------
    A wrapper of the np.pad function
    """

    def __init__(self,grid, pad_amount = None):
        assert isinstance(grid, UniformGrid)
        N1,N2 = grid.shape
        if pad_amount is None:
            pad_amount = ((0,0),(0,0))
        (self.pad_top,self.pad_bottom), (self.pad_left,self.pad_right) = pad_amount
        padded_grid = UniformGrid(np.arange(N1+self.pad_top+self.pad_bottom)*grid.spacing[0] + grid.axes[0][0] - self.pad_bottom*grid.spacing[0],
                        np.arange(N2+self.pad_left+self.pad_right)*grid.spacing[1] + grid.axes[1][0] - self.pad_left*grid.spacing[1],
                        dtype=grid.dtype
                        )
        super().__init__(domain=grid,codomain=padded_grid,linear =True)

    def _eval(self,x):    
        return np.pad(x,((self.pad_top,self.pad_bottom),(self.pad_left,self.pad_right)),'constant')
    
    def _adjoint(self,y):
        return y[self.pad_top:None if self.pad_bottom == 0 else -self.pad_bottom, \
                 self.pad_left:None if self.pad_right == 0 else -self.pad_right]

def fresnel_propagator(domain, fresnel_number, pad_amount=((0,0),(0,0))):
    r"""Operator that implements Fresnel-propagation of 2D-arrays, which models near-field
    diffraction in the regime of the free-space paraxial Helmholtz equation.

    Parameters
    ----------
    domain : regpy.vecsp.Discretization
        The domain on which the operator is defined.
    fresnel_number : float
        Fresnel number of the imaging setup, defined with respect to the lengthscale
        that corresponds to length 1 in domain.coords. Governs the strength of the
        diffractive effects modeled by the Fresnel-propagator
    pad_amount = ((pad_top,pad_bottom),(pad_left,pad_right)): amount of padding to avoid aliasing artifacts

    Returns
    -------
    regpy.operators.Operator

    Notes
    -----
    The Fresnel-propagator \(D_F\) is a unitary Fourier-multiplier defined by

    \[
        D_F(f) = FT^{-1}(m_F \cdot FT(f))
    \]

    where \(FT(f)(\nu) = \int_{\mathbb{R}^2} \exp(-i\xi \cdot x) f(x) Dx\)
    denotes the Fourier transform and the factor \(m_F\) is defined by
    \(m_F(\xi) := \exp(-i \pi |\nu|^2 / F)\) with the Fresnel-number \(F\).
    
    It should be noted that if the grid is not dimensionless, 
    the frequency vector (here defined in units of \(1/\text{length}\) instead of \(2\pi/\text{length}\) 
    is not dimensionless either. 
    In this case, the Fresnel number is \(F = 1 / (\lambda d)\)  
    with wavelength  \(lambda\) and propagation distance \(d\).
    """

    assert domain.ndim == 2
    assert domain.is_complex

    if pad_amount is None or pad_amount == ((0,0),(0,0)):
        ft = FourierTransform(domain)
        frqs = ft.codomain.coords
        propagation_factor = np.exp(
            (-1j * np.pi / fresnel_number) * (frqs[0]**2 + frqs[1]**2)
        )
        fresnel_multiplier = Ptw_Multiplication(ft.codomain, propagation_factor)

        return ft.adjoint * fresnel_multiplier * ft
    else:
        pad_op = PaddingOperator2D(domain,pad_amount)
        ft = FourierTransform(pad_op.codomain)
        frqs = ft.codomain.coords
        propagation_factor = np.exp(
            (-1j * np.pi / fresnel_number) * (frqs[0]**2 + frqs[1]**2)
        )
        fresnel_multiplier = Ptw_Multiplication(ft.codomain, propagation_factor)

        return pad_op.adjoint * ft.adjoint * fresnel_multiplier * ft * pad_op


def xray_phase_contrast(domain, fresnel_number, absorption_fraction=0.0):
    r"""Forward operator that models X-ray phase contrast imaging, also known as in-line
    holography or X-ray propagation imaging. Maps a given 2D-image phi, that describes
    the induced phase shifts in the X-ray wave-field directly behind the imaged sample,
    onto the intensities I of the near-field diffraction pattern (also known as hologram)
    recorded by a detector at some distance behind the sample. The forward operator
    models incident X-rays as a fully coherent plane wave.

    Parameters
    ----------
    domain : regpy.vecsp.Discretization
        The domain on which the operator is defined.
    fresnel_number : float
        Fresnel number of the imaging setup, defined with respect to the lengthscale
        that corresponds to length 1 in domain.coords. Governs the strength of the
        diffractive effects.
    absorption_fraction : float
        Assumed constant ratio of X-ray absorption compared to refractive phase shifts,
        i.e. the value of the constant \(c_{\beta/\delta}\) described in Notes.
        The default value 0 corresponds to assuming a completely non-absorbing sample,
        which is often a justified assumption for objects composed only of light chemical
        elements.

    Notes
    -----
    The forward operator \(F\) of X-ray phase contrast imaging is defined by

    \[
        F(\phi) = |D_F(exp(-(i + c_{\beta/\delta}) \cdot  phi))|^2 = I
    \]

    where \(D_F\) is the Fresnel-propagator and \(c_{\beta/\delta}\) is
    a constant that parametrizes the magnitude of X-ray absorption versus X-ray
    refraction for the imaged sample ( \(c_{\beta/\delta} = \beta/\delta\) ).
    """

    assert domain.ndim == 2
    assert not domain.is_complex

    # Embedding operator that interprets the real-valued input image as complex-valued arrays
    domain_complex = domain.complex_space()
    real_to_complex_op = RealPart(domain_complex).adjoint

    # Operator that maps the phase-image to the corresponding wave-field behind the object
    # phi |--> psi_0 = exp(-(1j+absorption_fraction) * phi)

    image_to_wavefield_op = Exponential(domain_complex) *Ptw_Multiplication(domain_complex, -1j - absorption_fraction)
    # Fresnel propagator: models diffractive effects as the wave-field propagates from
    # the object to the detector: psi_0 |--> psi_d = FresnelPropagator(psi_0)
    fresnel_prop = fresnel_propagator(domain_complex, fresnel_number)

    # Detection operator: Maps the wave-field psi_d at the detector onto the corresponding
    # intensities: psi_d |--> I = |psi_d|^2 (squared modulus operation that eliminates
    # phase-information)
    detection_op = SquaredModulus(domain_complex)

    # Return total operator
    return detection_op * fresnel_prop * image_to_wavefield_op * real_to_complex_op
