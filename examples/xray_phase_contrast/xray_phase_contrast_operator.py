import numpy as np

from regpy.operators import RealPart, SquaredModulus, Exponential, PtwMultiplication
from regpy.operators.convolution import FresnelPropagator
from regpy.vecsps import UniformGridFcts

def get_xray_phase_contrast(domain, fresnel_number, absorption_fraction=0.0):
    r"""Forward operator that models X-ray phase contrast imaging, also known as in-line
    holography or X-ray propagation imaging. Maps a given 2D-image phi, that describes
    the induced phase shifts in the X-ray wave-field directly behind the imaged sample,
    onto the intensities I of the near-field diffraction pattern (also known as hologram)
    recorded by a detector at some distance behind the sample. The forward operator
    models incident X-rays as a fully coherent plane wave.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
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

	.. math::
        F(\phi) = |D_F(exp(-(i + c_{\beta/\delta}) \cdot  phi))|^2 = I


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

    image_to_wavefield_op = Exponential(domain_complex) *PtwMultiplication(domain_complex, -1j - absorption_fraction)
    # Fresnel propagator: models diffractive effects as the wave-field propagates from
    # the object to the detector: psi_0 |--> psi_d = FresnelPropagator(psi_0)
    fresnel_prop = FresnelPropagator(domain_complex, fresnel_number)

    # Detection operator: Maps the wave-field psi_d at the detector onto the corresponding
    # intensities: psi_d |--> I = |psi_d|^2 (squared modulus operation that eliminates
    # phase-information)
    detection_op = SquaredModulus(domain_complex)

    # Return total operator
    return detection_op * fresnel_prop * image_to_wavefield_op * real_to_complex_op
