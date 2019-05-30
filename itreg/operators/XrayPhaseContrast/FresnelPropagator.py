import numpy as np

from itreg.operators import NonlinearOperator, PointwiseMultiplication, FourierTransform

def FresnelPropagator(domain, Fresnel_number):
    """Operator that implements Fresnel-propagation of 2D-arrays, which models near-field
    diffraction in the regime of the free-space paraxial Helmholtz equation.

    Parameters
    ----------
    domain : :class:`~itreg.spaces.Space`
        The domain on which the operator is defined.
    Fresnel_number : float
        Fresnel number of the imaging setup, defined with respect to the lengthscale
        that corresponds to length 1 in domain.coords. Governs the strength of the
        diffractive effects modeled by the Fresnel-propagator

    Notes
    -----
    The Fresnel-propagator :math:`D_F` is a unitary Fourier-multiplier defined by

    .. math:: D_F(f) = FT^{-1}(m_F \\cdot FT(f)) 

    where :math:`FT(f)(\\nu) = \int_{\\mathbb R^2}` \exp(-i\\xi \\cdot x) f(x) Dx`
    denotes the Fourier transform and the factor :math:`m_F` is defined by
    :math:`m_F(\\xi) := \\exp(-i \\pi |\\nu|^2 / F)` with the Fresnel-number :math:`F`.
    """

    assert domain.ndim == 2
    assert domain.is_complex

    propagation_factor = np.exp((-1j*np.pi/Fresnel_number) * (domain.dualgrid.coords[0]**2 + domain.dualgrid.coords[1]**2))
    fresnel_multiplier = PointwiseMultiplication(domain.dualgrid, propagation_factor)
    ft = FourierTransform(domain)

    return ft.adjoint * fresnel_multiplier * ft
