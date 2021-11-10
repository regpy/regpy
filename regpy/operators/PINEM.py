import numpy as numpy
from numpy.core.defchararray import endswith

from regpy.operators import Multiplication, RealPart, SquaredModulus, Vector_of_operators
from regpy.operators.fresnel import fresnel_propagator

def wave_field_reco_PINEM(domain, fresnel_number,mask):
    r"""Wavefield to measurement operator

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The domain on which the operator is defined.
    fresnel_number : float
        Fresnel number of the imaging setup, defined with respect to the lengthscale
        that corresponds to length 1 in domain.coords. Governs the strength of the
        diffractive effects modeled by the Fresnel-propagator

    Returns
    -------
    regpy.operators.Operator
    """
    assert domain.is_complex

    fresnel_prop1 = fresnel_propagator(domain, fresnel_number)
    fresnel_prop2 = fresnel_propagator(domain, -fresnel_number)
    detection_op0 = SquaredModulus(domain)
    detection_op1 = SquaredModulus(domain)
    detection_op2 = SquaredModulus(domain)

    return Vector_of_operators(detection_op1*fresnel_prop1*mask, \
        detection_op2*fresnel_prop2*mask, \
        detection_op0*mask)

