import numpy as np
from numpy.core.defchararray import endswith

from regpy.discrs import DirectSum
from regpy.operators import Operator, Ptw_Multiplication, RealPart, SquaredModulus
from regpy.operators import Vector_of_operators, Matrix_of_operators
from regpy.operators.fresnel import fresnel_propagator
from scipy.special import jv

class Nemitzky_op_for_g(Operator):
    """The pointwise exponential operator.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discretization.
    """
    def __init__(self, domain,N):
        assert domain.is_complex
        rdomain = domain.real_space()
        self.N =N
        super().__init__(DirectSum(rdomain,rdomain), domain)

    def _eval(self, x, differentiate=False):
        abs_g,arg_g = self.domain.split(x)
        #abs_g, arg_g = self.domain.split(x)
        if differentiate:
            self._factor_abs_g = (jv(self.N-1,2*abs_g)-jv(self.N+1,2*abs_g))*np.exp(self.N*1j*arg_g)
            self._factor_arg_g = self.N*1j*jv(self.N,2*abs_g)*np.exp(self.N*1j*arg_g)
        return jv(self.N,2*abs_g)*np.exp(self.N*1j*arg_g)

    def _derivative(self, x):
        abs_g,arg_g = self.domain.split(x)
        #abs_g = x.real
        #arg_g = x.imag
        return self._factor_abs_g * abs_g + self._factor_arg_g * arg_g

    def _adjoint(self, y):
        abs_res = self._factor_abs_g.real * y.real + self._factor_abs_g.imag * y.imag
        arg_res = self._factor_arg_g.real * y.real + self._factor_arg_g.imag * y.imag
        return self.domain.join(abs_res,arg_res)#abs_res + 1j*arg_res #

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
#    return Vector_of_operators([[detection_op1*fresnel_prop1*mask, \
#             detection_op2*fresnel_prop2*mask]])
    return Vector_of_operators([detection_op1*fresnel_prop1*mask, detection_op2*fresnel_prop2*mask]) #, \
#   detection_op0*mask)

def PINEM_g_to_data(domain, fresnel_number,masks,A_Psi0_Multiplier,N=1):
    assert not domain.is_complex
    cdomain = domain.complex_space()
    op_list = []
    for n in range(-N,N+1):
        if not n==0:
 #           op_list.append(fresnel_propagator(cdomain, n*fresnel_number)*Nemitzky_op_for_g(cdomain,n)*masks)
            op_list.append(
                Ptw_Multiplication(cdomain,A_Psi0_Multiplier)
                *fresnel_propagator(cdomain, fresnel_number)
                *Nemitzky_op_for_g(cdomain,n)
                *masks
                )
    g_to_modes = Vector_of_operators(op_list)

    op_mat = []
    for n in range(0,N):
        op_mat.append([None,SquaredModulus(cdomain)])
    for n in range(0,N):
        op_mat.append([SquaredModulus(cdomain),None]) 
    modes_to_data = Matrix_of_operators(op_mat)

    return modes_to_data*g_to_modes