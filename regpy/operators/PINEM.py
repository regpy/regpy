import numpy as np
from numpy.core.defchararray import endswith

from regpy.discrs import DirectSum as DirectSumSpaces
from regpy.operators import CoordinateProjection, Identity, Operator, Composition, RealPart, ImaginaryPart
from regpy.operators import Ptw_Multiplication, DirectSum, SquaredModulus, Exponential, Power
from regpy.operators import Vector_of_operators, Matrix_of_operators, Adjoint 
from regpy.operators.parallel_operators import Parallel_vector_of_operators
from regpy.operators.fresnel import fresnel_propagator
from scipy.special import jv

def wave_field_reco_PINEM(domain, fresnel_number,mask,sol_type = None,parallel = False):
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
    if parallel:
        vec = Parallel_vector_of_operators(
            [detection_op0,
            detection_op1*fresnel_prop1, 
            detection_op2*fresnel_prop2]
            ) * Exponential(domain) 
    else:
        vec = Vector_of_operators(
            [detection_op0,
            detection_op1*fresnel_prop1, 
            detection_op2*fresnel_prop2]
            ) * Exponential(domain) 
    if sol_type == 'phase':
        return vec*Adjoint(ImaginaryPart(domain)) * mask
    elif sol_type == 'modulus':
        return vec*Adjoint(RealPart(domain)) * mask
    else:
        return vec * mask

class Nemitzky_op_for_g(Operator):
    """
    Parameters: 
      - domain: A complex regpy.discrs.Discretization
      - N: an integer representing the order of Bessel functions

    Input of eval: 
    - A pair of real-valued vectors on domain. The first component represents the modulus |g| of g, the second 
      component the phase arg(g)=ln(g/|g|)/i. 

    Output of eval: 
    - A complex vector of the same size with entries 
            J_N(2|g|) * exp(i N arg(g)).
    """
    def __init__(self, N, domain):
        assert domain.is_complex
        rdomain = domain.real_space()
        self.N =N
        super().__init__(DirectSumSpaces(rdomain,rdomain), domain)

    def _eval(self, x, differentiate=False):
        abs_g,arg_g = self.domain.split(x)
        if differentiate:
            self._factor_abs_g = (jv(self.N-1,2*abs_g)-jv(self.N+1,2*abs_g))*np.exp(self.N*1j*arg_g)
            self._factor_arg_g = self.N*1j*jv(self.N,2*abs_g)*np.exp(self.N*1j*arg_g)
        return jv(self.N,2*abs_g)*np.exp(self.N*1j*arg_g)

    def _derivative(self, x):
        abs_g,arg_g = self.domain.split(x)
        return self._factor_abs_g * abs_g + self._factor_arg_g * arg_g

    def _adjoint(self, y):
        abs_res = self._factor_abs_g.real * y.real + self._factor_abs_g.imag * y.imag
        arg_res = self._factor_arg_g.real * y.real + self._factor_arg_g.imag * y.imag
        return self.domain.join(abs_res,arg_res)#abs_res + 1j*arg_res #

class ptw_divided_Bessel(Operator):
    """
    Parameters: 
      - domain: A real regpy.discrs.Discretization
      - N: an integer representing the order of Bessel functions

    Input of eval: 
    - A real-valued vector r on domain.  

    Output of eval: 
    - A real vector of the same size with entries 
            J_N(2r)*r^(-N ) * g^N.
    """    

    def __init__(self, N,domain):
        assert not domain.is_complex
        self.N =N
        super().__init__(domain, domain)

    def _eval(self, r, differentiate=False):
        N= self.N
        jv2r = jv(N,2*r)
        if differentiate:
            self._factor = (r*(jv(N-1,2*r)-jv(N+1,2*r)) - N* jv2r)/r**(N+1)
        return jv2r/(r**N)

    def _derivative(self, dr):
        return self._factor * dr

    def _adjoint(self, y):
        return self._factor * y

class complex_Nemitzky_op_for_g(Operator):
    """
    Parameters: 
      - domain: A complex regpy.discrs.Discretization
      - N: an integer representing the order of Bessel functions

    Input of eval: 
    - A complex vector g on domain. 

    Output of eval: 
    - A complex vector of the same size as g with entries 
            J_N(2|g|)*|g|^(-N ) * g^N  = J_N(2|g|) * exp(i N arg(g)).
    """
    def __init__(self, N,domain):
        assert domain.is_complex
        self.N =N
        self.pow = Power(N,domain)
        self.jv_div = ptw_divided_Bessel(N,domain.real_space())
        super().__init__(domain, domain)

    def _eval(self, g, differentiate=False): 
        if differentiate:
            self._abs_g = np.absolute(g)
            self._dir_g = np.nan_to_num(g/self._abs_g)
            factor_pow, self._pow_lin = self.pow.linearize(g)
            self._factor_real, self._real_lin = self.jv_div.linearize(self._abs_g)
            self._factor_pow = self._real_lin(np.ones_like(self._abs_g)) * factor_pow
            return self._factor_real * factor_pow
        else:
            return  self.pow(g) * self.jv_div(np.absolute(g))

    def _derivative(self, dg):
        return self._factor_real * self._pow_lin(dg) \
            + self._factor_pow * np.real(np.conjugate(self._dir_g)*dg) 

    def _adjoint(self, y):
        return  self._pow_lin.adjoint(np.conjugate(self._factor_real) *y) \
            + self._dir_g*np.real(np.conjugate(self._factor_pow)*y)



def PINEM_g_to_data(domain, fresnel_number,mask,A_Psi0_Multiplier,N=1,parallel = False):
    assert not domain.is_complex
    cdomain = domain.complex_space()
    complexProjection = CoordinateProjection(cdomain,mask)
    realProjection = DirectSum(
        CoordinateProjection(domain,mask),
        CoordinateProjection(domain,mask)
        )
    maskDomain = complexProjection.codomain
    op_list = []
    for n in range(-N,N+1):
        if not n==0:
            op_list.append(
                SquaredModulus(cdomain)
                *Ptw_Multiplication(cdomain,A_Psi0_Multiplier)
                *fresnel_propagator(cdomain, fresnel_number)
                *Adjoint(complexProjection)
                *Nemitzky_op_for_g(n,maskDomain)
                *realProjection
                )
    if parallel:
        g_to_modes = Parallel_vector_of_operators(op_list)
    else:
        g_to_modes = Vector_of_operators(op_list)
    op_mat = []
    for n in range(0,N):
        op_mat.append([None,Identity(domain)])
    for n in range(0,N):
        op_mat.append([Identity(domain),None]) 
    modes_to_data = Matrix_of_operators(op_mat)

    return modes_to_data*g_to_modes

def complex_PINEM_g_to_data(domain, fresnel_number,mask,A_Psi0_Multiplier,N=1,parallel = False):
    assert not domain.is_complex
    cdomain = domain.complex_space()
    complexProjection = CoordinateProjection(cdomain,mask)
    maskDomain = complexProjection.codomain
    op_list = []
    for n in range(-N,N+1):
        if not n==0:
            op_list.append(
                SquaredModulus(cdomain)
                *Ptw_Multiplication(cdomain,A_Psi0_Multiplier)
                *fresnel_propagator(cdomain, fresnel_number)
                *Adjoint(complexProjection)
                *complex_Nemitzky_op_for_g(n,maskDomain)
                *complexProjection
                )
    if parallel:
        g_to_modes = Parallel_vector_of_operators(op_list)
    else:
        g_to_modes = Vector_of_operators(op_list)
    op_mat = []
    for n in range(0,N):
        op_mat.append([None,Identity(domain)])
    for n in range(0,N):
        op_mat.append([Identity(domain),None]) 
    modes_to_data = Matrix_of_operators(op_mat)

    return modes_to_data*g_to_modes