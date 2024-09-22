from math import factorial
import numpy as np

from regpy.vecsps import DirectSum as DirectSumSpace 
from regpy.operators import Identity, Operator, RealPart, ImaginaryPart
from regpy.operators import PtwMultiplication, DirectSum, SquaredModulus, Exponential, Power
from regpy.operators import VectorOfOperators, MatrixOfOperators, Adjoint 
from regpy.operators.parallel_operators import ParallelVectorOfOperators
from regpy.operators.convolution import FresnelPropagator
from scipy.special import jv

def get_wave_field_reco(domain, fresnel_number,mask,sol_type = None,parallel = False):
    r"""Wavefield to measurement operator

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The domain on which the operator is defined.
    fresnel_number : float
        Fresnel number of the imaging setup, defined with respect to the lengthscale
        that corresponds to length 1 in domain.coords. Governs the strength of the
        diffractive effects modeled by the Fresnel-propagator
    sol_type : 
        TODO description. Defaults to None
    parallel : 
        If set True different components of the operator are computed in parallel. Defaults to False

    Returns
    -------
    regpy.operators.Operator
    """
    assert domain.is_complex

    fresnel_prop1 = FresnelPropagator(domain, fresnel_number)
    fresnel_prop2 = FresnelPropagator(domain, -fresnel_number)
    detection_op0 = SquaredModulus(domain)
    detection_op1 = SquaredModulus(domain)
    detection_op2 = SquaredModulus(domain)
    if parallel:
        vec = ParallelVectorOfOperators(
            [detection_op0,
            detection_op1*fresnel_prop1, 
            detection_op2*fresnel_prop2]
            ) * Exponential(domain) 
    else:
        vec = VectorOfOperators(
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

class NemitzkyOpForG(Operator):
    r"""
    Parameters: 
    ----------
    domain : regpy.vecsps.VectorSpace
        A complex regpy.vecsps.VectorSpace.
    N : int
        an integer representing the order of Bessel functions
    
    Notes:
    ----------
    Input of eval: 
    - A pair of real-valued vectors on domain. The first component represents the modulus \(|g|\) of \(g\), the second 
      component the phase \(\textrm{arg}(g)=\ln(\frac{g}{|g|})i^{-1}\). 

    Output of eval: 
    - A complex vector of the same size with entries 
            \[J_{N}(2|g|) * e^{i N \textrm{arg}(g)}\].
    """
    def __init__(self, N, domain):
        assert domain.is_complex
        rdomain = domain.real_space()
        self.N =N
        """An integer representing the order of Bessel functions."""
        super().__init__(DirectSumSpace(rdomain,rdomain), domain)

    def _eval(self, x, differentiate=False,adjoint_derivative=False):
        abs_g,arg_g = self.domain.split(x)
        if differentiate or adjoint_derivative:
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

class PtwDividedBessel(Operator):
    r"""
    Parameters: 
    -----------
    domain : regpy.vecsps.VectorSpace
        A real regpy.vecsps.VectorSpace
    N : int 
        an integer representing the order of Bessel functions

    Notes:
    ----------
    Input of eval: 
    - A real-valued vector r on domain.  

    Output of eval: 
    - A real vector of the same size with entries

	.. math::
		J_N(2r)r^{-N}.

    """    

    def __init__(self, N,domain):
        assert not domain.is_complex
        self.N =N
        """An integer representing the order of Bessel functions."""
        super().__init__(domain, domain)

    def _eval(self, r, differentiate=False, adjoint_derivative=False):
        N= self.N
        absN = np.absolute(N)
        jv2r = jv(N,2*r)
        # jv2r_at_null is the continuous extension of r|->jv(N,2*r)/r**N at r=0 
        jv2r_at_null = 1./(factorial(abs(N)))
        if N<0 and (N%2)==1:
             jv2r_at_null = - jv2r_at_null

        if differentiate or adjoint_derivative:
            mask = np.abs(r)>10**(-16./(absN+1))
            rmask = r[mask]
            self._factor = np.zeros_like(r)
            self._factor[mask]= (rmask*(jv(N-1,2*rmask)-jv(N+1,2*rmask)) - absN* jv2r[mask])/rmask**(absN+1) 
            self._factor[~mask] = 0.   
        result = np.zeros_like(r)
        mask = np.abs(r)>10**(-12./absN) 
        result[mask] = jv2r[mask]/r[mask]**absN
        result[~mask] = jv2r_at_null
        return result

    def _derivative(self, dr):
        return self._factor * dr

    def _adjoint(self, y):
        return self._factor * y

class ComplexNemitzkyOpForG(Operator):
    r"""
    Parameters: 
    ----------
    domain : regpy.vecsps.VectorSpace
        A complex regpy.vecsps.VectorSpace
    N : int
        an integer representing the order of Bessel functions

    Notes:
    ----------
    Input of eval: 
    - A complex vector g on domain. 

    Output of eval: 
    - A complex vector of the same size as g with entries 

	.. math::
        J_N(2|g|)  \exp(i N \textrm{arg}(g)) = 
        \begin{cases}
        J_N(2|g|)*|g|^{-N} g^N ,\; N>=0, \\
        J_N(2|g|)|g|^{-N} * \textrm{conj}(g)^N ,\; N <0
        \end{cases}

    """
    def __init__(self, N, domain):
        assert domain.is_complex
        self.N =N
        """An integer representing the order of Bessel functions. """
        self.pow = Power(np.uintc(np.absolute(N)),domain,integer=True)
        """Power operator with Poser N."""
        self.jv_div = PtwDividedBessel(N,domain.real_space())
        super().__init__(domain, domain)

    def _eval(self, g, differentiate=False, adjoint_derivative=False): 
        if differentiate or adjoint_derivative:
            self._abs_g = np.absolute(g)
            self._dir_g = np.nan_to_num(g/self._abs_g)
            factor_pow, self._pow_lin = self.pow.linearize(g)
            self._factor_real, self._real_lin = self.jv_div.linearize(self._abs_g)
            self._factor_pow = self._real_lin(np.ones_like(self._abs_g)) * factor_pow
            if self.N>=0:
                return self._factor_real * factor_pow
            else:
                return self._factor_real * np.conjugate(factor_pow)
        else:
            if self.N>=0:
                return  self.jv_div(np.absolute(g)) * self.pow(g)
            else: 
                return   self.jv_div(np.absolute(g)) * np.conjugate(self.pow(g))

    def _derivative(self, dg):
        if self.N>=0:
            return self._factor_real * self._pow_lin(dg) \
                +  np.real(np.conjugate(self._dir_g)*dg) * self._factor_pow 
        else: 
            return self._factor_real * np.conjugate(self._pow_lin(dg)) \
                + np.real(np.conjugate(self._dir_g)*dg) * np.conjugate(self._factor_pow) 

    def _adjoint(self, y):
        if self.N>=0:
            return  self._pow_lin.adjoint(np.conjugate(self._factor_real) *y) \
                + self._dir_g*np.real(np.conjugate(self._factor_pow)*y)
        else:
            return  self._pow_lin.adjoint(self._factor_real * np.conjugate(y)) \
                + self._dir_g*np.real(self._factor_pow*y)


def get_op_g_to_data(domain, fresnel_number,pad_amount,a_psi0_multiplier, \
     g_is_complex=False,N=1,list_of_filters=None,parallel = False):
    """Constructs full PINEM measurement operator that maps g to the obtained data

    Parameters:
    ----------
    domain : regpy.vecsps.VectorSpace
        The domain on which the operator is defined.
    fresnel_number : float
        Fresnel number of the imaging setup, defined with respect to the lengthscale
        that corresponds to length 1 in domain.coords. Governs the strength of the
        diffractive effects modeled by the Fresnel-propagator
    pad_amount : ((int,int),(int,int))
        amount of padding to avoid aliasing artifacts in Fresnel-propagator.
    a_psi0_multiplier : numpy.ndarray 
        TODO _description_
    g_is_complex : bool, optional
        Whether g is complex. Defaults to False.
    N : int, optional
        If list_of_filters is not given it is replaced by `[np.arange(1,N+1),np.arange(-1,-N-1,-1)]`. Defaults to 1.
    list_of_filters : list of numpy.ndarray, optional 
        List of arrays representing modes which are incoherently superposed, 
        i.e. the squares of the propagated fields are added. Defaults to None.
    parallel : bool, optional 
        If set True different components of the operator are computed in parallel. Defaults to False.

    Returns:
    ----------
    regpy.operators.Operator
        PINEM operator that maps g to the obtained data
    """    
    assert not domain.is_complex
    cdomain = domain.complex_space()
    if list_of_filters == None:
        list_of_filters = [np.arange(1,N+1),np.arange(-1,-N-1,-1)]
    modes = set()
    for filter in list_of_filters:
        modes.update(filter)
    modes = list(modes)
    op_list = []
    for n in modes:
        if g_is_complex:
            op_list.append(
                SquaredModulus(cdomain)
                *FresnelPropagator(cdomain, fresnel_number,pad_amount=pad_amount)
                *PtwMultiplication(cdomain,a_psi0_multiplier)
                *ComplexNemitzkyOpForG(n,cdomain))
        else:
            if not n==0:
                op_list.append(
                    SquaredModulus(cdomain)
                    *FresnelPropagator(cdomain, fresnel_number,pad_amount=pad_amount)
                    *PtwMultiplication(cdomain,a_psi0_multiplier)
                    *NemitzkyOpForG(n,cdomain)
                    *DirectSum(Exponential(domain), Identity(domain)))
    if parallel:
        g_to_modes = ParallelVectorOfOperators(op_list)
    else:
        g_to_modes = VectorOfOperators(op_list)
    op_mat = []
    for fil in list_of_filters:
        op_mat.append([Identity(domain,copy=False) if n in fil else None for n in modes])
    op_mat = list(map(list, zip(*op_mat)))
    modes_to_data = MatrixOfOperators(op_mat)

    return modes_to_data*g_to_modes