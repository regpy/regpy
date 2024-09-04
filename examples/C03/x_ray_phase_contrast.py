from regpy.operators import Exponential, FourierTransform, RealPart, SquaredModulus

import numpy as np
import logging

from regpy.operators import Operator
from regpy.vecsps import UniformGridFcts
    
class Corr(Operator):
    
    """Maps: x -> 2* D x Cov(u) x^* D^*
    """
    
    def __init__(self, domain, codomain, fp, cov_u):
        self.cov_u=cov_u
        self.fp=fp
        self.N=cov_u.shape[0]
        self.M=int(np.sqrt(self.N))
        super().__init__(domain, codomain, linear=False)

    def _eval(self, x, differentiate=False):
        mat=self.fp*x.T.flatten()
        #mat=self.fp*x.flatten()
        if differentiate==True:
            self.mat=mat.copy()
        return 2*mat.dot(self.cov_u).dot(mat.T.conj())
    
    def _derivative(self, x):
        res=(self.fp*x.T.flatten()).dot(self.cov_u).dot(self.mat.T.conj())
        return 2*(res+res.T.conj())

    def _adjoint(self, y):
        y_adj=y+y.T.conj()
        adj=np.diagonal(self.fp.T.conj().dot(y_adj).dot(self.mat).dot(self.cov_u))
        return 2*adj.reshape(self.M, self.M).T

        
def _build_fresnel_2(domain, number=complex(0,1)/80):
    
    """
    Returns the Fresnel-propagator
    """
    
    N=domain.shape[0]
    ft = FourierTransform(domain, centered=True)
    frqs = ft.codomain.coords*0.5*np.pi*N
    propagation_factor = np.exp((-number * (frqs[0]**2 + frqs[1]**2)))
    fresnel_multiplier = Ptw_Multiplication(ft.codomain, propagation_factor)
    return ft.adjoint * fresnel_multiplier * ft


class fresnel_prop(Operator):
    
    """
    Evaluation of Fresnelpropagator
    """
    
    def __init__(self, domain, number=complex(0,1)/80):
        self.N=domain.shape[0]
        self.number=number
        frqs = domain.coords*0.5*np.pi*self.N
        self.propagation_factor = np.fft.fftshift(np.exp((-number * (frqs[0]**2 + frqs[1]**2))))
        super().__init__(domain, domain, linear=True)
        
    def _eval(self, x):
        x=x.reshape(self.domain.shape)
        return np.fft.fftshift(np.fft.ifft2(self.propagation_factor*np.fft.fft2(np.fft.fftshift(x))))
    
    def _adjoint(self, y):
        return np.fft.ifftshift(np.fft.ifft2(self.propagation_factor.T.conj()*np.fft.fft2(np.fft.fftshift(y))))
        
        
"""class small_rank_basis(Operator):
    
    def __init__(self, random_coeffs, codomain):
        self.random_coeffs=random_coeffs
        domain=UniformGridFcts(len(self.random_coeffs), dtype=complex)
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        vec=self.codomain.zeros().flatten()
        vec[self.random_coeffs]=x
        return vec.reshape(self.codomain.shape)
    
    def _adjoint(self, y):
        return y.flatten()[self.random_coeffs]"""
    
    
class Ptw_Multiplication(Operator):
    """A multiplication operator by a constant factor.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discretization
    factor : array-like
        The factor by which to multiply. Can be anything that can be broadcast to `domain.shape`.
    """
    def __init__(self, domain, factor):
        factor = np.asarray(factor)
        # Check that factor can broadcast against domain elements without
        # increasing their size.
        if domain:
            factor = np.broadcast_to(factor, domain.shape)
            assert factor in domain
        self.factor = factor
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        return self.factor * x

    def _adjoint(self, x):
        if self.domain.is_complex:
            return np.conj(self.factor) * x
        else:
            return self.factor * x
        
        
class Tau(Operator):
    
    """Mapping of A_{ijk} -> A_{ijk} A^*_{ijl}
    """
    
    def __init__(self, domain, codomain):
        self.N=domain.shape[0]
        self.k=domain.shape[-1]
        super().__init__(domain, codomain, linear=False)
        
    def _eval(self, A, differentiate=True):
        if differentiate:
            self.A=A
        return A.reshape(self.N, self.N, self.k, 1)*A.conj().reshape(self.N, self.N, 1, self.k)
    
    def _derivative(self, B):
        first=self.A.reshape(self.N, self.N, self.k, 1)*B.conj().reshape(self.N, self.N, 1, self.k)
        second=B.reshape(self.N, self.N, self.k, 1)*self.A.conj().reshape(self.N, self.N, 1, self.k)
        return first+second
    
    def _adjoint(self, C):
        second_adj=np.sum(C*self.A.reshape(self.N, self.N, 1, self.k), axis=-1)
        first_adj=np.sum(C*self.A.conj().reshape(self.N, self.N, self.k, 1), axis=-2).conj()
        return first_adj+second_adj
    
class Reshape(Operator):
    
    """Reshaping an operator"""
    
    def __init__(self, domain, codomain):
        assert np.prod(domain.shape)==np.prod(codomain.shape)
        
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        return x.reshape(self.codomain.shape)
    
    def _adjoint(self, y):
        return y.reshape(self.domain.shape)
    
class Real_to_complex(Operator):
    
    def __init__(self, domain, codomain):
        assert codomain.dtype==complex
        assert domain.dtype!=complex
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        return x[0]+complex(0,1)*x[1]
    
    def _adjoint(self, y):
        res=self.domain.zeros()
        res[0]=y.real
        res[1]=y.imag
        return res
        
    
    
    
class Theta:
    
    """Mapping of DxE -> D E^*, we just apply _deriv_adjoint and _eval_adjoint
    """
    
    def __init__(self, N, k):
        self.N=N
        self.k=k
        
    """def _deriv_adjoint(self, D, E, dD, dE):
        D=D.reshape(self.N**2, self.k*2)
        E=E.reshape(self.N**2, self.k*2)
        dD=dD.reshape(self.N**2, self.k*2)
        dE=dE.reshape(self.N**2, self.k*2)
        first=np.dot(D, dE.T.conj().dot(E))+np.dot(dD, E.T.conj().dot(E))
        second=np.dot(dE, D.T.conj().dot(D))+np.dot(E, dD.T.conj().dot(D))
        return first.reshape(self.N, self.N, self.k, self.k), second.reshape(self.N, self.N, self.k, self.k)"""
    
    def _deriv_adjoint(self, D, E, dDE):
        dD=dDE[0, ...]
        dE=dDE[1, ...]
        first=self._backprop(D, E, D, dE)
        second=self._backprop(D, E, dD, E)
        return first+second
    
    def _eval_adjoint(self, D, E, C, k_C=None):
        C_1=C[0, ...]
        C_2=C[1, ...]
        return self._backprop(D, E, C_1, C_2, k_G=k_C)
    
    def _backprop(self, D, E, G_1, G_2, k_G=None):
        if k_G is None:
            k_G=self.k**2
        D=D.reshape(self.N**2, self.k**2)
        E=E.reshape(self.N**2, self.k**2)
        G_1=G_1.reshape(self.N**2, k_G)
        G_2=G_2.reshape(self.N**2, k_G)
        first=np.dot(G_1, G_2.T.conj().dot(E))
        second=np.dot(G_2, G_1.T.conj().dot(D))
        res=np.zeros((2, self.N, self.N, self.k, self.k), dtype=complex)
        res[0, ...]=first.reshape(self.N, self.N, self.k, self.k)
        res[1, ...]=second.reshape(self.N, self.N, self.k, self.k)
        return res
    
    
class Theta_2(Operator):
    
    """Mapping of DxE -> D E^*, we just apply _deriv_adjoint and _eval_adjoint
    """
    
    def __init__(self, domain, codomain, N, k):
        self.N=N
        self.k=k
        super().__init__(domain, codomain, linear=False)
    
    def _deriv_adjoint(self, DE, dDE):
        D=DE[0, ...]
        E=DE[1, ...]
        dD=dDE[0, ...]
        dE=dDE[1, ...]
        first=self._backprop(D, E, D, dE)
        second=self._backprop(D, E, dD, E)
        return first+second
    
    def _eval_adjoint(self, DE, C, k_C=None):
        D=DE[0, ...]
        E=DE[1, ...]
        C_1=C[0, ...]
        C_2=C[1, ...]
        return self._backprop(D, E, C_1, C_2, k_G=k_C)
    
    def _backprop(self, DE, G_1, G_2, k_G=None):
        if k_G is None:
            k_G=self.k**2
        D=DE[0, ...]
        E=DE[1, ...]
        D=D.reshape(self.N**2, self.k**2)
        E=E.reshape(self.N**2, self.k**2)
        G_1=G_1.reshape(self.N**2, k_G)
        G_2=G_2.reshape(self.N**2, k_G)
        first=np.dot(G_1, G_2.T.conj().dot(E))
        second=np.dot(G_2, G_1.T.conj().dot(D))
        res=np.zeros((2, self.N, self.N, self.k, self.k), dtype=complex)
        res[0, ...]=first.reshape(self.N, self.N, self.k, self.k)
        res[1, ...]=second.reshape(self.N, self.N, self.k, self.k)
        return res
    
    
class Proj(Operator):
    
    def __init__(self, domain, codomain):
        self.N=domain.shape[0]
        self.k=domain.shape[-1]
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        res=self.codomain.zeros()
        res[0, ...]=x
        res[1, ...]=x
        return res
    
    def _adjoint(self, y):
        return y[0, ...]+y[1, ...]
    
    
class Mat(Operator):
    
    """
    Maps x-> D e^f V
    """
    
    def __init__(self, domain, codomain, Vcov, fp):
        self.N=domain.shape[0]
        self.k=codomain.shape[-1]
        self.Vcov=Vcov #[N, k]
        self.fp=fp
        super().__init__(domain, codomain, linear=False)

    def _eval(self, x, differentiate=True):
        if differentiate:
            self.x=x
        mat=self.codomain.zeros()
        for i in range(0, self.k):
            mat[:, :,  i]=self.fp(np.exp(x)*self.Vcov[:, i].reshape(self.N, self.N))
        return mat
    
    def _derivative(self, h):
        mat=self.codomain.zeros()
        for i in range(0, self.k):
            mat[:, :, i]=self.fp(np.exp(self.x)*h*self.Vcov[:, i].reshape(self.N, self.N))
        return mat
    
    def _adjoint(self, y):
        res=self.domain.zeros()
        for i in range(0, self.k):
            right=self.Vcov.T.conj()[i, :].reshape(self.N, self.N)
            left=self.fp._adjoint(y[:, :, i])
            res+=right*left*np.exp(self.x.conj())
        return res


class power(Operator):
    
    def __init__(self, domain, codomain):
        self.N=domain.shape[0]
        super().__init__(self, domain, codomain, linear=False)
        
    def _eval(self, x, differentiate=True):
        return np.sum(abs(x)**2, axis=-1)
    
    def _derivative(self, h):
        return 2*np.sum(self.x*h.conj(), axis=-1).real
    
    def _adjoint(self, y):
        return 2*y.real.reshape(self.N, self.N, 1)*self.x.conj()
  
from regpy.operators import RealPart, ImaginaryPart
    
class ReIm(Operator):
    
    def __init__(self, domain, codomain):
        assert codomain.shape==(2,)+domain.shape
        super().__init__(domain, codomain, linear=True)
        self.Re=RealPart(domain)
        self.Im=ImaginaryPart(domain)        
        
    def _eval(self, x):
        res=np.zeros(self.codomain.shape)
        res[0]=self.Re(x)
        res[1]=self.Im(x)
        return res
    
    def _adjoint(self, x):
        return self.Re.adjoint(x[0])+self.Im.adjoint(x[1])


class summation(Operator):
    
    def __init__(self, domain, codomain):
        self.N=domain.shape[0]
        self.N_b=domain.shape[-1]
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        return np.sum(x.reshape(self.N, self.N, self.N_b**2), axis=-1)
    
    def _adjoint(self, y):
        return self.domain.zeros()+y.reshape(self.N, self.N, 1, 1)
