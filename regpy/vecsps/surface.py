from __future__ import annotations
import math
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from scipy.special import lpmv, factorial

from .numpy import MeasureSpaceFcts,GridFcts
from regpy.operators import Operator
from regpy.util import Errors

class S2Surface(ABC):
    """ Surfaces parameterized by the unit sphere :math:`S^2`. 
    More precisely, the surface is given by :math:`q(S^2)` for some smooth and smoothly invertible function 
    :math:`q:S^2\to R^3`. 
    This function is prepresented by its values on a Gauss-Legendre grid of :math:`(n+1) (2n+2)` 
    quadrature points in :math:`(theta,phi)` coordinates. 

    Parameters:
    -----------
    n: int

    Properties:
    --------
    n: int
        Determines the number of quadrature points, :math:`2(n+1)**2`
    xhat: np.ndarray
        quadrature points, shape (n+1,2n+2,3)
    theta: np.ndarray
        theta coordinates of quadrature points, shape (n+1,)
    phi: np.ndarray
        phi coordinates of quadrature points, shape (2*n+2,)
    Theta,Tphi: np.nd_array
        broadcast arrays of phi and theta to shape (n+1,2n+2)
    st,ct: no.ndarray
        sin(theta) and cos(theta), rsp
    sp,cp: no.ndarray
        sin(phi) and cos(phi), rsp 
    Etheta,Ephi,Erho: np.ndarray
        standard orthonormal basis vectors in spherical coordinates on the grid of quadrature points. 
        shape (n+1,2n+2,3)
    q: np.ndarray
        values o the function :math:`q` defining the surface on the quadrature points.
        shape (n+1,2n+2,3)
    Ttheta,Tphi: np.ndarray
        derivatives of q with respect to theta and phi on the quadrature points. shape (n+1,2n+2,3)
    normal: np.ndarray
        unit normal vector on the surface.  shape (n+1,2n+2,3)
    Jacobian: nd_array
        determinant of the Jacobian.  shape (n+1,2n+2)


    """
    def __init__(self, n: int):
        self.n = n
        self._evaluate()
        self._complete_evaluation()

    class sphere_quad_data:
        """ compute data for Gauss-Legendre quadrature on grid of :math:`(n+1) (2n+2)` 
        quadrature points in :math:`(theta,phi)` coordinates. 
        For parametric surfaces these data are set up only once in the parameter vector space.
        """
        def __init__(self, n:int):
            sh = (n+1,2*n+2)
            # Gauss–Legendre in μ = cos(θ)
            mu, self._alpha = np.polynomial.legendre.leggauss(n+1)
            self._theta = np.arccos(mu[::-1])               # θ ∈ (0, π)
            self._phi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)        
            self._st = np.sin(self._theta)
            self._ct = np.cos(self._theta)
            self._sp = np.sin(self._phi)
            self._cp = np.cos(self._phi)
            self._Erho   = np.stack((np.outer(self._st,self._cp), np.outer(self._st,self._sp), 
                                     np.broadcast_to(self._ct[:,None],sh)),axis=2)
            self._Etheta = np.stack((np.outer(self._ct,self._cp), np.outer(self._ct,self._sp),
                                     np.broadcast_to(-self._st[:,None],sh)),axis=2)
            self._Ephi   = np.stack((np.broadcast_to(-self._sp[None,:],sh), 
                                     np.broadcast_to(self._cp[None,:],sh), 
                                     np.zeros(sh)),axis=2)

    @property
    def n(self)->int:
        return self._n
    @n.setter
    def n(self,n_new:int):
        self._n = n_new
        dat = S2Surface.sphere_quad_data(n_new)
        self.__dict__.update(dat.__dict__)

    @property
    def grid_shape(self)->tuple[int]:
        return (self.n+1,2*self.n+2)
    @property
    def theta(self)->np.ndarray:
        return self._theta
    @property
    def phi(self)->np.ndarray:
        return self._phi
    @property
    def xhat(self)->np.ndarray:
        return self._Erho
    @property
    def theta_coo(self)->np.ndarray:
        return np.broadcast_to(self._theta[:,None],self.grid_shape)
    @property
    def phi_coo(self)->np.ndarray:
        return np.broadcast_to(self._phi[None,:],self.grid_shape)        
    @property
    def quadweights(self)->np.ndarray:
        return np.broadcast_to((np.pi/(self.n+1))*self._alpha[:,None],self.grid_shape)
    @property
    def st(self)->np.ndarray:
        return self._st
    @property
    def ct(self)->np.ndarray:
        return self._ct
    @property
    def sp(self)->np.ndarray:
        return self._sp
    @property
    def cp(self)->np.ndarray:
        return self._cp
    @property
    def Erho(self)->np.ndarray:
        return self._Erho
    @property
    def Ephi(self)->np.ndarray:
        return self._Ephi
    @property
    def Etheta(self)->np.ndarray:
        return self._Etheta            
    @property
    def q(self)->np.ndarray:
        return self._q   
    @property
    def Jacobian(self)->np.ndarray:
        return self._Jacobian
    @property
    def Ttheta(self)->np.ndarray:
        return self._Ttheta           
    @property
    def Tphi(self)->np.ndarray:
        return self._Tphi
    @property
    def normal(self)->np.ndarray:
        return self._normal 

    @abstractmethod
    def _evaluate(self)->None:
        """initializes q, Ttheta, Tphi"""

    def _complete_evaluation(self)->None:
        """ computes Jacobian and normal after Ttheta and Tphi have been evaluated for a subclass.
        """
        self._normal = np.cross(self.Ttheta, self.Tphi)
        self._Jacobian = np.linalg.norm(self._normal, axis=2)
        self._normal /= self._Jacobian[:,:,None]

    def grid_space(self)->GridFcts:
        """ returns the GridFcts space of the quadrature grid.
        """
        return MeasureSpaceFcts(self.quadweights)

    def area(self)->float:
        """returns area of the surface
        """
        return np.sum(self.Jacobian * self.quadweights)
    
    def center_of_gravity(self)->np.ndarray:
        """returns center of gravity of the surface, a np.ndarray of shape (3,)
        """
        return np.sum(self.q * self.Jacobian[:,:,None] * self.quadweights[:,:,None], axis=(0,1)) / self.area()
    

class StarSurface(S2Surface, ABC):
    """ star-shape S2Surface with parameterization given by
    :math:`q(xhat) = r(xhat)xhat`
    """
    def __init__(self, n:int):
        super().__init__(n)

    @abstractmethod
    def radius_and_derivatives(self)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Return r, dr/dθ, dr/dφ"""

    def _evaluate(self)->None:
        r, dr_dtheta, dr_dphi = self.radius_and_derivatives()
        self._q = r[:,:,None] * self.Erho 
        self._Ttheta = dr_dtheta[:,:,None] * self.Erho + r[:,:,None] * self.Etheta
        self._Tphi   = dr_dphi[:,:,None]   * self.Erho + r[:,:,None] * self.Ephi

class Sphere(StarSurface):
    def __init__(self, n:int, R=0.5):
        self.R = R
        super().__init__(n)

    def radius_and_derivatives(self)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        r = self.R * np.ones(self.grid_shape)
        return r, np.zeros(self.grid_shape), np.zeros(self.grid_shape)

class Ellipsoid(S2Surface):
    def __init__(self,n:int,A: float,B: float,C: float):
        self.scaling = np.broadcast_to(np.array([A,B,C])[None,None,:],(n+1,2*n+2,3))
        super().__init__(n)

    def _evaluate(self):
        self._q = self.scaling * self.Erho
        self._Ttheta = self.scaling * self.Etheta
        self._Tphi = self.scaling * self.Ephi

class Peanut(StarSurface):
    def __init__(self, n:int,alpha=0.25):
        self.alpha = alpha
        super().__init__(n)

    def radius_and_derivatives(self)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        s2 = 2*self.st*self.ct                          # sin^2(theta)
        c2 = self.ct*self.ct - self.st*self.st          # sin^2(theta)

        A = self.alpha + 1
        B = A - s2*s2                                   # = alpha + 1 - sin²(2θ)
        sqrtB = np.sqrt(B)

        C = c2 + sqrtB
        sqrtC = np.sqrt(C)

        denom = 2*np.sqrt(1+np.sqrt(A))
        r = sqrtC / denom

        # compute derivative w.r.t. theta
        ds2 = 2*c2                                      # d/dθ sin(2θ) = 2 cos(2θ)
        dc2 = -2*s2
        dB = -2*s2*ds2
        dC = dc2 + dB/(2*sqrtB)
        dr = dC/(2*sqrtC*denom)

        sh = self.grid_shape

        return np.broadcast_to(r[:,None],sh), np.broadcast_to(dr[:,None],sh), np.zeros(sh)

######################################################################

class ParameterizedSurface():
    r""" This is an abstract class for S2Surfacess parameterized by coefficients in some vector space.

    Typical forward operators are naturally defined on a shape space. In particular, they 
    are invariant under re-parametrizations of the surface. However, such shape spaces are not vector spaces, and we define forward operators on some vector space of coefficients paraemeterizing a certain class of surface (e.g., star-shape surfaces with radial functions given by spherical harminces of a cetrain degree).  

    One may think of the mapping taking coefficients to surfaces as an operator, and of the parameterized forward operators as compositions of this operator with the actual forward operator defined 
    on a shape space. However, as mentioned above, shape spaces aren't vector spaces, and it is often  
    computationally more convenient, not to implement the discrete Frechet derivative of the parameterized forward operator, but to discretize a continuous characterization of the Frechet derivative. For this reason the forward operators on shape spaces in this toolbox are not implemented as such compositions.

    Due to invariance under reparameterizations, Frechet derivatives of operators on shape spaces only 
    depend on the normal component of a perturbation of the surface, and this quantity is used 
    in our implementations of the derivatives of parameterized forward operators. More precisely, 
    these derivatives are compositions of a discretization of the continuous derivative on shape space 
    and the operator `self.der_normal`.
    """
    from regpy.operators import Operator
    def __init__(self,coeff:np.ndarray,der_normal:Operator,**kwargs):
        if not coeff in der_normal.domain:
            raise ValueError(Errors.not_a_vecsp(coeff,der_normal.domain))
        if not isinstance(der_normal.domain,ParameterizedSurfaceSpc):
            raise ValueError(Errors.not_instance(der_normal.domain,ParameterizedSurfaceSpc))
        if not isinstance(der_normal.codomain,MeasureSpaceFcts):
            raise ValueError(Errors.not_instance(der_normal.codomain,MeasureSpaceFcts))
        n = der_normal.codomain.shape[0]
        if not der_normal.codomain.shape==(n,2*n):
            raise ValueError(Errors.generic_message('codomain must have shape of length 2.'))
        self.coeff = coeff
        """The coefficient vector characterizing the surface.
        """
        self.der_normal = der_normal
        """Linear `regpy.operators.Operator` given by the inner product of the derivative of the coefficient-to-curve "operator" (i.e. the derivative of `ParameterizedCurveSpc.coeff2curve`) and the normal vector of length 1 of the curve. """


class ParameterizedSurfaceSpc(ABC):
    r""" Abstract base class of vector spaces of coefficients of `ParameterizedSurface`s. 
    """
    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def coeff2surface(self, coeff : np.ndarray, nderivs : int =0)->ParameterizedSurface:
        r"""Compute a curve for the given coefficients.
        """
        pass

class StarSHSurfaceSpc(ParameterizedSurfaceSpc,MeasureSpaceFcts):
    def __init__(self, n:int,sh_deg:int=3):
        if not isinstance(n,int) or not isinstance(sh_deg,int):
            raise TypeError(Errors.type_error())
        if sh_deg>n or n<0:
            raise ValueError(Errors.value_error(msg=f'need 0<=sh_deg<=n. {n},{sh_deg}'))
        MeasureSpaceFcts.__init__(self,measure = np.ones(((sh_deg+1)**2,),dtype=float))
        ParameterizedSurfaceSpc.__init__(self)
        self.n = n
        self.sh_deg = sh_deg
        dat = S2Surface.sphere_quad_data(n)
        self.__dict__.update(dat.__dict__)
        self._compute_SH_basis()

 
    def _compute_SH_basis(self):
        from scipy.special import sph_harm_y
        shape = (len(self._theta),len(self._phi),(self.sh_deg+1)**2)
        self.basis       = np.zeros(shape,dtype=float)
        self.basis_theta = np.zeros(shape,dtype=float)
        self.basis_phi   = np.zeros(shape,dtype=float)

        fac = np.sqrt(2)
        phi, theta = np.meshgrid(self._phi,self._theta)
        for l in range(self.sh_deg+1):
            y,yp = sph_harm_y(l,0,theta,phi,diff_n=1)
            self.basis[:,:,l*l+l] = y.real
            self.basis_theta[:,:,l*l+l] = yp[:,:,0].real
            self.basis_phi[:,:,l*l+l] = yp[:,:,1].real    
            for m in range(1,l+1):
                y,yp = sph_harm_y(l,m,theta,phi,diff_n=1)
                self.basis[:,:,l*l+l+m] = fac*y.real
                self.basis_theta[:,:,l*l+1+m] = fac*yp[:,:,0].real
                self.basis_phi[:,:,l*l+1+m] = fac*yp[:,:,1].real
                self.basis[:,:,l*l+l-m] = fac*y.imag
                self.basis_theta[:,:,l*l+1-m] = fac*yp[:,:,0].imag
                self.basis_phi[:,:,l*l+1-m] = fac*yp[:,:,1].imag          

    def coeff2surface(self,coeff:np.ndarray)->StarSHSurface:
        if len(coeff.shape)!=1 or len(coeff) != (self.sh_deg+1)**2:
            raise ValueError(f'coefficient vector must have shape (sh_deg**2,). sh_deg={self.sh_deg}, {coeff.shape}')
        return StarSHSurface(self,coeff)

    def sphere(self,radius:float=1.)->StarSHSurface:
        """ Return a StarSHSurface which is a sphere of a given radius. 
        This can be used as an initial guess.
        """
        coeff = self.zeros()
        coeff[0]=radius *np.sqrt(4*np.pi)
        return StarSHSurface(self,coeff)            


class StarSHSurface(StarSurface,ParameterizedSurface):
    """
    StarSurface with radial function given by spherical harmonics. 

    Parameters:
    -----------

    spc: StarSHSurfaceSpc
        The underlying vector space of of SH coefficients. 
    n: int
        Parameter of quadrature grid 
    coeff: np.ndarray
        coefficient vector
    """

    class der_normal_StarSHSurface(Operator):
        def __init__(self,spc:StarSHSurfaceSpc,surf:StarSHSurface):
                super().__init__(domain= spc,codomain=surf.grid_space(), linear=True)
                self.basis = spc.basis
                self.xhat_dot_normal = np.sum(surf.normal*surf.xhat,axis=2)

        def _eval(self,coeff):
            radial = np.sum(coeff[None,None,:] * self.basis,axis=2)
            radial *= self.xhat_dot_normal
            return radial

        def _adjoint(self, y):
            aux = y*self.xhat_dot_normal
            return np.sum(aux[:,:,None]*self.basis,axis=(0,1))

    def __init__(self, spc:StarSHSurfaceSpc, coeff:np.ndarray):
        """
        StarSurface with radial function r given by a linear combination of spherical harmonics 
        of some maximal degree.

        Parameters
        ----------
        sh_deg: int
            Maximum harmonic degree
        """

        if not isinstance(coeff,np.ndarray):
            raise TypeError(Errors.type_error("coeff must be a np.ndarray",coeff))
        self.spc = spc
        if not len(coeff.shape)==1 or (spc.sh_deg+1)**2 != len(coeff):
            raise ValueError(Errors.value_error(f"length of coeff must be a perfect square. Given shape: {coeff.shape}"))
        
        self.coeff=coeff
        StarSurface.__init__(self,spc.n)
        der_normal = StarSHSurface.der_normal_StarSHSurface(spc,self)
        ParameterizedSurface.__init__(self,coeff,der_normal=der_normal)

    @S2Surface.n.setter
    def n(self,n_new:int):
        self._n = n_new
        if self.spc.n!=n_new:
            raise RuntimeError(f"mismatch of spc.n and n_new. {self.spc.n},{n_new}")
        for k in ["_theta","_phi","_st","_ct","_sp","_cp","_Ephi","_Etheta","_Erho","_alpha"]:
            setattr(self, k, getattr(self.spc, k))
        #self._theta, self._phi = self.spc._theta, self.spc._phi 
        #self._st,self._ct, self._sp, self._cp =  self.spc._st,self.spc._ct, self.spc._sp, self.spc._cp
        #self._Erho, self._Etheta, self._Ephi = self.spc._Erho, self.spc._Etheta, self.spc._Ephi 

    def radius_and_derivatives(self)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Evaluate spherical harmonic field and its surface gradient

        Parameters
        ----------
        coeff : ((N+1)^2,) array
            Spherical harmonic coefficients in MATLAB ordering: index = l^2 + l + m
        """

        r      = np.sum(self.coeff[None,None,:] * self.spc.basis,axis=2)
        rtheta = np.sum(self.coeff[None,None,:] * self.spc.basis_theta,axis=2) 
        rphi   = np.sum(self.coeff[None,None,:] * self.spc.basis_phi,axis=2)

        return r, rtheta, rphi

    def field2SHcoeffs(self,u:np.ndarray)->np.ndarray:
        """ Computes the spherical harmonics coefficients of a real-valued field on the sphere given 
        at the quadrature points.

        Parameters:
        ----------
        u: np.ndarray

        Returns:
        ------------
        np.ndarray
        """
        if not isinstance(u,np.ndarray) or not np.issubdtype(u.dtype,np.floating):
            raise TypeError('u must be a real np.ndarray')
        if u.shape!=self.grid_shape:
            raise ValueError(f'shape of u does not match: {u.shape},{self.grid_shape}')
        return np.sum(u[:,:,None]*self.quadweights[:,:,None]*self.spc.basis,axis =(0,1))

"""
@dataclass
class SphericalHarmonicsCache:
    N: int
    P: list              # P[l][m+l, theta]
    dP: list             # d/dtheta P[l][m+l, theta]
    exp_imphi: np.ndarray  # shape (n_phi, 2N+1)

def radius_and_derivatives(self):
        N = self.N
        theta = self.theta
        phi = self.phi

        H = np.zeros(self.grid_shape, dtype=complex)
        dtheta_H = np.zeros_like(H)
        dphi_H = np.zeros_like(H)

        for m in range(-N, N + 1):
            res_P = np.zeros(self.n+1, dtype=complex)
            res_dP = np.zeros(self.n+1, dtype=complex) if compute_gradient else None

            for l in range(abs(m), N + 1):
                idx = l*l + l + m
                res_P += coeff[idx] * self.spc.P[l][m + l]
                res_dP += coeff[idx] * self.spc.dP[l][m + l]

            exp_mphi = self.spc.exp_imphi[:, m + N]  # shape (2n+2,)
            H += res_P[:, None] * exp_mphi[None, :]

            dtheta_H += res_dP[:, None] * exp_mphi[None, :]
            dphi_H += 1j * m * res_P[:, None] * exp_mphi[None, :]

            # dphi_H_flat = (dphi_H / sin_theta).reshape(1, -1)

        return H, dtheta_H, dphi_H

def harmonics_to_surface(
    coeff: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    Etheta: np.ndarray,
    Ephi: np.ndarray,
    *,
    N: int,
    cache: Optional[SphericalHarmonicsCache] = None,
    compute_gradient: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ###
    Parameters
    ----------
    coeff : ( (N+1)^2, ) complex
        Spherical harmonic coefficients, ordered as in MATLAB:
        index = l^2 + l + m
    theta : (n_theta,)
    phi   : (n_phi,)
    Etheta, Ephi : (3, n_theta * n_phi)
        Tangent vectors on S^2
    N : int
        Harmonic truncation
    cache : optional SphericalHarmonicsCache
    ###

    theta = theta.reshape(-1)
    phi = phi.reshape(-1)

    n_theta = theta.size
    n_phi = phi.size

    sin_theta = np.sin(theta)[:, None]

    # --------------------------------------------------
    # Precomputation (if not supplied)
    # --------------------------------------------------
    if cache is None:
        from scipy.special import lpmv

        P = []
        dP = []

        x = np.cos(theta)

        for l in range(N + 1):
            Pl = np.zeros((2*l + 1, n_theta))
            dPl = np.zeros_like(Pl)

            for m in range(-l, l + 1):
                Pl[m + l] = lpmv(m, l, x)
                # derivative w.r.t. theta
                dPl[m + l] = -np.sin(theta) * np.gradient(Pl[m + l], theta)

            P.append(Pl)
            dP.append(dPl)

        exp_imphi = np.exp(1j * phi[:, None] * np.arange(-N, N + 1))

    else:
        P = cache.P
        dP = cache.dP
        exp_imphi = cache.exp_imphi

    # --------------------------------------------------
    # Field evaluation
    # --------------------------------------------------
    H = np.zeros((n_theta, n_phi), dtype=complex)

    if compute_gradient:
        dtheta_H = np.zeros_like(H)
        dphi_H = np.zeros_like(H)

    for m in range(-N, N + 1):
        res_P = np.zeros(n_theta, dtype=complex)
        res_dP = np.zeros(n_theta, dtype=complex)

        for l in range(abs(m), N + 1):
            idx = l*l + l + m
            res_P += coeff[idx] * P[l][m + l]
            if compute_gradient:
                res_dP += coeff[idx] * dP[l][m + l]

        exp_mphi = exp_imphi[:, m + N]  # (n_phi,)

        H += res_P[:, None] * exp_mphi[None, :]

        if compute_gradient:
            dtheta_H += res_dP[:, None] * exp_mphi[None, :]
            dphi_H += (1j * m) * res_P[:, None] * exp_mphi[None, :]

    # --------------------------------------------------
    # Surface gradient
    # --------------------------------------------------
    gradS2 = None
    if compute_gradient:
        dtheta_H = dtheta_H.reshape(1, -1)
        dphi_H = (dphi_H / sin_theta).reshape(1, -1)

        gradS2 = (
            dtheta_H * Etheta
            +
            dphi_H * Ephi
        )

    return H, gradS2
"""


"""
@dataclass
class SphericalHarmonicsCache:
    N: int
    P: list              # P[l][m+l, theta]
    dP: list             # d/dtheta P[l][m+l, theta]
    exp_imphi: np.ndarray  # shape (n_phi, 2N+1)

def harmonics_to_surface(
    coeff: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    Etheta: np.ndarray,
    Ephi: np.ndarray,
    *,
    N: int,
    cache: Optional[SphericalHarmonicsCache] = None,
    compute_gradient: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ###
    Parameters
    ----------
    coeff : ( (N+1)^2, ) complex
        Spherical harmonic coefficients, ordered as in MATLAB:
        index = l^2 + l + m
    theta : (n_theta,)
    phi   : (n_phi,)
    Etheta, Ephi : (3, n_theta * n_phi)
        Tangent vectors on S^2
    N : int
        Harmonic truncation
    cache : optional SphericalHarmonicsCache
    ###

    theta = theta.reshape(-1)
    phi = phi.reshape(-1)

    n_theta = theta.size
    n_phi = phi.size

    sin_theta = np.sin(theta)[:, None]

    # --------------------------------------------------
    # Precomputation (if not supplied)
    # --------------------------------------------------
    if cache is None:
        from scipy.special import lpmv

        P = []
        dP = []

        x = np.cos(theta)

        for l in range(N + 1):
            Pl = np.zeros((2*l + 1, n_theta))
            dPl = np.zeros_like(Pl)

            for m in range(-l, l + 1):
                Pl[m + l] = lpmv(m, l, x)
                # derivative w.r.t. theta
                dPl[m + l] = -np.sin(theta) * np.gradient(Pl[m + l], theta)

            P.append(Pl)
            dP.append(dPl)

        exp_imphi = np.exp(1j * phi[:, None] * np.arange(-N, N + 1))

    else:
        P = cache.P
        dP = cache.dP
        exp_imphi = cache.exp_imphi

    # --------------------------------------------------
    # Field evaluation
    # --------------------------------------------------
    H = np.zeros((n_theta, n_phi), dtype=complex)

    if compute_gradient:
        dtheta_H = np.zeros_like(H)
        dphi_H = np.zeros_like(H)

    for m in range(-N, N + 1):
        res_P = np.zeros(n_theta, dtype=complex)
        res_dP = np.zeros(n_theta, dtype=complex)

        for l in range(abs(m), N + 1):
            idx = l*l + l + m
            res_P += coeff[idx] * P[l][m + l]
            if compute_gradient:
                res_dP += coeff[idx] * dP[l][m + l]

        exp_mphi = exp_imphi[:, m + N]  # (n_phi,)

        H += res_P[:, None] * exp_mphi[None, :]

        if compute_gradient:
            dtheta_H += res_dP[:, None] * exp_mphi[None, :]
            dphi_H += (1j * m) * res_P[:, None] * exp_mphi[None, :]

    # --------------------------------------------------
    # Surface gradient
    # --------------------------------------------------
    gradS2 = None
    if compute_gradient:
        dtheta_H = dtheta_H.reshape(1, -1)
        dphi_H = (dphi_H / sin_theta).reshape(1, -1)

        gradS2 = (
            dtheta_H * Etheta
            +
            dphi_H * Ephi
        )

    return H, gradS2
"""




"""import pyshtools as pysh
class SphericalHarmonicField:
    ###
    Represents a scalar field on the sphere expanded in spherical harmonics.
    Uses pyshtools internally for fast evaluation and gradient computation.
    ###
    def __init__(self, coeffs=None, lmax=None, normalization='schmidt'):
        ###
        Initialize the field.
        
        Parameters
        ----------
        coeffs : array_like, optional
            MATLAB-style 1D coefficients in order l^2 + l + m, including negative m.
        lmax : int
            Maximum degree of the expansion. Required if coeffs is given.
        normalization : str
            Normalization to use ('4pi', 'schmidt', or 'unnorm').
        ###
        self.normalization = normalization
        if coeffs is not None:
            assert lmax is not None, "Must provide lmax if coeffs are given"
            self.lmax = lmax
            self.coeffs = self._convert_matlab_coeffs(coeffs, lmax)
        else:
            self.lmax = 0
            self.coeffs = None

    def _convert_matlab_coeffs(self, coeffs, lmax):
        #Convert MATLAB linear coefficients to pyshtools (C_lm/S_lm) array.
        
        clm = np.zeros((2, lmax+1, lmax+1))
        for l in range(lmax+1):
            for m in range(0, l+1):
                # MATLAB linear index: l^2 + l + m
                idx_pos = l**2 + l + m
                clm[0, l, m] = coeffs[idx_pos].real  # cosine term
                clm[1, l, m] = coeffs[idx_pos].imag  # sine term
        return clm

def evaluate(self, theta, phi):
        ###
        Evaluate H(theta,phi) and gradS2(theta,phi) in Cartesian coordinates.
        theta, phi in radians
        Returns: H (lt x lp), gradS2 (3 x lt*lp)
        ###
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        lt, lp = theta.size, phi.size

        # Associated Legendre functions and derivatives
        plm = pysh.legendre.Plm(self.lmax, theta)
        dplm = pysh.legendre.PlmDeriv(self.lmax, theta)

        # exp(i*m*phi) matrix
        mvec = np.arange(-self.lmax, self.lmax+1)
        exp_imphi = np.exp(1j * np.outer(phi, mvec))  # lp x (2*lmax+1)

        # Initialize
        H = np.zeros((lt, lp), dtype=complex)
        grad_theta = np.zeros((lt, lp), dtype=complex)
        grad_phi = np.zeros((lt, lp), dtype=complex)

        # Vectorized sum over m
        for idx_m, m in enumerate(mvec):
            res = np.zeros(lt, dtype=complex)
            dres = np.zeros(lt, dtype=complex)
            for l in range(abs(m), self.lmax+1):
                c_idx = l**2 + l + m
                coef = self.coeffs[c_idx]
                res += coef * plm[l, abs(m), :]
                dres += coef * dplm[l, abs(m), :]
            exp_col = exp_imphi[:, idx_m]  # lp
            H += np.outer(res, exp_col)
            grad_theta += np.outer(dres, exp_col)
            grad_phi += np.outer(1j*m*res, exp_col)

        # Compute tangent vectors in Cartesian coordinates
        st = np.sin(theta)[:, None]
        ct = np.cos(theta)[:, None]
        sp = np.sin(phi)[None, :]
        cp = np.cos(phi)[None, :]

        # Etheta and Ephi as 3 x (lt*lp)
        Etheta = np.array([ct, np.zeros_like(ct), -st])  # along theta
        Ephi   = np.array([-sp, cp, np.zeros_like(sp)])  # along phi
        # Repeat to match grid
        Etheta = np.repeat(Etheta, lp, axis=1)
        Ephi = np.repeat(Ephi, lt, axis=1)

        # Vectorized gradS2
        gradS2 = (grad_theta.ravel()[None,:] * Etheta +
                  (grad_phi.ravel()/st.ravel())[None,:] * Ephi)

        return H.real, gradS2.real
def evaluate(self, theta, phi):
        ###
        Evaluate H(theta,phi) and gradS2(theta,phi) using full vectorization.
        theta, phi in radians
        Returns: H (lt x lp), gradS2 (3 x lt*lp)
        ###
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        lt, lp = theta.shape[0], phi.shape[0]

        # Compute associated Legendre functions
        # pyshtools: 'Plm' returns array of shape (lmax+1, lmax+1, theta.size)
        plm = pysh.legendre.Plm(self.lmax, theta)
        dplm = pysh.legendre.PlmDeriv(self.lmax, theta)

        # Build exp(i*m*phi) matrix (lp x 2*lmax+1)
        mvec = np.arange(-self.lmax, self.lmax+1)
        exp_imphi = np.exp(1j * np.outer(phi, mvec))  # shape: lp x (2*lmax+1)

        # Vectorized computation over m
        H = np.zeros((lt, lp), dtype=complex)
        grad_theta = np.zeros((lt, lp), dtype=complex)
        grad_phi = np.zeros((lt, lp), dtype=complex)

        for idx_m, m in enumerate(mvec):
            res = np.zeros(lt, dtype=complex)
            dres = np.zeros(lt, dtype=complex)
            for l in range(abs(m), self.lmax+1):
                c_idx = l**2 + l + m
                coef = self.coeffs[c_idx]
                res += coef * plm[l, abs(m), :]
                dres += coef * dplm[l, abs(m), :]
            exp_col = exp_imphi[:, idx_m]  # size lp
            H += np.outer(res, exp_col)
            grad_theta += np.outer(dres, exp_col)
            grad_phi += np.outer(1j*m*res, exp_col)

        # gradS2: combine Etheta and Ephi
        st = np.sin(theta)[:, None]
        Etheta = np.array([[1],[0],[0]])  # Placeholder; replace with actual 3D vectors
        Ephi = np.array([[0],[1],[0]])    # Placeholder; replace with actual 3D vectors
        gradS2 = (grad_theta.ravel()[None,:] * Etheta +
                  (grad_phi.ravel()/st.ravel())[None,:] * Ephi)

        return H.real, gradS2.real
    def evaluate(self, theta, phi):
        ###
        Evaluate the spherical harmonic field at points (theta, phi).
        Theta and phi are in radians.
        ###
        # Convert to grids suitable for pyshtools
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        # Make regular lat/lon grids for DH sampling
        lat = 90 - np.degrees(theta)  # colatitude -> latitude
        lon = np.degrees(phi)
        grid = pysh.expand.MakeGridDH(self.coeffs, sampling=2, lmax=self.lmax,
                                      norm=self.normalization)
        # Interpolation to arbitrary points
        from scipy.interpolate import RegularGridInterpolator
        lats = np.linspace(90, -90, grid.shape[0])
        lons = np.linspace(0, 360, grid.shape[1], endpoint=False)
        interp = RegularGridInterpolator((lats, lons), grid, bounds_error=False, fill_value=None)
        points = np.stack([lat.ravel(), lon.ravel()], axis=-1)
        values = interp(points).reshape(theta.shape)
        return values

    def gradient(self, theta, phi):
        ###
        Compute the gradient on the sphere (∇_θ, ∇_φ components).
        ###
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        # Compute theta/phi derivatives using pyshtools
        grad_theta, grad_phi = pysh.expand.SHGradDH(self.coeffs, lmax=self.lmax,
                                                    norm=self.normalization)
        # Interpolation to arbitrary points
        lat = 90 - np.degrees(theta)
        lon = np.degrees(phi)
        lats = np.linspace(90, -90, grad_theta.shape[0])
        lons = np.linspace(0, 360, grad_theta.shape[1], endpoint=False)
        interp_theta = RegularGridInterpolator((lats, lons), grad_theta, bounds_error=False, fill_value=None)
        interp_phi = RegularGridInterpolator((lats, lons), grad_phi, bounds_error=False, fill_value=None)
        points = np.stack([lat.ravel(), lon.ravel()], axis=-1)
        grad_theta_val = interp_theta(points).reshape(theta.shape)
        grad_phi_val = interp_phi(points).reshape(theta.shape)
        return grad_theta_val, grad_phi_val

    @classmethod
    def from_random(cls, lmax, seed=None, normalization='schmidt'):
        #Create a random spherical harmonic field.
        
        rng = np.random.default_rng(seed)
        coeffs = rng.standard_normal((2, lmax+1, lmax+1))
        field = cls(lmax=lmax, normalization=normalization)
        field.coeffs = coeffs
        return field
"""