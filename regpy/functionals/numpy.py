from math import inf

import numpy as np
from scipy.linalg import ishermitian

from regpy.operators import PtwMultiplication
from regpy.vecsps.numpy import *
from regpy.hilbert import L2

from .base import Functional, LinearFunctional,LinearCombination,HorizontalShiftDilation,NotInEssentialDomainError,NotTwiceDifferentiableError

__all__ = ["IntegralFunctionalBase","LppPower","L1MeasureSpace","KullbackLeibler","RelativeEntropy","Huber","QuadraticIntv","QuadraticBilateralConstraints","QuadraticLowerBound","QuadraticNonneg","QuadraticPositiveSemidef","L1Generic","TVGeneric","TVUniformGridFcts"]


class IntegralFunctionalBase(Functional):
    r"""
    This class provides a general framework for Integral functionals of the type
    
    .. math::
        F\colon X \to \mathbb{R}
        v\mapsto \Int_\Omega f(v(x),x)\mathrm{d}x

    with \(f\colon \mathbb{R}^2\to \mathbb{R})\. 

    Subclasses defining explicit functionals of this type have to implement
     * `_f` evaluation the function \(f)\
     * `_f_deriv` giving the derivative \(\partial_1 f)\
     * `_f_prox` giving the prox of \(v>->f(v,x))\
    
    since 

    .. math::
        F'[g]h = \int_\Omega h(x)(\partial_1 f)(g(x),x)

    is a functional of the same type and

    .. math::
        \mathrm{prox}_F(v)(x) = \mathrm{prox}_{f(\cdot,x)}(v(x)).


    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a MeasureSpaceFcts
    h_domain : `regpy.hilbert.HilbertSpace` [default: None]
        Hilbert space defined on `domain`. Proximal operator is computed  wrt to that. Default: `L2(domain)`
    """

    def __init__(self,domain,h_domain = None,**kwargs):
        assert isinstance(domain,MeasureSpaceFcts)
        assert domain == h_domain.vecsp
        self.kwargs = kwargs
        super().__init__(domain,**kwargs)
        self.h_domain = L2(domain) if h_domain is None else h_domain
        """ Hilbert space on `domain` wrt to which is the prox computed."""

    def _eval(self, v):
        return np.sum(self._f(v,**self.kwargs)*self.domain.measure)

    def _conj(self,vstar):
        return np.sum(self._f_conj(vstar/self.domain.measure,**self.kwargs)*self.domain.measure)

    def _subgradient(self, v):
        return self._f_deriv(v,**self.kwargs)*self.domain.measure

    def _hessian(self, v):
        return PtwMultiplication(self.domain,self._f_second_deriv(v,**self.kwargs)*self.domain.measure)

    def _proximal(self, v, tau):
        return self._f_prox(v,tau,**self.kwargs)
    
    def _conj_proximal(self, vstar, tau):
        return self._f_conj_prox(vstar/self.domain.measure,tau,**self.kwargs)*self.domain.measure
    
    def _conj_subgradient(self, vstar):
        return self._f_conj_deriv(vstar/self.domain.measure,**self.kwargs)
    
    def _conj_hessian(self, vstar):
        return PtwMultiplication(self.domain,self._f_conj_second_deriv(vstar/self.domain.measure,**self.kwargs))

    def _f(self,v,**kwargs):
        raise NotImplementedError
    
    def _f_deriv(self,v,**kwargs):
        raise NotImplementedError

    def _f_second_deriv(self,v,**kwargs):
        raise NotImplementedError

    def _f_prox(self,v,tau,**kwargs):
        """TODO: write default implementation by Newton's method"""
        raise NotImplementedError
    
    def _f_conj(self,vstar,**kwargs):
        raise NotImplementedError
    
    def _f_conj_deriv(self,vstar,**kwargs):
        raise NotImplementedError

    def _f_conj_second_deriv(self,vstar,**kwargs):
        raise NotImplementedError

    def _f_conj_prox(self,vstar,tau,**kwargs):
        raise NotImplementedError
    
class LppPower(IntegralFunctionalBase):
    r"""
    Implements the \(p)\-power of the \(L^p)\ norm on some domain in `MeasureSpaceFcts`
    as an integral functional.

    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a MeasureSpaceFcts
    p: float >1 [option]
        exponent
    """

    def __init__(self, domain, p=2):
        assert np.isscalar(p) and p >1
        self.p = p
        self.q = p/(p-1)
        super().__init__(domain, L2(domain),
                         convexity_param = 2 if p==2 else 0,
                         Lipschitz = 2 if p==2 else inf
                         )

    def _f(self,v,**kwargs):
        return np.abs(v)**self.p/self.p
    
    def _f_deriv(self, v,**kwargs):
        return np.abs(v)**(self.p-1)*np.sign(v)
    
    def _f_second_deriv(self, v,**kwargs):
        return (self.p-1)*np.abs(v)**(self.p-2)
    
    def _f_prox(self,v,tau,**kwargs):
        if self.p==2:
            return v/(1+tau)
        else:
            raise NotImplementedError('LppPower')
    
    def _f_conj(self, vstar,**kwargs):
        return np.abs(vstar)**self.q/self.q

    def _f_conj_deriv(self, vstar,**kwargs):
        return np.abs(vstar)**(self.q-1)*np.sign(vstar)
    
    def _f_conj_second_deriv(self, vstar,**kwargs):
        return (self.q-1)*np.abs(vstar)**(self.q-2)
    
    def _f_conj_prox(self,v_star,tau,**kwargs):
        if self.p==2:
            return v_star/(1+tau)
        else:
            raise NotImplementedError('prox of conjugate of LppPower')

class L1MeasureSpace(IntegralFunctionalBase):
    r""":math:`L ^1` Functional on `MeasureSpace`. Proximal implemented for default :math:`L^2` as `h_domain`.

    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        Domain on which to define the generic L1.
    """
    def __init__(self, domain):
        super().__init__(domain,L2(domain))

    def _f(self, v,**kwargs):
        return np.abs(v)

    def _f_deriv(self, v,**kwargs):
        return np.sign(v)

    def _f_second_deriv(self, v,**kwargs):
        if np.any(v==0):
            raise NotTwiceDifferentiableError('L1')
        else:
            return np.zeros_like(v)

    def _f_prox(self, v,tau,**kwargs):
        return np.maximum(0, np.abs(v)-tau)*np.sign(v)

    def _f_conj(self, v_star,**kwargs):
        ind = (np.abs(v_star)>1)
        res = np.zeros_like(v_star)
        res[ind]= inf
        return res
    
    def _f_conj_deriv(self, v_star,**kwargs):
        if np.max(np.abs(v_star))>1:
            raise NotInEssentialDomainError()
        else:
            return np.zeros_like(v_star)

    def _f_conj_second_deriv(self, v_star,**kwargs):
        if np.max(np.abs(v_star))>=1:
            raise NotTwiceDifferentiableError('L1')
        else:
            return self.domain.zeros()

    def _f_conj_prox(self,vstar,tau,**kwargs):
        return vstar/np.maximum(np.abs(vstar),1)

    def is_subgradient(self, vstar, x, eps=1e-10):
        zeroind = (x==0)
        if np.any(zeroind) and np.max(np.abs(vstar[zeroind]))>1:
            return False
        else:
            vstar[zeroind]=0
            return super().is_subgradient(vstar, x, eps)

    def _conj_is_subgradient(self, v, xstar, eps=1e-10):
        return np.max(np.abs(xstar)<=1) and v[xstar==1]>=0 and v[xstar==-1]<=0 and v[np.abs(xstar)<1] ==0

class KullbackLeibler(IntegralFunctionalBase):
    r"""Kullback-Leiber divergence defined by

    .. math::
        F(u,w) = KL(w,u) = \int (u(x) -w(x) - w(x)\ln \frac{u(x)}{w(x)}) dx


    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        Domain on which to define the Kullback-Leibler divergence
    w: domain [optional, no default value]
        First argument of Kullback-Leibler divergence.
        Formally optional, but required for proper functioning. 
    """

    def __init__(self, domain,**kwargs):
        if 'w' not in kwargs.keys():
            raise NotImplemented("The Kulback Leibler divergence requires to define a keyword arguent w for its first component.")
        w=kwargs['w']
        assert w in domain
        assert np.min(w)>=0
        super().__init__(domain,L2(domain))
        self.kwargs = kwargs

    def _f(self, u,**kwargs):
        w= kwargs['w']
        ind_inf=(u<0)|((u==0)&(w>0))
        ind_else=~(ind_inf|(w==0))
        res=np.copy(u)
        res[ind_inf]=np.inf
        res[ind_else]=u[ind_else]-w[ind_else] - w[ind_else] * np.log(u[ind_else]/w[ind_else])
        return res    
   
    def _f_deriv(self, u,**kwargs):
        w=kwargs['w']
        assert np.min(u)>=0
        assert np.all(np.logical_or(np.logical_not(u==0),w==0))
        res = np.ones_like(u)-w/u
        res[u==0] = 1
        return res

    def _f_second_deriv(self, u, **kwargs):
        w=kwargs['w']
        assert np.min(u)>0
        return w/u**2

    def _f_conj(self, u_star,**kwargs):
        w=kwargs['w']
        if np.any(u_star)>1:
            return np.inf 
        elif np.any(np.logical_and(u_star == 1,np.logical_not(w==0))):
            return np.inf 
        else:
            return -w*np.log(1-u_star)

    def _f_conj_deriv(self, u_star,**kwargs):
        w=kwargs['w']        
        assert np.max(u_star)<=1
        assert np.all(np.logical_or(np.logical_not(u_star==1),w==0))
        return w/(1-u_star)
    
    def _f_conj_second_deriv(self, u_star,**kwargs):
        w=kwargs['w']
        assert np.max(u_star)<=1
        assert np.all(np.logical_or(np.logical_not(u_star==1),w==0))
        return w/(1-u_star)**2

class RelativeEntropy(IntegralFunctionalBase):
    r"""Kullback-Leiber divergence define by

    .. math::
        F(u,w) = \int (u(x)\ln \frac{u(x)}{w(x)}) dx


    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        Domain on which to define the Kullback-Leibler divergence
    w: domain [optional, no default value]
        reference value.
        Formally optional, but required for proper functioning. 
    """

    def  __init__(self, domain,**kwargs):
        super().__init__(domain,L2(domain))
        w = kwargs['w']
        assert w in domain
        assert np.min(w)>0
        self.kwargs = kwargs

    def _f(self, u,**kwargs):
        w=kwargs['w']        
        ind_upos=(u>0)
        res=np.zeros_like(u)
        res[u<0] = np.inf
        res[ind_upos]=u[ind_upos] * np.log(u[ind_upos]/w[ind_upos])
        return res    
   
    def _f_deriv(self, u,**kwargs):
        w=kwargs['w']        
        assert np.min(u)>0
        res = np.ones_like(u)+np.log(u/w)
        return res

    def _f_second_deriv(self, u, **kwargs):
        assert np.min(u)>0
        return 1/u

    def _f_conj(self, u_star,**kwargs):
        w=kwargs['w']
        return w*(np.exp(u_star-1))

    def _f_conj_deriv(self, u_star,**kwargs):
        w=kwargs['w']
        return w*np.exp(u_star-1)
    
    def _f_conj_second_deriv(self, u_star,**kwargs):
        w=kwargs['w']
        return w*np.exp(u_star-1)

class Huber(IntegralFunctionalBase):
    r"""Huber functional 

    .. math::
        F(x) = 1/2 |x|^2                if  |x|\leq \sigma
        F(x) = \sigma |x|-\sigma^2/2    if  |x|>\sigma


    Parameters 
    ----------
    domain: regpy.vecsps.MeasureSpaceFcts
        domain on which Huber functional is defined
    sigma: float or domain [default: 1]
        parameter in the Huber functional. 
    as_primal: boolean [default:True]
        If False, then the functional is initiated as conjugate of QuadraticIntv. Then the dual metric is used, 
        and precautions against an infinite recursion of conjugations are taken.
    eps: float [default: 0.]
        Only used for conjugate functional. See description of `QuadraticIntv`
    """

    def  __init__(self, domain,as_primal=True,sigma = 1.,eps=0.):
        if as_primal:
            super().__init__(domain,L2(domain),Lipschitz=1)
            self.conjugate = QuadraticIntv(domain,as_primal=False,sigma=sigma,eps=eps)
        else:
            super().__init__(domain,L2(domain,weights=1./domain.measure**2), Lipschitz=1)        
            
        assert isinstance(sigma, (float,int)) or sigma in domain 
        assert np.min(sigma)>0
        if isinstance(sigma, (float,int)) :
            self.sigma = np.real(sigma * domain.ones())
        else:
            self.sigma = np.real(sigma) 

    def _f(self, u,**kwargs):
        return np.where(np.abs(u)<=self.sigma,0.5*np.abs(u)**2,self.sigma*np.abs(u)-0.5*self.sigma**2)

           
    def _f_deriv(self, u,**kwargs):
        return np.where(np.abs(u)<=self.sigma,u,self.sigma*u/np.abs(u))


    def _f_second_deriv(self, u, **kwargs):
        return (np.abs(u)<=self.sigma).astype(float)

    def _f_conj(self, ustar,**kwargs):
        return self.conjugate._f(ustar)    
   
    def _f_conj_deriv(self, ustar,**kwargs):
        return self.conjugate._f_deriv(ustar)

    def _f_conj_second_deriv(self, ustar,**kwargs):
        return self.conjugate._f_second_deriv(ustar)

    def _f_conj_prox(self,ustar,tau,**kwargs):
        return self.conjugate._f_prox(ustar,tau)


class QuadraticIntv(IntegralFunctionalBase):
    r"""Functional 

    .. math::
        F(x) = 1/2 |x|^2    if |x|\leq \sigma(x)
        F(x) = \infty    if |x|>\sigma(x)


    Parameters
    ----------
    regpy.vecsps.MeasureSpaceFcts
        domain on which Huber functional is defined
    sigma: float or domain [default: 1]
        interval width. 
    as_primal: boolean [default:True]
        If False, then the functional is initiated as conjugate of Huber. Then the dual metric is used, 
        and precautions against an infinite recursion are taken.
    eps: float [default: 0.]
        sigma is replace by sigma*(1+eps) on all operations except the proximal mapping to avoid np.inf return values 
        or NotInEssentialDomain exceptions in the presence of rounding errors
    """

    def  __init__(self, domain,as_primal=True,sigma=1.,eps=0.):
        if as_primal:
            super().__init__(domain,L2(domain),convexity_param=1)
            self.conjugate = Huber(domain,as_primal=False,sigma=sigma)
        else:
            super().__init__(domain,L2(domain,weights=1./domain.measure**2), convexity_param=1)
        assert isinstance(sigma, (float,int)) or sigma in domain 
        assert np.min(sigma)>0
        if isinstance(sigma, (float,int)):
            self.sigma = sigma * domain.ones()
        else:
            self.sigma = sigma 
        self.sigmaeps = self.sigma*(1+eps) if eps>0 else self.sigma

    def _f(self, u,**kwargs):
        res =  0.5*np.abs(u)**2
        res[np.abs(u)>self.sigmaeps] = np.inf
        return res    
   
    def _f_deriv(self, u,**kwargs):
        if np.max(np.abs(u)/self.sigmaeps)>1.:
            raise NotInEssentialDomainError('QuadraticIntv')
        return u.copy()

    def _f_prox(self,u,tau,**kwargs):
        res = u/(1+tau)
        return res/np.maximum(np.abs(res)/self.sigma,1)

    def _f_second_deriv(self, u,**kwargs):
        if np.max(np.abs(u)/self.sigmaeps)>=1.:
            raise NotTwiceDifferentiableError('QuadraticIntv')
        else:
            return np.ones_like(u)

    def _f_conj(self, ustar,**kwargs):
        return self.conjugate._f(ustar)    
   
    def _f_conj_deriv(self, ustar,**kwargs):
        return self.conjugate._f_deriv(ustar)

    def _f_conj_second_deriv(self, ustar,**kwargs):
        return self.conjugate._f_second_deriv(ustar)

    def _f_conj_prox(self,ustar,tau,**kwargs):
        return self.conjugate._f_prox(ustar,tau)

    def is_subgradient(self, vstar, x, eps=1e-10):
        grad = self.subgradient(x)
        if(not np.all(np.abs(x)<=self.sigma)):
            return False
        if(not np.all(vstar[self.sigma==x]>=self.sigma)):
            return False
        if(not np.all(vstar[-self.sigma==x]<=-self.sigma)):
            return False
        if(np.linalg.norm(grad[np.abs(x)<self.sigma]-vstar[np.abs(x)<self.sigma]) <= eps*np.linalg.norm(grad[np.abs(x)<self.sigma])):
            return True
        return False


class QuadraticNonneg(IntegralFunctionalBase):
    r"""Functional 

    .. math::
        F(x) = 1/2 |x|^2    if x\geq 0
        F(x) = \infty       if  x<0

    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        domain on which functional is defined 

    """

    def  __init__(self, domain):
        super().__init__(domain,L2(domain),convexity_param = 1.)

    def _f(self, u,**kwargs):
        res =  u*u/2
        res[u<0] = np.inf
        return res    

    def _f_deriv(self, u,**kwargs):
        if np.min(u)<0:
            raise NotInEssentialDomainError('QuadraticNonneg')
        return u.copy()

    def _f_prox(self,u,tau,**kwargs):
        return np.maximum(u/(1+tau),0)

    def _f_second_deriv(self, u,**kwargs):
        if np.min(u)<0:
            raise NotTwiceDifferentiableError('QuadraticNonneg')
        else:
            return np.ones_like(u)

    def _f_conj(self, ustar,**kwargs):
        res = ustar*ustar/2
        res[ustar<0] = 0
        return res

    def _f_conj_deriv(self, ustar,**kwargs):
        res = ustar.copy()
        res[ustar<0] = 0
        return res

    def _f_conj_second_deriv(self, ustar,**kwargs):
        return 1.* (ustar>=0)

    def _f_conj_prox(self,ustar,tau,**kwargs):
        res=ustar.copy()
        res[ustar>0]*=(1/(1+tau))
        return res
    
    def is_subgradient(self, vstar, x, eps=1e-10):
        return np.max(vstar[x<0])<=0 and np.linalg.norm(x[x>=0]-vstar[x>=0]) <= eps*np.linalg.norm(x[x>=0])




class QuadraticBilateralConstraints(LinearCombination):
    r""" Returns `Functional` defined by 

    .. math::
        F(x) = \frac{\alpha}{2}\|x-x0\|^2  if lb\leq x\leq ub
        F(x) = np.inf else


    Parameters
    ----------
    domain: regpy.vecsps.MeasureSpaceFcts
        domain on which functional is defined
    lb: domain
        lower bound
    ub: domain
        upper bound
    x0: domain
        reference value
    alpha: float [default: 1]
        regularization parameter
    eps: real [default: 0]
        Tolerance parameter for violations of the hard constraints (which may occur due to rounding errors).
        If constraints are violated by less then eps times the interval width, the polynomial is evaluated, rather than returning np.inf.
    """

    def __init__(self,domain, lb=None, ub=None, x0=None,alpha=1.,eps=0.):
        assert isinstance(domain,MeasureSpaceFcts)
        if isinstance(lb,(float,int)):
            lb = lb*domain.ones()
        elif lb is None:
            lb = domain.zeros()
        assert lb in domain
        if isinstance(ub,(float,int)):
            ub = ub*domain.ones()
        elif ub is None:
            ub = domain.zeros()
        assert ub in domain
        assert np.all(lb<ub)
        if x0 is None:
            x0 =0.5*(lb+ub)
        elif isinstance(x0,(float,int)):
            x0 = x0*domain.ones()
        assert x0 in domain 
        assert isinstance(alpha,(float,int))

        self.lb = lb; self.ub = ub; self.x0 =x0; self.alpha = alpha
        F = QuadraticIntv(domain,sigma=(ub-lb)/2.,eps=eps)
        center = (ub+lb)/2
        lin = LinearFunctional(center-x0,
                            domain=domain,
                            gradient_in_dual_space=False
                            )
        offset = 0.5*(np.sum((x0**2-center**2)*domain.measure))
        # return  alpha*HorizontalShiftDilation(F,shift=center) + alpha*lin + alpha*offset
        super().__init__((alpha,HorizontalShiftDilation(F,shift=center)+offset),
                          (alpha,lin)
                          )

def QuadraticLowerBound(domain, lb=None, x0=None,a=1.):
    r""" Returns `Functional` defined by 

    \[F(x) = \frac{a}{2}\|x-x0\|^2  if lb\leq x
     F(x) = np.inf else


    Parameters
    ----------
    domain: `vecsps.MeasureSpaceFcts`
        domain on which the functional is defined
    lb: domain or float [default: None]
        lower bound (zero in the default case)
    x0: domain or float [default: None]
        lower bound (zero in the default case)
    """
    assert isinstance(domain,MeasureSpaceFcts)
    if isinstance(lb,(float,int)):
        lb = lb*domain.ones()
    elif lb is None:
        lb = domain.zeros()
    assert lb in domain
    if isinstance(x0,(float,int)):
        x0 = x0*domain.ones()
    elif x0 is None:
        x0 = domain.zeros()
    assert x0 in domain 
    assert isinstance(a,(float,int))

    F = QuadraticNonneg(domain)
    lin = LinearFunctional(lb-x0,domain=domain,gradient_in_dual_space=False)
    offset = 0.5*(np.sum((x0**2-lb**2)*domain.measure))
    return a*HorizontalShiftDilation(F,shift=lb)+ a*lin + a*offset

class QuadraticPositiveSemidef(Functional):
    r"""Functional 

    .. math::
        F(x) = 1/2 ||x||_{HS}^2    \text{if } x\geq 0 \text{ and (optional) } tr(x)=c
        F(x) = \infty       \text{else}

    Here x is a quadratic matrix and HS is the Hilbert-Schmidt norm. Conjugate functional
    and prox are only correct for hermitian inputs.

    Parameters
    ---------
    domain: regpy.vecsps.UniformGridFcts
        two dimensional domain on which functional is defined, volume_elements have to be one
    trace_val: float or None, optional
        desired value of trace or None for no trace constraint. Defaults to None.
    tol: float, optional
        tolerance for comparisons determining positive semidefiniteness and correctness of trace 

    """

    def  __init__(self, domain,trace_val=None,tol=1e-15):
        assert isinstance(domain,UniformGridFcts)
        assert domain.ndim==2
        assert domain.shape[0]==domain.shape[1]
        assert domain.volume_elem==1
        assert tol>=0
        assert trace_val is None or trace_val>0
        self.tol=tol
        if(trace_val is not None):
            self.has_trace_constraint=True
            self.trace_val=trace_val
        else:
            self.has_trace_constraint=False
        super().__init__(domain,L2(domain),Lipschitz=1,convexity_param=1)

    def is_in_essential_domain(self,rho):
        if(not ishermitian(rho,atol=self.tol)):
            return False
        if(self.has_trace_constraint):
            if(np.abs(np.trace(rho)-self.trace_val)>self.tol):
                return False
        evs=np.linalg.eigvalsh(rho)
        return evs[0]>-self.tol
    
    @staticmethod
    def closest_point_simplex(p,a):
        r'''
        Algorithm from Held, Wolfe and Crowder (1974) to project onto simplex :math:`\{q:q_{i}\qeq 0,\sum q_{i}=a\}`.
        It uses that p is already sorted in increasing order.

        Parameters
        ---------
        p: numpy.ndarray
            Input point sorted in increasing order
        a: float
            positive value that is the sum of the elements in the result
        '''
        p_flipped=np.flip(p)
        comp_vals=(np.cumsum(p_flipped)-a)/np.arange(1,p.shape[0]+1)
        k=np.where(comp_vals<p_flipped)[0][-1]
        t=comp_vals[k]
        return np.maximum(p-t,0)
        

    def _eval(self, x):
        if(self.is_in_essential_domain(x)):
            return np.sum(np.abs(x)**2)/2
        else:
            return np.inf

    def _proximal(self, x, tau):
        evs,U=np.linalg.eigh(x)
        evs/=(1+tau)
        if(self.has_trace_constraint):
            proj_evs=QuadraticPositiveSemidef.closest_point_simplex(evs,self.trace_val)
        else:
            proj_evs=np.maximum(0,evs)
        return U@np.diag(proj_evs)@np.conj(U).T
        
    def _subgradient(self, x):
        if(self.is_in_essential_domain(x)):
            return np.copy(x)
        else:
            return NotInEssentialDomainError
        
    def _hessian(self,x):
        if(self.is_in_essential_domain(x)):
            return self.domain.identity
        else:
            return NotInEssentialDomainError
        
    def _conj(self,xstar):
        evs=np.linalg.eigvalsh(xstar)
        if(self.has_trace_constraint):
            cps=QuadraticPositiveSemidef.closest_point_simplex(evs,self.trace_val)
            return (np.sum(evs**2)+np.sum((cps-evs)**2))/2
        else:
            return (np.sum(evs**2)+np.sum(evs**2,where=evs<0))/2


class L1Generic(Functional):
    r"""Generic :math:`L ^1` Functional. Proximal implemented for default :math:`L^2` as `h_domain`.

    Parameters
    ----------
    domain : regpy.vecsps.NumPyVectorSpace
        Domain on which to define the generic L1.
    """
    def __init__(self, domain):
        super().__init__(domain)
        assert isinstance(self.domain,NumPyVectorSpace)

    def _eval(self, x):
        return np.sum(np.abs(x))

    def _subgradient(self, x):
        return np.sign(x)

    def _hessian(self, x):
        # Even approximate Hessians don't work here.
        raise NotImplementedError

    def _proximal(self, x, tau):
        return np.maximum(0, np.abs(x)-tau)*np.sign(x)


class TVGeneric(Functional):
    r"""Generic TV Functional. Proximal implemented for default `L2` as `h_space`

    NotImplemented yet!
    """
    def __init__(self, domain, h_domain=L2):
        super().__init__(domain,h_domain=h_domain)

    def _subgradient(self, x):
        return NotImplementedError

    def _hessian(self, x):
        return NotImplementedError
    
    def _proximal(self, x, tau):
        return NotImplementedError


class TVUniformGridFcts(Functional):
    r"""Total Variation Norm: For :math:`C^1` functions the :math:`l^1`-norm of the gradient on a `UniformGrid`

    Parameters
    ----------
    domain : regpy.vecsps.UniformGridFcts
        Underlying domain. 
    h_domain : regpy.hilbert.HilbertSapce (defaul: L2)
        Underlying Hilbert space for proximal. 
    """
    def __init__(self, domain, h_domain=None):
        assert isinstance(domain, UniformGridFcts)
        self.dim = np.size(domain.shape)
        """Dimension of the Uniform Grid functions.
        """
        super().__init__(domain,h_domain=h_domain)

    def _eval(self, x):
        if self.dim==1:
            return np.sum(np.abs(self._gradientuniformgrid(x)))
        else:
            return np.sum(np.linalg.norm(self._gradientuniformgrid(x), axis=0))

    def _subgradient(self, x):
        if self.dim==1:
            return np.sign(self._gradientuniformgrid(x)).reshape(self.domain.shape)
        else:
            grad = self._gradientuniformgrid(x)
            grad_norm = np.linalg.norm(grad, axis=0)
            toret = np.zeros(x.shape)
            toret = np.where(grad_norm != 0, np.sum(grad, axis=0) / grad_norm, toret)
            return toret

    def _hessian(self, x):
        raise NotImplementedError

    def _proximal(self, x, tau, stepsize=0.1, maxiter=10):
        shape = [self.dim]+list(x.shape)
        p = np.zeros(shape)
        for i in range(maxiter):
            update = stepsize*self._gradientuniformgrid( self.h_domain.gram_inv( self._divergenceuniformgrid(p))-x/tau)
            p = (p+update) / (1+np.abs(update))
        return x-tau*self._divergenceuniformgrid(p)

    def _gradientuniformgrid(self, u):
        r"""Computes the gradient of field given by 'u'. 'u' is defined on a 
        equidistant grid. Returns a list of vectors that are the derivatives in each 
        dimension."""
        # Need to reshape spacing otherwise getting braodcasting error
        shape = [self.domain.ndim]+[1 for _ in self.domain.shape]
        return 1/self.domain.spacing.reshape(shape)*np.array(np.gradient(u))

    def _divergenceuniformgrid(self, u):
        r"""Computes the divergence of a vector field 'u'. 'u' is assumed to be
        a list of matrices u=(u_x, u_y, u_z, ...) holding the values for u on a
        regular grid"""
        return np.ufunc.reduce(np.add, [np.gradient(u[i], axis=i)/h for i,h in enumerate(self.domain.spacing)])