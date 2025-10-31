from math import inf

import numpy as np
from scipy.linalg import ishermitian
from scipy.special import lambertw

from regpy.operators import PtwMultiplication
from regpy.vecsps.numpy import *
from regpy.hilbert import L2
import logging
from copy import deepcopy

from .base import Functional, Conj, LinearFunctional,LinearCombination,HorizontalShiftDilation,NotInEssentialDomainError,NotTwiceDifferentiableError

__all__ = ["IntegralFunctionalBase","LppPower","L1MeasureSpace","KullbackLeibler","RelativeEntropy","Huber","QuadraticIntv","QuadraticBilateralConstraints","QuadraticLowerBound","QuadraticNonneg","QuadraticPositiveSemidef","L1Generic","TVGeneric","TVUniformGridFcts"]


class IntegralFunctionalBase(Functional):
    r"""
    This class provides a general framework for integral functionals of the type
    
    .. math::
        F\colon X \to \mathbb{R}
        v\mapsto \Int_\Omega f(v(x),x)\mathrm{d}x

    with \(f\colon \mathbb{R}^2\to \mathbb{R})\. 

    Subclasses defining explicit functionals of this type have to implement
     * `_f` evaluation the function \(f)\
     * `_f_deriv` the derivative \(\partial_v f)\
     * `_f_second_deriv' the second derivative \(\partial f^2/\partial v^2)\ (often not needed!)
     * `_f_prox` giving the proximal function \(\mathrm{prox}_{\tau f(.,x)})\ for each \(x\in\Omega)\
     * `_f_conj` evaluation the Fenchel conjugate function \(f^*(v^*,x))\
     * `_f_conj_deriv` the derivative \(\partial_{v^*}f*)\
     * `_f_conj_second_deriv' the second derivative \(\partial f*^2/\partial v_*^2)\ (often not needed!)
     * `_f_conj_prox` giving the proximal  function \( \(\mathrm{prox}_{\tau f^*(\cdot,x)})\     
    
    since 

    .. math::
        F'[g]h = \int_\Omega h(x)(\partial_1 f)(g(x),x)

    is a functional of the same type, and

    .. math::
        \mathrm{prox}_{\tau F}(v)(x) = \mathrm{prox}_{\tau f(\cdot,x)}(v(x)).


    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a MeasureSpaceFcts
    dom_l,dom_u : float or np.ndarray [default: -np.inf and np.inf, rsp.]
        lower and upper bound on the essential domain of f (the interval on which f is finite)
        If dom_l or dom_u are finite, f should be finite at these points (possibly very large if f tends to infinity there) 
        If the domain depends on the point x and/or arguments in **kwargs, this should be a numpy array.
    conj_dom_l,conj_dom_u : float or np.ndarray [default: -np.inf and np.inf, rsp.]
        lower and upper bound on the essential domain of the conjugate of f (the interval on on which f^* is finite)    
        (as vector in primal space)
    constr_u: None or float or np.ndarray [default: None]
        If not None, an upper constraint is imposed, i.e. f(v,x) is replaced by a function that takes the value np.inf 
        if v>contr_u(x).
    constr_l: None or float or np.ndarray [default: None]
        As constr_u, but for a lower constraint. 
    lin_taylor_u: None or float or np.ndarray [default: None]
        If not None, f(v,x) is replaced by its first order Taylor expansion 
        \( f(r(x),x) + (v-r(x)) \partial_v f(r(x),x) )\ if \(x>r(x):=lin_taylor_u(x) )\
    lin_taylor_l: None or float or np.ndarray [default: None]
        Analogous to right linearization, but for small values of v.
    quad_taylor_u: None or float or np.ndarray [default: None]
        Analogous to lin_taylor_u, but with a quadratic Taylor expansion
    quad_taylor_l: None or float or np.ndarray [default: None]
        Analogous to quad_taylor_u, but for small values of v
    """

    def __init__(self,domain,
                 dom_l=-np.inf, dom_u=np.inf, 
                 conj_dom_l=-np.inf,conj_dom_u=np.inf,
                 constr_l=None, constr_u=None,
                 lin_taylor_l=None, lin_taylor_u=None,
                 quad_taylor_l=None, quad_taylor_u= None,
                 Lipschitz = np.inf, convexity_param=0.,
                 **kwargs):
        assert isinstance(domain,MeasureSpaceFcts)
        self.h_domain = L2(domain)
        self.measure = np.broadcast_to(domain.measure,domain.shape) if np.isscalar(domain.measure) else domain.measure
        self.domain = domain

        if sum([trunc is not None for trunc in [constr_l,lin_taylor_l,quad_taylor_l]])>1:
            raise ValueError('At most one of the parameters constr_l,lin_taylor_l, quad_taylor_l may be specified.')
        if sum([trunc is not None for trunc in [constr_u,lin_taylor_u,quad_taylor_u]])>1:
            raise ValueError('At most one of the parameters constr_u,lin_taylor_u, quad_taylor_u may be specified.')        

        assert np.isscalar(dom_l) or dom_l in domain
        assert np.isscalar(dom_u) or dom_u in domain
        assert constr_l is None or np.isscalar(constr_l) or constr_l in domain
        assert constr_u is None or np.isscalar(constr_u) or constr_u in domain        
        assert lin_taylor_l is None or np.isscalar(lin_taylor_l) or lin_taylor_l in domain
        assert lin_taylor_u is None or np.isscalar(lin_taylor_u) or lin_taylor_u in domain
        assert quad_taylor_l is None or np.isscalar(quad_taylor_l) or quad_taylor_l in domain
        assert quad_taylor_u is None or np.isscalar(quad_taylor_u) or quad_taylor_u in domain

        self.constr_l_active = False if constr_l is None else np.any(constr_l>dom_l)
        self.constr_u_active = False if constr_u is None else np.any(constr_u<dom_u)

        if self.constr_l_active:
            active_ind_l = constr_l> (dom_l if dom_l in domain else np.broadcast_to(dom_l,domain.shape))
        if self.constr_u_active:
            active_ind_u = constr_u< (dom_u if dom_u in domain else np.broadcast_to(dom_u,domain.shape))

        self.quad_taylor_l_active = (quad_taylor_l is not None)
        self.quad_taylor_u_active = (quad_taylor_u is not None)

        dom_l = np.full(domain.shape,dom_l) if np.isscalar(dom_l) else dom_l
        if lin_taylor_l is None and quad_taylor_l is None: 
            if not constr_l is None:
                dom_l = np.maximum(dom_l, constr_l)
        else:
            taylor_l = lin_taylor_l if lin_taylor_l is not None else quad_taylor_l
            dom_l[taylor_l>=dom_l] = -np.inf
        self._if_constant_broadcast(dom_l)
  
        dom_u = np.full(domain.shape,dom_u) if np.isscalar(dom_u) else dom_u
        if lin_taylor_u is None and quad_taylor_u is None:
            if not constr_u is None:
                dom_u = np.minimum(dom_u, constr_u)
        else:
            taylor_u = lin_taylor_u if lin_taylor_u is not None else quad_taylor_u
            dom_u[taylor_u<=dom_u] = np.inf
        self._if_constant_broadcast(dom_u)

        if np.isscalar(conj_dom_l) and isinstance(domain,UniformGridFcts):
            conj_dom_l = np.broadcast_to(conj_dom_l*self.measure,domain.shape)
        else:
            conj_dom_l = self.h_domain.gram(np.broadcast_to(conj_dom_l,domain.shape))

        if np.isscalar(conj_dom_u) and isinstance(domain,UniformGridFcts):
            conj_dom_u = np.broadcast_to(conj_dom_u*self.measure,domain.shape)
        else:
            conj_dom_u = self.h_domain.gram(np.broadcast_to(conj_dom_u,domain.shape))

        super().__init__(domain,Lipschitz=Lipschitz,convexity_param=convexity_param,
                         separable=True,
                         dom_l = dom_l if dom_l in domain else np.broadcast_to(dom_l,domain.shape),
                         dom_u = dom_u if dom_u in domain else np.broadcast_to(dom_u,domain.shape),
                         conj_dom_l = conj_dom_l, 
                         conj_dom_u = conj_dom_u
                         )

        if self.constr_l_active:
            self.conj_taylor_l = np.full(domain.shape,-np.inf)
            self.conj_taylor_l[active_ind_l] = self._f_deriv(dom_l[active_ind_l],mask=active_ind_l,**kwargs)*self.measure[active_ind_l]
            self._if_constant_broadcast(self.conj_taylor_l)

            self.f_dom_l = self.domain.zeros()
            self.f_dom_l[active_ind_l] = self._f(dom_l[active_ind_l],mask=active_ind_l,**kwargs)
            self._if_constant_broadcast(self.f_dom_l)

            self.conj_dom_l = np.array(self.conj_dom_l)
            self.conj_dom_l[active_ind_l] = -np.inf
            self._if_constant_broadcast(self.conj_dom_l)

        if self.constr_u_active:
            self.conj_taylor_u = np.full(domain.shape,np.inf)
            self.conj_taylor_u[active_ind_u] = self._f_deriv(dom_u[active_ind_u],mask=active_ind_u,**kwargs)*self.measure[active_ind_u]
            self._if_constant_broadcast(self.conj_taylor_u)

            self.f_dom_u = self.domain.zeros()
            self.f_dom_u[active_ind_u] = self._f(dom_u[active_ind_u],mask=active_ind_u,**kwargs)
            self._if_constant_broadcast(self.f_dom_u)

            self.conj_dom_u = np.array(self.conj_dom_u)
            self.conj_dom_u[active_ind_u] = np.inf
            self._if_constant_broadcast(self.conj_dom_u)


        if lin_taylor_l is not None:
            self.taylor_l = lin_taylor_l if lin_taylor_l in domain else np.broadcast_to(lin_taylor_l,domain.shape)
            conj_active_ind_l = self.taylor_l>self.dom_l
            self.conj_constr_l_active = np.any(conj_active_ind_l)
            if self.conj_constr_l_active:
                self.conj_dom_l = np.array(self.conj_dom_l)
                self.conj_dom_l[conj_active_ind_l] = self._f_deriv(self.taylor_l[conj_active_ind_l],mask=conj_active_ind_l,**kwargs)*self.measure[conj_active_ind_l]
                self._if_constant_broadcast(self.conj_dom_l)
                
                self.conj_f_conj_dom_l = domain.zeros()
                self.conj_f_conj_dom_l[conj_active_ind_l] = self._f_conj(self.conj_dom_l[conj_active_ind_l]/self.measure[conj_active_ind_l],mask=conj_active_ind_l,**kwargs)
                self._if_constant_broadcast(self.conj_f_conj_dom_l)
        else:
            self.conj_constr_l_active = False

        if lin_taylor_u is not None:
            self.taylor_u = lin_taylor_u if lin_taylor_u in domain else np.broadcast_to(lin_taylor_u,domain.shape)
            conj_active_ind_u = self.taylor_u<self.dom_u
            self.conj_constr_u_active = np.any(conj_active_ind_u)
            if self.conj_constr_u_active:
                self.conj_dom_u = np.array(self.conj_dom_u)
                self.conj_dom_u[conj_active_ind_u] = self._f_deriv(self.taylor_u[conj_active_ind_u],mask=conj_active_ind_u,**kwargs)*self.measure[conj_active_ind_u]
                self._if_constant_broadcast(self.conj_dom_u)

                self.conj_f_conj_dom_u = domain.zeros()
                self.conj_f_conj_dom_u[conj_active_ind_u] = self._f_conj(self.conj_dom_u[conj_active_ind_u]/self.measure[conj_active_ind_u],mask=conj_active_ind_u,**kwargs)
                self._if_constant_broadcast(self.conj_f_conj_dom_u)
        else:
            self.conj_constr_u_active = False

        if quad_taylor_l is not None:
            self.taylor_l = quad_taylor_l if quad_taylor_l in domain else np.broadcast_to(quad_taylor_l,domain.shape)
            self.conj_constr_l_active = False
            taylor_active_l = self.taylor_l>=self.dom_l
            if np.any(taylor_active_l):
                self.conj_taylor_l = np.full(self.domain.shape,-np.inf)
                self.conj_taylor_l[taylor_active_l] = self._f_deriv(self.taylor_l[taylor_active_l],mask=taylor_active_l,**kwargs)*self.measure[taylor_active_l]
                self._if_constant_broadcast(self.conj_taylor_l)

                self.conj_f_conj_dom_l = domain.zeros()
                self.conj_f_conj_dom_l[taylor_active_l] = self._f_conj(self.conj_taylor_l[taylor_active_l]/self.measure[taylor_active_l],mask=taylor_active_l,**kwargs)
                self._if_constant_broadcast(self._conj_f_conj_dom_l)

                self._fprime_l = domain.zeros()
                self._fprime_l[taylor_active_l] = self._f_deriv(self.taylor_l[taylor_active_l]/self.measure[taylor_active_l],mask=taylor_active_l,**kwargs)
                self._if_constant_broadcast(self._fprime_l)
 
        if quad_taylor_u is not None:
            self.taylor_u = quad_taylor_u if quad_taylor_u in domain else np.broadcast_to(quad_taylor_u,domain.shape)
            self.conj_constr_u_active = False
            taylor_active_u = self.taylor_u<=self.dom_u
            if np.any(taylor_active_u):
                self.conj_taylor_u = np.full(self.domain.shape,np.inf)
                self.conj_taylor_u[taylor_active_u] = self._f_deriv(self.taylor_u[taylor_active_u],mask=taylor_active_u,**kwargs)*self.measure[taylor_active_u]
                self._if_constant_broadcast(self.conj_taylor_u)

                self.conj_f_conj_dom_u = domain.zeros()
                self.conj_f_conj_dom_u[taylor_active_u] = self._f_conj(self.conj_taylor_u[taylor_active_u]/self.measure[taylor_active_u],mask=taylor_active_u,**kwargs)
                self._if_constant_broadcast(self.conj_f_conj_dom_u)

                self._fprime_u = domain.zeros()
                self._fprime_u[taylor_active_u] = self._f_deriv(self.taylor_u[taylor_active_u]/self.measure[taylor_active_u],mask=taylor_active_u,**kwargs)
                self._if_constant_broadcast(self._fprime_u)

        if (self.constr_l_active or self.constr_u_active):
            self.Lipschitz=np.inf
        if (self.conj_constr_l_active or self.conj_constr_u_active):
            self.convexity_param = 0.

        self.everywhere_finite = (np.all(dom_l==-np.inf) and np.all(dom_u==np.inf))
        self.conj_everywhere_finite = (np.all(conj_dom_l==-np.inf) and np.all(conj_dom_u==np.inf))

        self._buf = self.domain.zeros()
        self.kwargs = kwargs
        if 'logging_level' in kwargs.keys():
            self.log.setLevel(kwargs['logging_level'])

    def _if_constant_broadcast(self,x):
        assert isinstance(x,np.ndarray)
        assert x.shape == self.domain.shape
        if np.allclose(x, x.flatten()[0],rtol=1e-10,atol=1e-12):
            x = np.broadcast_to(x.flatten()[0],self.domain.shape)

    def _assert_essential_domain(self,v,eps=1e-10,msg=None):
        self._buf = v-self.dom_u
        assert np.all(self._buf<=eps), msg+f" argument too large in {self}. diff:{np.max(self._buf)}, eps={eps}"
        self._buf = self.dom_l-v
        assert np.all(self._buf<=eps), msg+f" argument too small in {self}. diff:{np.max(self._buf)}, eps={eps}"

    def _assert_conj_essential_domain(self,vstar,eps=1e-10,msg=None):
        self._buf = vstar-self.conj_dom_u
        assert np.all(self._buf<=eps), msg+f" argument too large in {self}. diff:{np.max(self._buf)}, eps={eps}"
        self._buf = self.conj_dom_l-vstar
        assert np.all(self._buf<=eps), msg+f" argument too small in {self}. diff:{np.max(self._buf)}, eps={eps}"

    def _eval(self, v, func_vals =None):
        # see comment in _conj! 
        if self.conj_constr_l_active:
            v_small = (v<self.taylor_l)
            if np.any(v_small):    
                mask = ~v_small
                self._buf[v_small] = v[v_small]
                self._buf[v_small] *= self.conj_dom_l[v_small]/self.measure[v_small]
                self._buf[v_small] -= self.conj_f_conj_dom_l[v_small]
        if self.conj_constr_u_active:
            v_large = (v>self.taylor_u)
            if np.any(v_large):    
                mask = np.logical_and(mask,~v_large) if 'mask' in locals() else ~v_large
                self._buf[v_large] = v[v_large]
                self._buf[v_large] *= self.conj_dom_u[v_large]/self.measure[v_large]
                self._buf[v_large] -= self.conj_f_conj_dom_u[v_large]  
        if 'mask' in locals():
            self._buf[mask] = self._f(v[mask],mask=mask,**self.kwargs)
        else:
            self._buf = self._f(v,**self.kwargs)            
        if self.constr_l_active:
            self._buf[v<self.dom_l] = np.inf
        if self.constr_u_active:
            self._buf[v>self.dom_u] = np.inf
        if func_vals is not None:
            np.copyto(func_vals,self._buf)
        self._buf *= self.measure
        return np.sum(self._buf)

    def _conj(self,vstar,func_vals=None):
        # Using the definition of the conjugate, for v*<conj_taylor_l := f'(dom_l)  we get  
        # f^*(v^*) = v^* dom_l - f(dom_l).
        # By Young's equality conj_taylor_l * dom_l = f(dom_l) + f^*(conj_taylor_l) 
        # and by the fact that f^*' and f' are inverse to each other, f^*(v^*) equals the first order Taylor approximation
        # f^*(v^*) = f^*(conj_taylor_l) + f^*'(conj_taylor_l)(v^*-conj_taylor_l)
        # This identity is used in the _eval.
        self._buf2 = vstar/self.measure
        if self.constr_l_active:
            vstar_small = (vstar<self.conj_taylor_l)
            if np.any(vstar_small):
                mask = ~vstar_small            
                self._buf[vstar_small] = self._buf2[vstar_small]
                self._buf[vstar_small] *= self.dom_l[vstar_small]
                self._buf[vstar_small] -= self.f_dom_l[vstar_small]
        if self.constr_u_active:
            vstar_large = (vstar>self.conj_taylor_u)
            if np.any(vstar_large):   
                mask = np.logical_and(mask,~vstar_large) if 'mask' in locals() else ~vstar_large
                self._buf[vstar_large] = self._buf2[vstar_large]
                self._buf[vstar_large] *= self.dom_u[vstar_large]
                self._buf[vstar_large] -= self.f_dom_u[vstar_large]        
        if 'mask' in locals():
            self._buf[mask] = self._f_conj(self._buf2[mask],mask=mask,**self.kwargs)
        else:
            self._buf = self._f_conj(self._buf2,**self.kwargs)
        if self.conj_constr_l_active:
            self._buf[vstar<self.conj_dom_l] = np.inf
        if self.conj_constr_u_active:
            self._buf[vstar>self.conj_dom_u] = np.inf
        if func_vals is not None:
            np.copyto(func_vals,self._buf)
        self._buf *= self.measure
        return np.sum(self._buf)

    def _subgradient(self, v):
        if not self.everywhere_finite:
            self._assert_essential_domain(v,msg='_subgradient')
        if self.conj_constr_l_active:
            v_small = (v<self.taylor_l)
            if np.any(v_small):
                mask = np.logical_not(v_small)   
                self._buf[v_small] = self.conj_dom_l[v_small]
        if self.conj_constr_u_active:
            v_large = (v>self.taylor_u)
            if np.any(v_large):
                mask = np.land(mask,~v_large) if 'mask' in locals() else ~v_large
                self._buf[v_large] = self.conj_dom_u[v_large]     
        if 'mask' in locals():
            self._buf[mask] = self._f_deriv(v[mask],mask=mask,**self.kwargs)*self.measure[mask]
        else:
            self._buf = self._f_deriv(v,**self.kwargs)*self.measure
        return self._buf.copy()

    def _conj_subgradient(self, vstar):
        if not self.conj_everywhere_finite:
            self._assert_conj_essential_domain(vstar,msg='_conj_subgradient')
        self._buf2 = vstar/self.measure
        if self.constr_l_active:
            vstar_small = (vstar<self.conj_taylor_l)
            if np.any(vstar_small):
                mask = np.land(mask,~vstar_small) if 'mask' in locals() else ~vstar_small  
                self._buf[vstar_small] = self.dom_l[vstar_small]
        if self.constr_u_active:
            vstar_large = (vstar>self.conj_taylor_u)
            if np.any(vstar_large):
                mask = np.land(mask,~vstar_large) if 'mask' in locals() else ~vstar_large
                self._buf[vstar_large] = self.dom_u[vstar_large]
        if 'mask' in locals():
            self._buf[mask] = self._f_conj_deriv(self._buf2[mask],mask=mask,**self.kwargs)
        else:
            self._buf = self._f_conj_deriv(self._buf2,**self.kwargs)
        return self._buf.copy()
    
    def _hessian(self, v):
        if not self.everywhere_finite:
            self._assert_essential_domain(v,msg='_hessian')
        self._buf = self.domain.ones()
        if self.conj_constr_l_active:
            self._buf[v<self.taylor_l] = 0.
        if self.conj_constr_u_active:
            self._buf[v>self.taylor_u] = 0.
        if self.conj_constr_l_active or self.conj_constr_u_active:
            mask = (self._buf==1)
            self._buf[mask] =  self._f_second_deriv(v[mask],mask=mask,**self.kwargs)
        else:
            self._buf =  self._f_second_deriv(v,**self.kwargs)
        self._buf *= self.measure
        return PtwMultiplication(self.domain,self._buf.copy())

    def _conj_hessian(self, vstar):
        if not self.conj_everywhere_finite:
            self._assert_conj_essential_domain(vstar,msg='_conj_hessian')
        self._buf2 = vstar/self.measure
        self._buf = self.domain.ones()
        if self.constr_l_active:
            self._buf[self._buf2<self.conj_taylor_l] = 0.
        if self.constr_u_active:
            self._buf[self._buf2>self.conj_taylor_u] = 0.   
        if self.constr_l_active or self.constr_u_active:   
            mask = (self._buf==1)
            self._buf[mask] = self._f_conj_second_deriv(self._buf2[mask],mask=mask,**self.kwargs)
        else:
            self._buf = self._f_conj_second_deriv(self._buf2,**self.kwargs)
        self._buf /= self.measure
        return PtwMultiplication(self.domain, self._buf.copy())
    
    def _proximal(self, v, tau,mask=None):
        if mask is None:
            res = self._f_prox(v,tau,**self.kwargs)
        else:
            res = self._f_prox(v,tau,mask=mask,**self.kwargs)

        if self.constr_l_active:
            res = np.maximum(res,self.dom_l[mask if mask is not None else slice(None)])
        if self.constr_u_active:
            res = np.minimum(res,self.dom_u[mask if mask is not None else slice(None)])
        
        if self.conj_constr_l_active:
            corr = (tau*self.conj_dom_l/self.measure)[mask if mask is not None else slice(None)]                
            res = np.minimum(res,v-corr,out = res)
        if self.conj_constr_u_active:
            corr = (tau*self.conj_dom_u/self.measure)[mask if mask is not None else slice(None)]               
            res = np.maximum(res,v-corr,out = res)
        return res
    
    def _conj_proximal(self, vstar, tau,mask=None):
        if mask is None:
            res = vstar/self.measure
            res = self._f_conj_prox(res,tau,**self.kwargs)
            res *= self.measure
        else:
            res = vstar/self.measure[mask]
            res = self._f_conj_prox(res,tau,mask=mask,**self.kwargs)
            res *= self.measure[mask]

        if self.conj_constr_l_active:
            res = np.maximum(res,self.conj_dom_l[mask if mask is not None else slice(None)])
        if self.constr_u_active:
            res = np.minimum(res,self.conj_dom_u[mask if mask is not None else slice(None)])

        if self.constr_l_active:
            corr = (tau*self.dom_l*self.measure)[mask if mask is not None else slice(None)]
            res = np.minimum(res,vstar-corr,out = res)
        if self.constr_u_active:
            corr = (tau*self.dom_u*self.measure)[mask if mask is not None else slice(None)]
            res = np.maximum(res,vstar-corr,out = res)   

        return res

    def _f(self,v,**kwargs):
        raise NotImplementedError
    
    def _f_deriv(self,v,**kwargs):
        raise NotImplementedError

    def _f_second_deriv(self,v,**kwargs):
        raise NotImplementedError

    def _f_prox(self,v,tau,tol=1e-12, maxNewtonIter=15,maxBisecIter=300,maxBoundsIter=100,**kwargs):
        if self.__class__.__dict__.get("_f_deriv") is not None:
            vclip = np.minimum(v,self.dom_u)
            vclip = np.maximum(vclip,self.dom_l)
            f_second_deriv = self._f_second_deriv if self._f_second_deriv is not None else None
            return self._numerical_prox(v,tau,self._f_deriv,f_second_deriv,self.dom_l, self.dom_u, 
                                        tol=tol,maxNewtonIter=maxNewtonIter,maxBisecIter=maxBisecIter,maxBoundsIter=maxBoundsIter,
                                        **kwargs
                                        )
        else:
            NotImplementedError('Need first derivative for numerical prox operator')
    
    def _f_conj(self,vstar,**kwargs):
        raise NotImplementedError
    
    def _f_conj_deriv(self,vstar,**kwargs):
        raise NotImplementedError

    def _f_conj_second_deriv(self,vstar,**kwargs):
        raise NotImplementedError

    def _f_conj_prox(self,vstar,tau,tol=1e-12, maxNewtonIter=15,maxBisecIter=300,maxBoundsIter=100,**kwargs):
        if self.__class__.__dict__.get("_f_conj_deriv") is not None:
            print('check existence of _conj_prox',self.__class__.__dict__.get("_f_conj_prox") is not None)
            vclip = np.minimum(vstar,self.conj_dom_u)
            vclip = np.maximum(vclip,self.conj_dom_l)
            vclip /= self.measure
            #self._f_conj_deriv(vclip,**kwargs)
            #self._f_conj_second_deriv(vclip,**kwargs)
            f_conj_second_deriv = self._f_conj_second_deriv if self._f_conj_second_deriv is not None else None
            return self._numerical_prox(vstar,tau,
                                        self._f_conj_deriv,f_conj_second_deriv,
                                        self.conj_dom_l/self.measure, 
                                        self.conj_dom_u/self.measure, 
                                        tol=tol,maxNewtonIter=maxNewtonIter,maxBisecIter=maxBisecIter,maxBoundsIter=maxBoundsIter,
                                        **kwargs
                                        )
        else:
        #except NotImplementedError:
            NotImplementedError('Need first derivative of conjugate functional for numerical conjugate prox operator')

    def _numerical_prox(self, y, tau, 
                        fp, fpp, dom_l, dom_u, 
                        start=None,ub=None, lb=None,
                        tol=1e-12, maxNewtonIter=15,maxBisecIter=300,maxBoundsIter=100,**kwargs):
        """
        Vectorized proximal operator of a smooth convex scalar function
        using Newton's method with automatic fallback to bisection.

        Parameters
        ----------
        y : ndarray
            Input array.
        tau : float
            Prox parameter (>0).
        f, fp, fpp : callables
            f, f', f'' (accept numpy arrays).
        dom_l, dom_u : floats
            Domain bounds (can be -np.inf, np.inf).
        start: np.ndarray or None
            starting point
        lb,ub: np.ndarray or None
            guesses for lower and upper bounds on solution (must not be valid!)
        tol : float
            Root-finding tolerance for Newton/bisection.
        maxiter : int
            Max Newton iterations before falling back.
        **kwargs:
            Passed to f,fp and fpp

        Returns
        -------
        prox : ndarray
            Proximal points, same shape as y.
        """

        
        assert isinstance(y,np.ndarray)
        if np.isscalar(dom_u):
            dom_u = np.full_like(y,dom_u)
        if np.isscalar(dom_l):
            dom_l = np.full_like(y,dom_l)    
        if not np.all(dom_l<=dom_u):
            raise ValueError('Upper/lower bounds of essential domain invalid.')
  

        #initial guess
        x = start if not (start is None) else y
        x = np.maximum(dom_l,x)
        x = np.minimum(x,dom_u)


        r = fp(x,**kwargs) + (x - y)/tau
        converged = (dom_u-dom_l<tol) # width of essential domain <tol
        converged = converged | (r==0) # optimality condition satisfied 
        converged = converged | ((x==dom_l) & (r>=0)) # optimality condition at left boundary satisfied
        converged = converged | ((x==dom_u) & (r<=0)) # optimality condition at right boundary satiesfied

        self.log.debug(f'initially {np.sum(converged)} converged.')
        # obtain valid and finite upper and lower bounds
        ub = x.copy()
        x_too_small = (r<0) & (~converged)       
        if np.any(x_too_small):
            xs = x[x_too_small]
            rs = r[x_too_small]
            ds = dom_u[x_too_small]
            up = np.minimum(tau*np.ones_like(rs),-tau*rs)
            up2 = np.minimum(xs+up,ds)
            ub[x_too_small] = up2
#            ub[x_too_small] = np.minimum(x[x_too_small] 
#                                      + np.minimum(np.ones_like(r[x_too_small]), - tau*r[x_too_small]),
#                                        dom_u[x_too_small]),
                                    
        lb = x.copy()
        x_too_large = (r>0) & (~converged)
        if np.any(x_too_large):
           lb[x_too_large] = np.maximum(x[x_too_large]
                                     - np.minimum(tau*np.ones_like(r[x_too_large]), tau*r[x_too_large]),
                                        dom_l[x_too_large])
        """
        if lb is None:
            lb = np.minimum(x,dom_u)-1.
        lb[lb==-np.inf]=x[lb==-np.inf]-1.
        lb = np.maximum(lb,dom_l)
        print(f'lb corrected: {lb}')
        if ub is None:
            ub = np.maximum(x,dom_l)+1.
        ub[ub==np.inf]=x[ub==np.inf]+1.
        ub = np.minimum(ub,dom_u)
        print(f'ub corrected: {ub}')
        """

        #self.log.debug(f'before correction: r: {r}\n ub: {ub}\n lb: {lb}\n diff: {ub-lb}')
        if not np.all(lb<=ub):
            ind = lb>ub
            raise RuntimeError(f'lb<=ub violated for input values {y[lb>ub]}. lb: {lb[lb>ub]}, ub: {ub[lb>ub]} ')
        # decrease lb where necessary to make it a valid lower bound
        for it in range(maxBoundsIter):
            v = fp(lb,**kwargs) + (lb- y)/tau            
            ind = (v>0) & ~converged
            if it==0 and not np.all(ub[ind]-lb[ind]>tol):
                ii = ind & (ub-lb<=tol)
                lb[ii] = np.maximum(lb[ii]-tol,dom_l[ii])
                self.log.debug(f'adding {np.sum(ii & (lb==dom_l))} indices as converged.')
                converged = converged | (ii & (lb==dom_l))
            self.log.debug(f'lower bound it {it}: {np.sum(ind)} indices invalid.')
            if np.sum(ind)==0:
                break
            else:
                lb[ind] = np.maximum(dom_l[ind],2*lb[ind]-ub[ind])
        if not np.sum(ind)==0:
            raise RuntimeError("Could not determine lower bound. Increase maxBoundsIter!") 
        r = fp(lb,**kwargs) + (lb - y)/tau 
        converged = converged | ((lb==dom_l) & (r>=0))
        if not np.all((r<=0) | converged):
            raise RuntimeError(f'lower bound not satisfied for indices {np.where((r>0) & ~converged)}')

        # increase ub where necessary to make it a valid upper bound
        for it in range(maxBoundsIter):
            v = fp(ub,**kwargs) + (ub- y)/tau            
            ind = (v<0) & ~converged
            self.log.debug(f'upper bound it {it}: {np.sum(ind)} indices invalid')
            if it==0 and not np.all(ub[ind]-lb[ind]>tol):
                ii = ind & (ub-lb<=tol)
                ub[ii] = np.minimum(dom_u[ii],lb[ii]+tol)
                self.log.debug(f'adding {np.sum(ii & (ub==dom_u))} indices as converged.')
                converged = converged | (ii & (ub==dom_u))
            if np.sum(ind)==0:
                break
            else:
                ub[ind] = np.minimum(dom_u[ind], 2*ub[ind]-lb[ind])
        if not np.sum(ind)==0:
            raise RuntimeError("Could not determine upper bound. Increase maxBoundsIter!")
        r = fp(ub,**kwargs) + (ub - y)/tau 
        converged = converged | ((ub==dom_u) & (r<=0))
        if not np.all((r>=0) | converged): 
            raise RuntimeError(f'upper bound not satisfied for indices {np.where((r<0) & ~converged)}')
        assert np.all(lb>=dom_l)
        assert np.all(ub<=dom_u)

        self.log.debug(f'lb: {lb}\n diff: {ub-lb}')

        # --- Newton phase ---
        if fpp is not None:
            iter=0
            for iter in range(maxNewtonIter):
                d = fpp(x,**kwargs) + 1.0/tau
                step = r / d

                xnew = x - step
                xnew = np.maximum(xnew, lb)
                xnew = np.minimum(xnew, ub)
        
                summand1 = fp(x,**kwargs)
                summand2 = (x - y)/tau
                r = summand1 + summand2
                lb[r<=0] = x[r<=0]
                lb[r>0]  = np.maximum(lb[r>0],x[r>0] - tau*r[r>0])
                ub[r>=0] = x[r>=0]
                ub[r<0]  = np.minimum(ub[r<0],x[r<0] - tau*r[r<0])
                # Check convergence
                #conv = np.abs(r) < tol
                conv = np.abs(ub-lb) < tol
                converged = converged | conv
                x = np.where(conv, x, xnew)
                self.log.debug(f'Newton it {iter}: {np.sum(converged==True)} out of {len(x)} converged.')
    
                if np.all(converged):
                    return x
                if np.allclose(summand1,-summand2,rtol=1e-14):
                    self.log.info('Required tolerance cannot be guaranteed since prox of scalar function is too ill-conditioned.')
                    break

            self.log.debug(f'diff ub-lb after Newton: {ub-lb}')

        # --- Fallback to bisection for non-converged entries ---
        mask = ~converged
        if np.any(mask):
            yi = y[mask]
            lb = lb[mask]
            ub = ub[mask]

            for iter in range(maxBisecIter):  # max bisection iters
                Mi = 0.5*(lb+ub)
                fM = fp(Mi,mask=mask,**kwargs) + (Mi - yi)/tau
                right = fM > 0
                ub = np.where(right, Mi, ub)
                lb = np.where(~right, Mi, lb)
                self.log.debug(f'bisection it. {iter}: {np.sum(ub-lb>tol)} not converged')
                if np.all((ub-lb) < tol):
                    break
            x[mask] = 0.5*(lb+ub)
            if np.max(ub-lb)>tol:
                raise RuntimeError('Could not satisfy tolerance criterium in bisection algorithm.')

        return x        

class LppPower(IntegralFunctionalBase):
    r"""
    Implements the norm power functional \(v\mapsto \frac{1}{p} \|v\|_{L^p}^p$-power on some domain in `MeasureSpaceFcts`
    as an integral functional. This corresponds to the function \(f(v):=\frac{1}{p}|v|^p)\.

    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a MeasureSpaceFcts
    p: float >1 [option]
        exponent
    constr_l,constr_u,lin_taylor_l, lin_taylor_u: None, np.isscalar or np.ndarray
        see IntegralFunctional  
    """

    def __init__(self, domain, p=2.,
                 constr_l=None, constr_u=None, lin_taylor_l=None, lin_taylor_u=None,
                  quad_taylor_l=None, quad_taylor_u=None,
                 **kwargs):
        assert np.isscalar(p) and p >1.
        self.p = p
        self.q = p/(p-1)

        dom_l = -np.inf if constr_l is None else constr_l        
        dom_u = np.inf if constr_u is None else constr_u
        taylor_l = -np.inf if lin_taylor_l is None else lin_taylor_l
        taylor_u = np.inf if lin_taylor_u is None else lin_taylor_u

        if p<2:
            aux = np.max(np.maximum(-dom_l,dom_u))
            convexity_param = (self.p-1) * aux**(self.p-2) if aux<np.inf else 0

            aux = np.min(np.minimum(taylor_l,-taylor_u))
            Lipschitz = np.maximum((self.p-1) * aux**(self.p-2),aux**(self.p-1)) if aux>0 else np.inf

        if p>2:
            aux = np.min(np.minimum(dom_l, -dom_u))
            convexity_param = (self.p-1) * aux**(self.p-2)  if aux>0 else 0

            aux =  np.max(np.maximum(-taylor_l,taylor_u))
            Lipschitz = np.maximum((self.p-1) * np.abs(aux)**(self.p-2),np.abs(aux)**(self.p-1)) if aux<np.inf else np.inf

        if p==2:
            convexity_param = 1.
            Lipschitz = 1.

        super().__init__(domain, 
                         convexity_param=convexity_param,
                         Lipschitz = Lipschitz,
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                        quad_taylor_l=quad_taylor_l, quad_taylor_u=quad_taylor_u,
                         **kwargs
                         )

    def _f(self,v,**kwargs):
        # member efficient implementation of 
        # res = np.abs(v)**self.p/self.p
        res = np.abs(v)
        np.power(res,self.p,out=res)
        res /=self.p
        return res
    
    def _f_deriv(self, v,**kwargs):
        # member efficient implementation of  
        # res = np.abs(v)**(self.p-1)*np.sign(v) 
        res = np.abs(v)
        res = np.power(res,self.p-1) 
        aux = np.sign(v)
        res *= aux
        return res
    
    def _f_second_deriv(self, v,**kwargs):
        # member efficient implementation of  
        # res = (self.p-1)*np.abs(v)**(self.p-2)
        res = np.abs(v)
        res = np.power(res,self.p-2,out=res)        
        res *= (self.p-1)
        return res
    
    def _f_prox(self,v,tau,**kwargs):
        if self.p==2:
            return v/(1+tau)
        else:
            return self._numerical_prox(v,tau,
                                self._f_deriv,self._f_second_deriv,self.dom_l, self.dom_u, 
                                tol=1e-12,maxNewtonIter=10,maxBisecIter=300,maxBoundsIter=300,
                                **kwargs
                                )
    
    def _f_conj(self, vstar,**kwargs):
        # member efficient implementation of
        # res = np.abs(vstar)**self.q/self.q
        res = np.abs(vstar)
        np.power(res,self.q,out=res)     
        res /= self.q
        return res 

    def _f_conj_deriv(self, vstar,**kwargs):
        if not hasattr(self, '_aux'):
            self._aux = self.domain.zeros()        
        # member efficient implementation of 
        # res = np.abs(vstar)**(self.q-1)*np.sign(vstar) 
        res = np.abs(vstar)
        np.power(res,self.q-1,out=res)     
        self._aux = np.sign(vstar)
        res *= self._aux       
        return res
    
    def _f_conj_second_deriv(self, vstar,**kwargs):
        # member efficient implementation of 
        # res = (self.q-1)*np.abs(vstar)**(self.q-2)   
        res = np.abs(vstar)
        np.power(res,self.q-2,out=res)     
        res *= (self.q-1)
        return res
    
    def _f_conj_prox(self,v_star,tau,**kwargs):
        if self.p==2:
            return v_star/(1+tau)
        else:
            return self._numerical_prox(v_star,tau,
                                self._f_conj_deriv,self._f_conj_second_deriv,
                                np.broadcast_to(-inf,self.domain.shape), 
                                np.broadcast_to(inf,self.domain.shape), 
                                tol=1e-12,maxNewtonIter=10,maxBisecIter=300,maxBoundsIter=300,
                                **kwargs
                                )

class L1MeasureSpace(IntegralFunctionalBase):
    r""":math:`L ^1` Functional on `MeasureSpace`. Proximal implemented for default :math:`L^2` as `h_domain`.

    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        Domain on which to define the generic L1.
    constr_l,constr_u,lin_taylor_l, lin_taylor_u: None, np.isscalar or np.ndarray
        see IntegralFunctional          
    """
    def __init__(self, domain,
                constr_l=None, constr_u=None, lin_taylor_l=None, lin_taylor_u=None,
                quad_taylor_l=None, quad_taylor_u=None):
        super().__init__(domain,conj_dom_u=1.,conj_dom_l=-1.,
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                         quad_taylor_l=quad_taylor_l, quad_taylor_u=quad_taylor_u)

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
        if not hasattr(self, '_aux'):
            self._aux = self.domain.zeros()    
        # res = np.maximum(0, np.abs(v)-tau)*np.sign(v)        
        res = np.abs(v)
        res -= tau
        np.maximum(0,res,out=res)
        self._aux = np.sign(v)
        res *= self._aux
        return res

    def _f_conj(self, v_star,**kwargs):
        res = np.abs(v_star)    
        ind = (res>1)
        res *= 0.
        res[ind]= inf
        return res
    
    def _f_conj_deriv(self, v_star,**kwargs):
        return np.zeros_like(v_star)

    def _f_conj_second_deriv(self, v_star,**kwargs):
        return self.domain.zeros()

    def _f_conj_prox(self,vstar,tau,**kwargs):
        # res  = vstar/np.maximum(np.abs(vstar),1)
        #res = np.abs(vstar)
        #res = np.maximum(res,1.,out=res)
        #np.divide(vstar,res, out =res)
        res = np.minimum(vstar,1.)
        np.maximum(res,-1.,out=res)
        return res

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
        F_w(u) = KL(w,u) = \int (u(x) -w(x) - w(x)\ln \frac{u(x)}{w(x)}) \mathrm{d}x


    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        Domain on which to define the Kullback-Leibler divergence
    w: domain 
        First argument of Kullback-Leibler divergence.
    constr_l,constr_u,lin_taylor_l, lin_taylor_u: None, np.isscalar or np.ndarray
        see IntegralFunctional  
    """

    def __init__(self, domain, w,
                 constr_l=None, constr_u=None, lin_taylor_l=None, lin_taylor_u=None,
                 quad_taylor_l=None, quad_taylor_u=None):
        if not w in domain:
            raise ValueError('w not in domain.')
        if np.min(w)<0:
            raise ValueError('w must be non-negative.')
        self.w = w
        Lipschitz =  np.max(np.maximum(w/lin_taylor_l**2,np.abs(1-w/lin_taylor_l))) if lin_taylor_l is not None else np.inf
        convexity_param = np.min(w/constr_u**2) if constr_u is not None else 0
        super().__init__(domain,dom_l=1e-14*w,
                         conj_dom_u=domain.ones()-1e-14*w,
                         convexity_param = convexity_param, Lipschitz= Lipschitz,                              
                          constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                          quad_taylor_l=quad_taylor_l,quad_taylor_u=quad_taylor_u
                         )

    def _f(self, u,**kwargs):
        if 'w' in kwargs.keys():
            raise ValueError('second parameter w of KullbackLeibler must be fixed in constructor.')
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        res=np.zeros_like(u)
        # memory efficient implementation of 
        # res[ind_else]=u[ind_else]-self.w[ind_else] - self.w[ind_else] * np.log(u[ind_else]/self.w[ind_else])
        np.divide(u,wm, out=res)
        with np.errstate(invalid='ignore', divide='ignore'):
            np.log(res,out=res)
        res *= wm
        res *= -1
        res += u
        res -= wm
        # end
        res[(u<0)|((u==0)&(wm>0))]= np.inf
        return res    
   
    def _f_deriv(self, u,**kwargs):
        if 'w' in kwargs.keys():
            raise ValueError('first parameter w of KullbackLeibler must be fixed in constructor.')
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        if not np.all(np.logical_or(np.logical_not(u==0),wm==0)):
            raise ValueError('argument cannot be 0 at positions where w is not 0')
        # memory efficient implementation of 
        # res = np.ones_like(u)-wm/u
        res = np.divide(wm,u)
        res *= -1.
        res += 1.
        # end
        res[u==0] = 1
        return res

    def _f_second_deriv(self, u, **kwargs):
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w   
        # memory efficient computation of 
        # res = wm/u**2
        res = np.divide(wm,u)
        res /= u
        # end
        return res

    def _f_conj(self, u_star,**kwargs):
        if 'w' in kwargs.keys():
            raise ValueError('second parameter w of KullbackLeibler must be fixed in constructor.')
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        # memory efficient computation of
        # res = -wm*np.log(1-u_star)
        res = np.subtract(1.,u_star)
        with np.errstate(invalid='ignore', divide='ignore'):
            np.log(res,out=res)
        res *= wm
        res *= -1
        # end
        res[(u_star>1) | ((u_star == 1) & (wm>0))] = np.inf
        return res

    def _f_conj_deriv(self, u_star,**kwargs):
        if 'w' in kwargs.keys():
            raise ValueError('second parameter w of KullbackLeibler must be fixed in constructor.')
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w    
        if not np.all(np.logical_or(np.logical_not(u_star==1),wm==0)):
            raise ValueError('argument cannot be 1 at positions where w is not 0.')
        # memory efficient implementation of
        # toret = wm/(1-u_star)
        toret = np.subtract(1.,u_star)
        np.divide(wm,toret,out = toret)
        # end
        toret[u_star==1] = 0
        return toret 
    
    def _f_conj_second_deriv(self, u_star,**kwargs):
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        assert np.all(np.logical_or(np.logical_not(u_star==1),wm==0))
        # memory efficient implementation of 
        # toret = wm/(1-u_star)**2
        toret = np.subtract(1.,u_star)
        toret *= toret
        np.divide(wm,toret,out = toret)        
        # end
        return toret

    def _f_prox(self, v, tau, **kwargs):
        # memory efficient implementation of 
        # toret = -0.5*(tau-v) + np.sqrt(0.25*(tau-v)**2+tau*self.w)
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        toret = np.subtract(tau,v)
        toret *= toret
        toret *= 0.25
        aux = np.multiply(tau,wm)
        toret += aux
        toret = np.sqrt(toret,out=toret)
        aux = np.subtract(tau,v)
        aux *= -0.5
        toret += aux
        # end
        return toret

    def _f_conj_prox(self, vstar, tau, **kwargs):
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        # memory efficient implementation of 
        # toret = 0.5*(1.+vstar) - np.sqrt(0.25*(1.+vstar)**2 + tau*self.w-vstar)
        toret = np.add(1.,vstar)
        toret *= toret
        toret *= 0.25
        aux = np.multiply(tau,wm)
        aux -= vstar
        toret += aux
        np.sqrt(toret,out=toret)
        toret *= -1
        aux = np.add(1.,vstar,out=aux)
        aux *= 0.5
        toret += aux
        # end
        return toret


class RelativeEntropy(IntegralFunctionalBase):
    r"""Kullback-Leiber divergence define by

    .. math::
        F_w(u) = KL(u,w) = \int (u(x)\ln \frac{u(x)}{w(x)}) \mathrm{d}x

    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        Domain on which to define the Kullback-Leibler divergence
    w: scalar or in domain [optional, default: 1]
        second argument of the Kullback-Leibler diverengence; reference value if used as penalty functional
    constr_l,constr_u,lin_taylor_l, lin_taylor_u: None, scalar or in domain
        see IntegralFunctional  
    """

    def  __init__(self, domain,w=1.,
                  constr_l=None, constr_u=None, lin_taylor_l=None, lin_taylor_u=None,
                  quad_taylor_l=None, quad_taylor_u=None):
        if np.isscalar(w):
            w = np.broadcast_to(w,domain.shape)
        assert w in domain
        assert np.min(w)>0
        self.w= w
        convexity_param = np.min(1/constr_u) if constr_u is not None else 0
        Lipschitz = np.max(np.maximum(1/lin_taylor_l,np.abs(1.+np.log(lin_taylor_l/w)))) if lin_taylor_l is not None else np.inf
        super().__init__(domain,dom_l = 1e-14*w,
                         convexity_param=convexity_param, Lipschitz= Lipschitz, 
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                         quad_taylor_l=quad_taylor_l,quad_taylor_u=quad_taylor_u)

    def _f(self, u,**kwargs):
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        res=np.zeros_like(u)
        # memory efficient implementation of 
        # res[ind_upos]=u[ind_upos] * np.log(u[ind_upos]/wm[ind_upos])
        np.divide(u,wm, out= res)
        with np.errstate(invalid='ignore', divide='ignore'):    
            np.log(res,out=res)
        res *= u
        # end
        res[u<0] = np.inf
        res[u==0] = 0.
        return res    
   
    def _f_deriv(self, u,**kwargs):
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        # memory efficient implementation of 
        # res = np.ones_like(u)+np.log(u/wm)
        res = np.divide(u,wm)
        np.log(res,out=res)
        res += 1.
        # end
        return res

    def _f_second_deriv(self, u, **kwargs):     
        return 1/u

    def _f_prox(self, v, tau, **kwargs):
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        
        v_mod = (v<=700*tau) 
        # For v>=710*tau an overflow occurs in the exponential. 
        # For such values we use an approximation via linearization (= one Newon step) instead of the exact formula in terms of the Lambert-w function. 
        
        # memory efficient implementation of 
        # toret = (1/tau)*self.w*np.exp(v/tau-1.)
        toret = np.divide(v[v_mod],tau)
        toret -= 1.
        np.exp(toret,out=toret)
        toret *= wm[v_mod]
        toret /= tau
        #end
        #if not hasattr(self, '_aux'):
        #    self._aux = self.domain.complex_space().zeros()        
        aux =  lambertw(toret)
        toret = aux.real
        toret *= tau

        if np.all(v_mod):
            return toret.reshape(v.shape)
        else:
            res = np.zeros_like(v)
            res[v_mod] = toret
            vl = v[~v_mod]       
            res[~v_mod] = (vl - tau*np.log(vl/wm[~v_mod])) / (1. + tau/vl)
            return res

    def _f_conj_prox(self, vstar, tau, **kwargs):
        toret = (1/tau)*vstar
        aux = self._f_prox(toret,1/tau, **kwargs)
        aux *= tau
        toret *= tau
        toret -= aux
        return toret

    def _f_conj(self, u_star,**kwargs):  
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        # memory efficient implementation of 
        # toret =  wm*(np.exp(u_star-1))
        toret = np.subtract(u_star,1.)
        np.exp(toret,out=toret)
        toret *= wm
        # end
        return toret

    def _f_conj_deriv(self, u_star,**kwargs):
        return self._f_conj(u_star,**kwargs)
    
    def _f_conj_second_deriv(self, u_star,**kwargs):
        return self._f_conj(u_star,**kwargs)

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

    def  __init__(self, domain,as_primal=True,sigma = 1.,eps=1e-10,**kwargs):
        assert isinstance(sigma, (float,int)) or sigma in domain
        if isinstance(sigma, (float,int)) :
            self.sigma = np.broadcast_to(np.real(sigma),domain.shape)
        else:
            self.sigma = np.real(sigma)
        if np.min(sigma)<=0:
            raise ValueError(f'sigma must be positive. min(sigma)={np.min(sigma)}')
        if as_primal:
            super().__init__(domain,Lipschitz=1,
                             conj_dom_l=-self.sigma, conj_dom_u = self.sigma,
                             **kwargs)
            self.conjugate = QuadraticIntv(domain,as_primal=False,sigma=sigma,eps=eps)
        else:
            dual_domain = deepcopy(domain)
            dual_domain.measure = 1./domain.measure
            super().__init__(dual_domain, Lipschitz=1, **kwargs)
        # auxiliary vectors
        self._abs_u = self.domain.zeros() 
        self._small = np.zeros(self.domain.shape,dtype=bool)

    def _f(self, u,**kwargs):
        sigma = self.sigma[kwargs['mask']] if ('mask' in kwargs.keys() and not np.isscalar(self.sigma)) else self.sigma
        # res = np.where(np.abs(u)<=sigma,0.5*np.abs(u)**2,sigma*np.abs(u)-0.5*sigma**2)
        self._abs_u = np.abs(u)
        self._small = (self._abs_u<=sigma)
        res = np.multiply(self._abs_u,2.)
        res -= sigma
        res *= sigma
        res *= 0.5
        self._abs_u *=self._abs_u
        self._abs_u *= 0.5
        res[self._small] = self._abs_u[self._small]
        return res 

    def _f_deriv(self, u,**kwargs):
        sigma = self.sigma[kwargs['mask']] if ('mask' in kwargs.keys() and not np.isscalar(self.sigma)) else self.sigma
        # res = np.where(np.abs(u)<=sigma,u,sigma*u/np.abs(u))
        self._abs_u = np.abs(u)
        self._small = (self._abs_u<=sigma)
        res = sigma*u
        with np.errstate(invalid='ignore',divide='ignore'):
           res /= self._abs_u
        res[self._small] = u[self._small]
        return res

    def _f_second_deriv(self, u, **kwargs):
        sigma = self.sigma[kwargs['mask']] if ('mask' in kwargs.keys() and not np.isscalar(self.sigma)) else self.sigma
        return (np.abs(u)<=sigma).astype(float)

    def _f_conj(self, ustar,**kwargs):
        return self.conjugate._f(ustar,**kwargs)    
   
    def _f_conj_deriv(self, ustar,**kwargs):
        return self.conjugate._f_deriv(ustar,**kwargs)

    def _f_conj_second_deriv(self, ustar,**kwargs):
        return self.conjugate._f_second_deriv(ustar,**kwargs)

    def _f_conj_prox(self,ustar,tau,**kwargs):
        return self.conjugate._f_prox(ustar,tau,**kwargs)


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

    def  __init__(self, domain,as_primal=True,sigma=1.,eps=1e-10,**kwargs):
        assert isinstance(sigma, (float,int)) or sigma in domain 
        assert np.min(sigma)>0
        self.eps=eps
        if isinstance(sigma, (float,int)):
            self.sigma = np.broadcast_to(np.real(sigma), domain.shape)
            self.sigmaeps = np.broadcast_to(sigma*(1+eps), domain.shape)           
        else:
            self.sigma = sigma 
            self.sigmaeps = self.sigma*(1+eps) if eps>0 else self.sigma            
        if as_primal:
            super().__init__(domain,convexity_param=1,dom_l=-self.sigmaeps,dom_u=self.sigmaeps,**kwargs)
            self.conjugate = Huber(domain,as_primal=False,sigma=sigma)
        else:
            dual_domain = deepcopy(domain)
            dual_domain.measure = 1./domain.measure
            super().__init__(dual_domain, convexity_param=1,**kwargs)
        self._aux = domain.zeros()

    def _f(self, u,**kwargs):
        # res =  0.5*np.abs(u)**2
        self._aux = np.abs(u)
        res = self._aux**2
        res *= 0.5 
        res[self._aux>self.sigmaeps] = np.inf        
        return res  
   
    def _f_deriv(self, u,**kwargs):
        return u.copy()

    def _f_prox(self,u,tau,**kwargs):
        res = u/(1+tau)
        self._aux = np.abs(res)
        self._aux /= self.sigma
        return res/np.maximum(self._aux,1)

    def _f_second_deriv(self, u,**kwargs):
        return np.ones_like(u)

    def _f_conj(self, ustar,**kwargs):
        return self.conjugate._f(ustar,**kwargs)    
   
    def _f_conj_deriv(self, ustar,**kwargs):
        return self.conjugate._f_deriv(ustar,**kwargs)

    def _f_conj_second_deriv(self, ustar,**kwargs):
        return self.conjugate._f_second_deriv(ustar,**kwargs)

    def _f_conj_prox(self,ustar,tau,**kwargs):
        return self.conjugate._f_prox(ustar,tau,**kwargs)

    def is_subgradient(self, vstar, x, eps=1e-10):
        grad = self.subgradient(x)
        self._aux = np.abs(x)
        if(not np.all(self._aux<=self.sigma)):
            return False
        if(not np.all(vstar[self.sigma==x]>=self.sigma)):
            return False
        if(not np.all(vstar[-self.sigma==x]<=-self.sigma)):
            return False
        ind = (self._aux<self.sigma)
        if(np.linalg.norm(grad[ind]-vstar[ind]) <= eps*np.linalg.norm(grad[ind])):
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

    def  __init__(self, domain,**kwargs):
        super().__init__(domain,convexity_param = 1.,dom_l=0.,**kwargs)

    def _f(self, u,**kwargs):
        res =  u*u
        res *= 0.5
        res[u<0] = np.inf
        return res    

    def _f_deriv(self, u,**kwargs):
        return u.copy()

    def _f_prox(self,u,tau,**kwargs):
        res = u/(1+tau)
        np.maximum(res,0,out=res)
        return res

    def _f_second_deriv(self, u,**kwargs):
        return np.ones_like(u)

    def _f_conj(self, ustar,**kwargs):
        res = ustar*ustar
        res *= 0.5
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
        xnonneg = (x>=0)
        return np.max(vstar[~xnonneg])<=0 and np.linalg.norm(x[xnonneg]-vstar[xnonneg]) <= eps*np.linalg.norm(x[xnonneg])
    

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

    def __init__(self,domain, lb=None, ub=None, x0=None,alpha=1.,**kwargs):
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
        F = QuadraticIntv(domain,sigma=(ub-lb)/2.,**kwargs)
        center = (ub+lb)/2
        lin = LinearFunctional(center-x0,
                            domain=domain,
                            gradient_in_dual_space=False,
                            **kwargs
                            )
        offset = 0.5*(np.sum((x0**2-center**2)*domain.measure))
        # return  alpha*HorizontalShiftDilation(F,shift=center) + alpha*lin + alpha*offset
        super().__init__((alpha,HorizontalShiftDilation(F,shift=center)+offset),
                          (alpha,lin),
                          **kwargs
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

    def  __init__(self, domain,trace_val=None,tol=1e-15,**kwargs):
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
        super().__init__(domain,Lipschitz=1,convexity_param=1,**kwargs)

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

    def _f_prox(self, x, tau):
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