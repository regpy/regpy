from math import inf
from functools import partial
from copy import deepcopy

import numpy as np
from scipy.linalg import ishermitian
from scipy.special import lambertw

from regpy.util import Errors, memoized_property
from regpy.vecsps.numpy import *
import regpy.operators as rop
from regpy.hilbert import L2

from .base import Functional, Conj, LinearFunctional,LinearCombination,HorizontalShiftDilation,NotInEssentialDomainError,NotTwiceDifferentiableError, AbstractFunctional, Composed

__all__ = ["IntegralFunctionalBase","VectorIntegralFunctional","LppL2","L1L2","HuberL2","LppPower","L1MeasureSpace","KullbackLeibler","RelativeEntropy","Huber","QuadraticIntv","QuadraticBilateralConstraints","QuadraticLowerBound","QuadraticNonneg","QuadraticPositiveSemidef","L1Generic","TVGeneric","TVUniformGridFcts"]


class IntegralFunctionalBase(Functional):
    r"""
    This class provides a general framework for integral functionals of the type
    
    .. math::
        F\colon X &\to \mathbb{R} \\
        v&\mapsto \int_\Omega f(v(x),x)\mathrm{d}x

    with :math:`f\colon \mathbb{R}^2\to \mathbb{R})`. 

    Subclasses defining explicit functionals of this type have to implement
     * `_f` evaluation the function :math:`f`
     * `_f_deriv` the derivative :math:`\partial_v f`
     * `_f_second_deriv' the second derivative :math:`\partial f^2/\partial v^2` (often not needed!)
     * `_f_prox` giving the proximal function :math:`\mathrm{prox}_{\tau f(.,x)}` for each :math:`x\in\Omega`
     * `_f_conj` evaluation the Fenchel conjugate function :math:`f^*(v^*,x)`
     * `_f_conj_deriv` the derivative :math:`\partial_{v^*}f*`
     * `_f_conj_second_deriv' the second derivative :math:`\partial f*^2/\partial v_*^2` (often not needed!)
     * `_f_conj_prox` giving the proximal  function :math:`\mathrm{prox}_{\tau f^*(\cdot,x)}`     
    
    since 

    .. math::
        F'[g]h = \int_\Omega h(x)(\partial_1 f)(g(x),x)

    is a functional of the same type, and

    .. math::
        \mathrm{prox}_{\tau F}(v)(x) = \mathrm{prox}_{\tau f(\cdot,x)}(v(x)).


    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a `regpy.vecsps.MeasureSpaceFcts`
    dom_l,dom_u : float or np.ndarray, default: -np.inf, np.inf
        lower and upper bound on the essential domain of :math:`f` (the interval on which :math:`f` is finite)
        If dom_l or dom_u are finite, :math:`f` should be finite at these points (possibly very large if :math:`f` tends to infinity there) 
        If the domain depends on the point `x` and/or arguments in `**kwargs`, this should be a numpy array.
    conj_dom_l,conj_dom_u : float or np.ndarray [default: -np.inf and np.inf, rsp.]
        lower and upper bound on the essential domain of the conjugate of :math:`f` (the interval on on which the conjugate :math:`f^*` is finite)    
        (as vector in primal space)
    constr_u: None or float or np.ndarray [default: None]
        If not None, an upper constraint is imposed, i.e. :math:`f(v,x)` is replaced by a function that takes the value `np.inf` 
        if `v>contr_u(x)`.
    constr_l: None or float or np.ndarray [default: None]
        As `constr_u`, but for a lower constraint. 
    lin_taylor_u: None or float or np.ndarray [default: None]
        If not None, :math:`f(v,x)` is replaced by its first order Taylor expansion 
        :math: `f(r(x),x) + (v-r(x)) \partial_v f(r(x),x) ` if :math:`x>r(x):=lin_taylor_u(x)`
    lin_taylor_l: None or float or np.ndarray [default: None]
        Analogous to right linearization, but for small values of :math:`v`.
    quad_taylor_u: None or float or np.ndarray [default: None]
        Analogous to lin_taylor_u, but with a quadratic Taylor expansion
    quad_taylor_l: None or float or np.ndarray [default: None]
        Analogous to quad_taylor_u, but for small values of :math:`v`
    methods, conj_methods = set of strings or None [default: None]
        Names of the methods implemented by an IntegralFunctionalBase instance (see Functional!).
        In the default case all methods are indicated as being implemented. 
    """

    def __init__(self,domain,
                 dom_l=-np.inf, dom_u=np.inf, 
                 conj_dom_l=-np.inf,conj_dom_u=np.inf,
                 constr_l=None, constr_u=None,
                 lin_taylor_l=None, lin_taylor_u=None,
                 quad_taylor_l=None, quad_taylor_u= None,
                 Lipschitz = np.inf, convexity_param=0.,
                 methods = None, conj_methods = None,
                 **kwargs):
        if not isinstance(domain,MeasureSpaceFcts): raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,add_info="Integral functionals are only implemented for MeasureSPaceFcts domains."))
        self.h_domain = L2(domain)
        self.measure = np.broadcast_to(domain.measure,domain.shape) if np.isscalar(domain.measure) else domain.measure
        self.domain = domain

        if sum([trunc is not None for trunc in [constr_l,lin_taylor_l,quad_taylor_l]])>1:
            raise ValueError(Errors.value_error('At most one of the parameters constr_l,lin_taylor_l, quad_taylor_l may be specified.',self))
        if sum([trunc is not None for trunc in [constr_u,lin_taylor_u,quad_taylor_u]])>1:
            raise ValueError(Errors.value_error('At most one of the parameters constr_u,lin_taylor_u, quad_taylor_u may be specified.',self))     

        if np.any(dom_l != -np.inf) and not np.isscalar(dom_l) and dom_l not in domain: 
            raise TypeError(Errors.type_error(f"The lower bound of the essential domain if defined, can only be either a scaler or an element in the domain. Given dom_l = {dom_l}"))
        if np.any(dom_u != np.inf) and not np.isscalar(dom_u) and dom_u not in domain: 
            raise TypeError(Errors.type_error(f"The upper bound of the essential domain if defined, can only be either a scaler or an element in the domain. Given dom_u = {dom_u}"))
        if constr_l is not None and not np.isscalar(constr_l) and constr_l not in domain: 
            raise TypeError(Errors.type_error(f"The lower constraint if defined, can only be either a scaler or an element in the domain. Given constr_l = {constr_l}"))
        if constr_u is not None and not np.isscalar(constr_u) and constr_u not in domain: 
            raise TypeError(Errors.type_error(f"The upper constraint if defined, can only be either a scaler or an element in the domain. Given constr_u = {constr_u}"))
        if lin_taylor_l is not None and not np.isscalar(lin_taylor_l) and lin_taylor_l not in domain: 
            raise TypeError(Errors.type_error(f"The lower point at which to replace with an Taylor first order expansion if defined, can only be either a scaler or an element in the domain. Given lin_taylor_l = {lin_taylor_l}"))
        if lin_taylor_u is not None and not np.isscalar(lin_taylor_u) and lin_taylor_u not in domain: 
            raise TypeError(Errors.type_error(f"The upper point at which to replace with an Taylor first order expansion if defined, can only be either a scaler or an element in the domain. Given lin_taylor_u = {lin_taylor_u}"))
        if quad_taylor_l is not None and not np.isscalar(quad_taylor_l) and quad_taylor_l not in domain: 
            raise TypeError(Errors.type_error(f"The lower point at which to replace with an Taylor second order expansion if defined, can only be either a scaler or an element in the domain. Given quad_taylor_l = {quad_taylor_l}"))
        if quad_taylor_u is not None and not np.isscalar(quad_taylor_u) and quad_taylor_u not in domain: 
            raise TypeError(Errors.type_error(f"The upper point at which to replace with an Taylor second order expansion if defined, can only be either a scaler or an element in the domain. Given quad_taylor_u = {quad_taylor_u}"))

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
            if constr_l is not None:
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

        all_methods = {'eval', 'subgradient', 'hessian', 'proximal', 'dist_subdiff'}

        super().__init__(domain,Lipschitz=Lipschitz,convexity_param=convexity_param,
                         separable=True,
                         dom_l = dom_l if dom_l in domain else np.broadcast_to(dom_l,domain.shape),
                         dom_u = dom_u if dom_u in domain else np.broadcast_to(dom_u,domain.shape),
                         conj_dom_l = conj_dom_l if conj_dom_l in domain else np.broadcast_to(conj_dom_l,domain.shape), 
                         conj_dom_u = conj_dom_u if conj_dom_u in domain else np.broadcast_to(conj_dom_u,domain.shape),
                         methods = methods if methods is not None else all_methods, 
                         conj_methods = conj_methods if conj_methods is not None else all_methods
                         )

        if self.constr_l_active:
            self.conj_taylor_l = np.full(domain.shape,-np.inf)
            self.conj_taylor_l[active_ind_l] = self._f_deriv(dom_l[active_ind_l],mask=active_ind_l,**kwargs)*self.measure[active_ind_l]
            self._if_constant_broadcast(self.conj_taylor_l)

            self.f_l = self.domain.zeros()
            self.f_l[active_ind_l] = self._f(dom_l[active_ind_l],mask=active_ind_l,**kwargs)
            self._if_constant_broadcast(self.f_l,mask=active_ind_l)

            self.conj_dom_l = np.array(self.conj_dom_l)
            self.conj_dom_l[active_ind_l] = -np.inf
            self._if_constant_broadcast(self.conj_dom_l)

        if self.constr_u_active:
            self.conj_taylor_u = np.full(domain.shape,np.inf)
            self.conj_taylor_u[active_ind_u] = self._f_deriv(dom_u[active_ind_u],mask=active_ind_u,**kwargs)*self.measure[active_ind_u]
            self._if_constant_broadcast(self.conj_taylor_u)

            self.f_u = self.domain.zeros()
            self.f_u[active_ind_u] = self._f(dom_u[active_ind_u],mask=active_ind_u,**kwargs)
            self._if_constant_broadcast(self.f_u,mask=active_ind_u)

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
                
                self.conj_f_l = domain.zeros()
                self.conj_f_l[conj_active_ind_l] = self._f_conj(self.conj_dom_l[conj_active_ind_l]/self.measure[conj_active_ind_l],mask=conj_active_ind_l,**kwargs)
                self._if_constant_broadcast(self.conj_f_l,mask=conj_active_ind_l)
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

                self.conj_f_u = domain.zeros()
                self.conj_f_u[conj_active_ind_u] = self._f_conj(self.conj_dom_u[conj_active_ind_u]/self.measure[conj_active_ind_u],mask=conj_active_ind_u,**kwargs)
                self._if_constant_broadcast(self.conj_f_u,mask=conj_active_ind_u)
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

                self.f_l = self.domain.zeros()
                self.f_l[taylor_active_l] = self._f(self.taylor_l[taylor_active_l],mask=taylor_active_l,**kwargs)
                self._if_constant_broadcast(self.f_l,mask=taylor_active_l)

                self.conj_f_l = domain.zeros()
                self.conj_f_l[taylor_active_l] = self._f_conj(self.conj_taylor_l[taylor_active_l]/self.measure[taylor_active_l],mask=taylor_active_l,**kwargs)
                self._if_constant_broadcast(self.conj_f_l,mask=taylor_active_l)

                self.fpp_l = domain.zeros()
                self.fpp_l[taylor_active_l] = self._f_second_deriv(self.taylor_l[taylor_active_l],mask=taylor_active_l,**kwargs)
                self._if_constant_broadcast(self.fpp_l,mask=taylor_active_l)
 
                self.fstarpp_l = domain.zeros()
                self.fstarpp_l[taylor_active_l] = self._f_conj_second_deriv(self.conj_taylor_l[taylor_active_l]/self.measure[taylor_active_l],mask=taylor_active_l,**kwargs)
                self._if_constant_broadcast(self.fstarpp_l,mask=taylor_active_l)

                self.conj_dom_l = np.array(self.conj_dom_l)
                self.conj_dom_l[taylor_active_l] =  -np.inf
                self._if_constant_broadcast(self.conj_dom_l)


        if quad_taylor_u is not None:
            self.taylor_u = quad_taylor_u if quad_taylor_u in domain else np.broadcast_to(quad_taylor_u,domain.shape)
            self.conj_constr_u_active = False
            taylor_active_u = self.taylor_u<=self.dom_u
            if np.any(taylor_active_u):
                self.conj_taylor_u = np.full(self.domain.shape,np.inf)
                self.conj_taylor_u[taylor_active_u] = self._f_deriv(self.taylor_u[taylor_active_u],mask=taylor_active_u,**kwargs)*self.measure[taylor_active_u]
                self._if_constant_broadcast(self.conj_taylor_u)

                self.f_u = self.domain.zeros()
                self.f_u[taylor_active_u] = self._f(self.taylor_u[taylor_active_u],mask=taylor_active_u,**kwargs)
                self._if_constant_broadcast(self.f_u,mask=taylor_active_u)

                self.conj_f_u = domain.zeros()
                self.conj_f_u[taylor_active_u] = self._f_conj(self.conj_taylor_u[taylor_active_u]/self.measure[taylor_active_u],mask=taylor_active_u,**kwargs)
                self._if_constant_broadcast(self.conj_f_u,mask=taylor_active_u)

                self.fpp_u = domain.zeros()
                self.fpp_u[taylor_active_u] = self._f_second_deriv(self.taylor_u[taylor_active_u],mask=taylor_active_u,**kwargs)
                self._if_constant_broadcast(self.fpp_u,mask=taylor_active_u)

                self.fstarpp_u = domain.zeros()
                self.fstarpp_u[taylor_active_u] = self._f_conj_second_deriv(self.conj_taylor_u[taylor_active_u]/self.measure[taylor_active_u],mask=taylor_active_u,**kwargs)
                self._if_constant_broadcast(self.fstarpp_u,mask=taylor_active_u)

                self.conj_dom_u = np.array(self.conj_dom_u)
                self.conj_dom_u[taylor_active_u] =  np.inf
                self._if_constant_broadcast(self.conj_dom_u)

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

    def _if_constant_broadcast(self,x,mask=None):
        if not isinstance(x,np.ndarray) and x.shape != self.domain.shape:
            raise TypeError(Errors.type_error(f"If constant use broadcasting in the IntegralFunctional init only allowed for numpy arrays of same shape as domain." +"\n\t" + f"x = {x}"))
        if mask is None:
            if np.allclose(x, x.flatten()[0],rtol=1e-10,atol=1e-12):
                x = np.broadcast_to(x.flatten()[0],self.domain.shape)
        else:
            if np.allclose(x[mask], x[mask].flatten()[0],rtol=1e-10,atol=1e-12):
                x = np.broadcast_to(x[mask].flatten()[0],self.domain.shape)

    def _assert_essential_domain(self,v,eps=1e-10,msg=None):
        self._buf = v-self.dom_u
        if np.any(self._buf>eps): raise  RuntimeError(Errors.generic_message(msg+f" argument too large in {self}."+"\n\t" + f"diff:{np.max(self._buf)},"+"\n\t"+ f"eps={eps}"))
        self._buf = self.dom_l-v
        if np.any(self._buf>eps): raise  RuntimeError(Errors.generic_message(msg+f" argument too small in {self}."+"\n\t"+f"diff:{np.max(self._buf)},"+"\n\t"+f"eps={eps}"))

    def _assert_conj_essential_domain(self,vstar,eps=1e-10,msg=None):
        self._buf = vstar-self.conj_dom_u
        if np.any(self._buf>eps): raise  RuntimeError(Errors.generic_message(msg+f" argument too large in {self}."+"\n\t"+f"diff:{np.max(self._buf)},"+"\n\t"+f"eps={eps}"))
        self._buf = self.conj_dom_l-vstar
        if np.any(self._buf>eps): raise  RuntimeError(Errors.generic_message(msg+f" argument too small in {self}."+"\n\t"+f"diff:{np.max(self._buf)},"+"\n\t"+f"eps={eps}"))

    def _eval(self, v, func_vals =None):
        # see comment in _conj! 
        if self.conj_constr_l_active or self.quad_taylor_l_active:
            v_small = (v<self.taylor_l)
            if np.any(v_small):    
                mask = ~v_small
                self._buf[v_small] = v[v_small]
                self._buf[v_small] *= self.conj_taylor_l[v_small] if self.quad_taylor_l_active else self.conj_dom_l[v_small]
                self._buf[v_small] /= self.measure[v_small] 
                self._buf[v_small] -= self.conj_f_l[v_small]
                if self.quad_taylor_l_active:
                    aux = v[v_small]-self.taylor_l[v_small]
                    aux *= aux
                    aux *= 0.5
                    aux *= self.fpp_l[v_small]
                    self._buf[v_small] += aux
        if self.conj_constr_u_active or self.quad_taylor_u_active:
            v_large = (v>self.taylor_u)
            if np.any(v_large):    
                mask = np.logical_and(mask,~v_large) if 'mask' in locals() else ~v_large
                self._buf[v_large] = v[v_large]
                self._buf[v_large] *= self.conj_taylor_u[v_large] if self.quad_taylor_u_active else self.conj_dom_u[v_large]
                self._buf[v_large] /= self.measure[v_large]
                self._buf[v_large] -= self.conj_f_u[v_large]
                if self.quad_taylor_u_active:
                    aux = v[v_large]-self.taylor_u[v_large]
                    aux *= aux
                    aux *= 0.5
                    aux *= self.fpp_u[v_large]
                    self._buf[v_large] += aux
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
        if self.constr_l_active or self.quad_taylor_l_active:
            vstar_small = (vstar<self.conj_taylor_l)
            if np.any(vstar_small):
                mask = ~vstar_small            
                self._buf[vstar_small] = self._buf2[vstar_small]
                self._buf[vstar_small] *= self.taylor_l[vstar_small] if self.quad_taylor_l_active else self.dom_l[vstar_small]
                self._buf[vstar_small] -= self.f_l[vstar_small]
                if self.quad_taylor_l_active:
                    aux = self._buf2[vstar_small]-self.conj_taylor_l[vstar_small]/self.measure[vstar_small]
                    aux *= aux
                    aux *= 0.5
                    aux *= self.fstarpp_l[vstar_small]
                    self._buf[vstar_small] += aux
        if self.constr_u_active or self.quad_taylor_u_active:
            vstar_large = (vstar>self.conj_taylor_u)
            if np.any(vstar_large):   
                mask = np.logical_and(mask,~vstar_large) if 'mask' in locals() else ~vstar_large
                self._buf[vstar_large] = self._buf2[vstar_large]
                self._buf[vstar_large] *= self.taylor_u[vstar_large] if self.quad_taylor_u_active else self.dom_u[vstar_large]
                self._buf[vstar_large] -= self.f_u[vstar_large]
                if self.quad_taylor_u_active:
                    aux = self._buf2[vstar_large]-self.conj_taylor_u[vstar_large]/self.measure[vstar_large]
                    aux *= aux
                    aux *= 0.5
                    aux *= self.fstarpp_u[vstar_large]
                    self._buf[vstar_large] += aux                
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
        if self.conj_constr_l_active or self.quad_taylor_l_active:
            v_small = (v<self.taylor_l)
            if np.any(v_small):
                mask = np.logical_not(v_small)   
                self._buf[v_small] = self.conj_taylor_l[v_small] if self.quad_taylor_l_active else self.conj_dom_l[v_small]
                if self.quad_taylor_l_active:
                    aux = v[v_small]-self.taylor_l[v_small]
                    aux *= self.fpp_l[v_small]
                    aux *= self.measure[v_small]
                    self._buf[v_small] += aux  
        if self.conj_constr_u_active or self.quad_taylor_u_active:
            v_large = (v>self.taylor_u)
            if np.any(v_large):
                mask = np.logical_and(mask,~v_large) if 'mask' in locals() else ~v_large
                self._buf[v_large] = self.conj_taylor_u[v_large] if self.quad_taylor_u_active else self.conj_dom_u[v_large]
                if self.quad_taylor_u_active:
                    aux = v[v_large]-self.taylor_u[v_large]
                    aux *= self.fpp_u[v_large]
                    aux *= self.measure[v_large]
                    self._buf[v_large] += aux                
        if 'mask' in locals():
            self._buf[mask] = self._f_deriv(v[mask],mask=mask,**self.kwargs)*self.measure[mask]
        else:
            self._buf = self._f_deriv(v,**self.kwargs)*self.measure
        return self._buf.copy()

    def dist_subdiff(self, vstar, x):
        self._assert_essential_domain(x,msg='dist_subdiff')
        diff = self._ptw_dist_subdiff(vstar,x)
        ind = np.where(np.isclose(x,self.dom_u))
        if np.any(ind):
            diff[ind]=np.minimum(vstar[ind]-self._f_deriv(x[ind],mask =ind)*self.measure[ind],0.)
        ind = np.where(np.isclose(x,self.dom_l))
        if np.any(ind):
            diff[ind]=np.maximum(vstar[ind]-self._f_deriv(x[ind],mask =ind)*self.measure[ind],0.)
        return self.h_domain.dual_space().norm(diff)

    def _conj_subgradient(self, vstar):
        if not self.conj_everywhere_finite:
            self._assert_conj_essential_domain(vstar,msg='_conj_subgradient')
        self._buf2 = vstar/self.measure
        if self.constr_l_active or self.quad_taylor_l_active:
            vstar_small = (vstar<self.conj_taylor_l)
            if np.any(vstar_small):
                mask = np.logical_and(mask,~vstar_small) if 'mask' in locals() else ~vstar_small  
                self._buf[vstar_small] = self.taylor_l[vstar_small] if self.quad_taylor_l_active else self.dom_l[vstar_small]
                if self.quad_taylor_l_active:
                    aux = self._buf2[vstar_small]-self.conj_taylor_l[vstar_small]/self.measure[vstar_small]
                    aux *= self.fstarpp_l[vstar_small]
                    self._buf[vstar_small] += aux
        if self.constr_u_active or self.quad_taylor_u_active:
            vstar_large = (vstar>self.conj_taylor_u)
            if np.any(vstar_large):
                mask = np.logical_and(mask,~vstar_large) if 'mask' in locals() else ~vstar_large
                self._buf[vstar_large] = self.taylor_u[vstar_large] if self.quad_taylor_u_active else self.dom_u[vstar_large]
                if self.quad_taylor_u_active:
                    aux = self._buf2[vstar_large]-self.conj_taylor_u[vstar_large]/self.measure[vstar_large]
                    aux *= self.fstarpp_u[vstar_large]
                    self._buf[vstar_large] += aux
        if 'mask' in locals():
            self._buf[mask] = self._f_conj_deriv(self._buf2[mask],mask=mask,**self.kwargs)
        else:
            self._buf = self._f_conj_deriv(self._buf2,**self.kwargs)
        return self._buf.copy()

    def _conj_dist_subdiff(self, v, xstar):
        self._assert_conj_essential_domain(xstar, msg='_conj_dist_subdiff')
        diff = self._conj_ptw_dist_subdiff(v,xstar)
        ind = np.where(np.isclose(xstar, self.conj_dom_u))
        if np.any(ind):
            diff[ind]=np.minimum(v[ind]-self._f_conj_deriv(xstar[ind]/self.measure[ind],mask =ind),0.)
        ind = np.where(np.isclose(xstar,self.conj_dom_l))
        if np.any(ind):
            diff[ind]=np.maximum(v[ind]-self._f_conj_deriv(xstar[ind]/self.measure[ind],mask =ind),0.)         
        return self.h_domain.norm(diff)

    def _hessian(self, v):
        if not self.everywhere_finite:
            self._assert_essential_domain(v,msg='_hessian')
        self._buf = np.full(self.domain.shape,np.inf)

        if self.conj_constr_l_active:
            self._buf[v<self.taylor_l] = 0.
        if self.quad_taylor_l_active:
            v_small = v<self.taylor_l
            self._buf[v_small] = self.fpp_l[v_small]
        if self.conj_constr_u_active:
            self._buf[v>self.taylor_u] = 0.
        if self.quad_taylor_u_active:
            v_large = v>self.taylor_u
            self._buf[v_large] = self.fpp_u[v_large]

        if self.conj_constr_l_active or self.conj_constr_u_active or self.quad_taylor_l_active or self.quad_taylor_u_active:
            mask = (self._buf==np.inf)
            self._buf[mask] =  self._f_second_deriv(v[mask],mask=mask,**self.kwargs)
        else:
            self._buf =  self._f_second_deriv(v,**self.kwargs)
        self._buf *= self.measure
        return rop.PtwMultiplication(self.domain,self._buf.copy())

    def _conj_hessian(self, vstar):
        if not self.conj_everywhere_finite:
            self._assert_conj_essential_domain(vstar,msg='_conj_hessian')
        self._buf2 = vstar/self.measure
        self._buf = np.full(self.domain.shape,np.inf)

        if self.constr_l_active:
            self._buf[self._buf2<self.conj_taylor_l] = 0.
        if self.constr_u_active:
            self._buf[self._buf2>self.conj_taylor_u] = 0.
        if self.quad_taylor_l_active:
            vstar_small = vstar<self.conj_taylor_l
            self._buf[vstar_small] = self.fstarpp_l[vstar_small]
        if self.quad_taylor_u_active:
            vstar_large = vstar>self.conj_taylor_u
            self._buf[vstar_large] = self.fstarpp_u[vstar_large]

        if self.constr_l_active or self.constr_u_active or self.quad_taylor_l_active or self.quad_taylor_u_active:   
            mask = (self._buf==np.inf)
            self._buf[mask] = self._f_conj_second_deriv(self._buf2[mask],mask=mask,**self.kwargs)
        else:
            self._buf = self._f_conj_second_deriv(self._buf2,**self.kwargs)
        self._buf /= self.measure
        return rop.PtwMultiplication(self.domain, self._buf.copy())
    
    def _proximal(self, v, tau,mask=None):
        if mask is None:
            res = self._f_prox(v,tau,**self.kwargs)
        else:
            res = self._f_prox(v,tau,mask=mask,**self.kwargs)
        mask = mask if mask is not None else slice(None)            

        if self.constr_l_active:
            res = np.maximum(res,self.dom_l[mask])
        if self.constr_u_active:
            res = np.minimum(res,self.dom_u[mask])
        
        if self.conj_constr_l_active:
            linprox = v- tau*self.conj_dom_l[mask]/self.measure[mask]                
            res[linprox<=self.taylor_l] = linprox[linprox<=self.taylor_l]
        if self.conj_constr_u_active:
            linprox = v-tau*self.conj_dom_u[mask]/self.measure[mask]
            res[linprox>=self.taylor_u] = linprox[linprox>=self.taylor_u]
        if self.quad_taylor_l_active:
            taufpp=tau*self.fpp_l[mask]
            taufp=tau*self.conj_taylor_l[mask]/self.measure[mask]
            quadprox=(v+taufpp*self.taylor_l-taufp)/(1.+taufpp)
            res[quadprox<=self.taylor_l] = quadprox[quadprox<=self.taylor_l]
        if self.quad_taylor_u_active:
            taufpp=tau*self.fpp_u[mask]
            taufp=tau*self.conj_taylor_u[mask]/self.measure[mask]
            quadprox = (v+taufpp*self.taylor_u-taufp)/(1.+taufpp)
            res[quadprox>=self.taylor_u] = quadprox[quadprox>=self.taylor_u]
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
        mask = mask if mask is not None else slice(None)

        if self.conj_constr_l_active:
            res = np.maximum(res,self.conj_dom_l[mask])
        if self.conj_constr_u_active:
            res = np.minimum(res,self.conj_dom_u[mask])

        if self.constr_l_active:
            linprox = vstar -  tau*self.dom_l[mask]*self.measure[mask]
            res[linprox<=self.conj_taylor_l] = linprox[linprox<=self.conj_taylor_l]
        if self.constr_u_active:
            linprox = vstar -  tau*self.dom_u[mask]*self.measure[mask]
            res[linprox>=self.conj_taylor_u] = linprox[linprox>=self.conj_taylor_u]           

        if self.quad_taylor_l_active:
            taufstarpp=tau*self.fstarpp_l[mask]
            taufstarp=tau*self.taylor_l[mask]*self.measure[mask]
            quadprox=(vstar + taufstarpp*self.conj_taylor_l[mask] - taufstarp) / (1.+taufstarpp)
            res[quadprox<=self.conj_taylor_l] = quadprox[quadprox<=self.conj_taylor_l] 
        if self.quad_taylor_u_active:
            taufstarpp=tau*self.fstarpp_u[mask]
            taufstarp=tau*self.taylor_u[mask]*self.measure[mask]
            quadprox=(vstar + taufstarpp*self.conj_taylor_u[mask] - taufstarp) / (1.+taufstarpp) 
            res[quadprox>=self.conj_taylor_u] = quadprox[quadprox>=self.conj_taylor_u]

        return res

    def _f(self,v,**kwargs):
        raise NotImplementedError
    
    def _f_deriv(self,v,**kwargs):
        raise NotImplementedError

    def _ptw_dist_subdiff(self,vstar,w,**kwargs):
        return vstar-self.subgradient(w,**kwargs)

    def _f_second_deriv(self,v,**kwargs):
        raise NotImplementedError

    def _f_prox(self,v,tau,tol=1e-12, abstol=1e-12,maxNewtonIter=15,maxBisecIter=300,maxBoundsIter=100,**kwargs):
        if self.__class__.__dict__.get("_f_deriv") is not None:
            vclip = np.minimum(v,self.dom_u)
            vclip = np.maximum(vclip,self.dom_l)
            f_second_deriv = self._f_second_deriv if self._f_second_deriv is not None else None
            return self._numerical_prox(v,tau,self._f_deriv,f_second_deriv,self.dom_l, self.dom_u, 
                                        tol=tol,abstol=abstol,
                                        maxNewtonIter=maxNewtonIter,maxBisecIter=maxBisecIter,maxBoundsIter=maxBoundsIter,
                                        **kwargs
                                        )
        else:
            NotImplementedError('Need first derivative for numerical prox operator')
    
    def _f_conj(self,vstar,**kwargs):
        raise NotImplementedError
    
    def _f_conj_deriv(self,vstar,**kwargs):
        raise NotImplementedError

    def _conj_ptw_dist_subdiff(self,v,wstar,**kwargs):
        return v-self.conj.subgradient(wstar,**kwargs)

    def _f_conj_second_deriv(self,vstar,**kwargs):
        raise NotImplementedError

    def _f_conj_prox(self,vstar,tau,tol=1e-12, abstol=1e-12, 
                     maxNewtonIter=15,maxBisecIter=300,maxBoundsIter=100,
                     **kwargs
                     ):
        if self.__class__.__dict__.get("_f_conj_deriv") is not None:
            vclip = np.minimum(vstar,self.conj_dom_u)
            vclip = np.maximum(vclip,self.conj_dom_l)
            vclip /= self.measure
            f_conj_second_deriv = self._f_conj_second_deriv if self._f_conj_second_deriv is not None else None
            return self._numerical_prox(vstar,tau,
                                        self._f_conj_deriv,f_conj_second_deriv,
                                        self.conj_dom_l/self.measure, 
                                        self.conj_dom_u/self.measure, 
                                        tol=tol, abstol=abstol,
                                        maxNewtonIter=maxNewtonIter,maxBisecIter=maxBisecIter,maxBoundsIter=maxBoundsIter,
                                        **kwargs
                                        )
        else:
            NotImplementedError('Need first derivative of conjugate functional for numerical conjugate prox operator')

    def _numerical_prox(self, y, tau, fp, fpp, 
                        dom_l, dom_u, 
                        start=None,ub=None, lb=None,
                        tol=1e-15, abstol=1-15, 
                        maxNewtonIter=15,maxBisecIter=300,maxBoundsIter=100,**kwargs):
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
        if not isinstance(y,np.ndarray):
            raise TypeError(Errors.not_instance(y,np.ndarray,add_info="The numerical prox in IntegralFunctionals can only be computed for y an ndarray"))
        if np.isscalar(dom_u):
            dom_u = np.broadcast_to(dom_u,y.shape)
        if np.isscalar(dom_l):
            dom_l = np.broadcast_to(dom_l,y.shape)    
        if not np.all(dom_l<=dom_u):
            raise ValueError(Errors.value_error('Upper/lower bounds of essential domain invalid. dom_l not less or equal to dom_u!',self,"_numerical_prox"))
  

        #initial guess
        x = start if not (start is None) else y
        x = np.maximum(dom_l,x)
        x = np.minimum(x,dom_u)


        r = fp(x,**kwargs) + (x - y)/tau
        converged = ((dom_u-dom_l<abstol) | (dom_u-dom_l < tol * np.maximum(dom_u,-dom_l)))   # width of essential domain <tol
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
           xl = x[x_too_large]
           rl = r[x_too_large]
           dl = dom_l[x_too_large]
           up = np.minimum(tau*np.ones_like(xl),tau*rl)
           up2 = np.maximum(xl-up,dl)
           lb[x_too_large] = up2
#           lb[x_too_large] = np.maximum(x[x_too_large]
#                                     - np.minimum(tau*np.ones_like(r[x_too_large]), tau*r[x_too_large]),
#                                        dom_l[x_too_large])

        self.log.debug(f'before correction: r: {r}\n ub: {ub}\n lb: {lb}\n diff: {ub-lb}')
        if not np.all(lb<=ub):
            ind = lb>ub
            raise RuntimeError(Errors.runtime_error(f'lb<=ub violated for input values {y[lb>ub]}. lb: {lb[lb>ub]}, ub: {ub[lb>ub]} ',self,"_numerical_prox"))

        # decrease lb where necessary to make it a valid lower bound
        for it in range(maxBoundsIter):
            v = fp(lb,**kwargs) + (lb- y)/tau            
            ind = (v>0) & (~converged) & (lb>dom_l)
            if it==0 and not np.all(ub[ind]-lb[ind]>tol*np.maximum(ub[ind],-lb[ind])):
                ii = ind & ((ub-lb<=tol * np.maximum(ub,-lb)))
                lb[ii] = np.maximum(lb[ii]-tol*np.abs(lb[ii])-abstol,dom_l[ii])
                self.log.debug(f'adding {np.sum(ii & (lb==dom_l))} indices as converged.')
                converged = converged | (ii & (lb==dom_l))
            self.log.debug(f'lower bound it {it}: {np.sum(ind)} indices invalid.')
            if np.sum(ind)==0:
                break
            else:
                lb[ind] = np.maximum(dom_l[ind],2*lb[ind]-ub[ind])
        if not np.sum(ind)==0:
            raise RuntimeError(Errors.runtime_error(f"Could not determine lower bound. Increase maxBoundsIter! lb: {lb[ind]}, ub: {ub[ind]}, dom_l: {dom_l[ind]}",self,"_numerical_prox")) 
        r = fp(lb,**kwargs) + (lb - y)/tau 
        converged = converged | ((lb==dom_l) & (r>=0))
        if not np.all((r<=0) | converged):
            raise RuntimeError(Errors.runtime_error(f'lower bound not satisfied for indices {np.where((r>0) & ~converged)}',self,"_numerical_prox"))

        # increase ub where necessary to make it a valid upper bound
        for it in range(maxBoundsIter):
            v = fp(ub,**kwargs) + (ub- y)/tau            
            ind = (v<0) & (~converged)
            self.log.debug(f'upper bound it {it}: {np.sum(ind)} indices invalid')
            if it==0 and not np.all(ub[ind]-lb[ind]>tol*np.maximum(ub[ind],-lb[ind])):
                ii = ind & (ub-lb<=tol * np.maximum(ub,-lb))
                ub[ii] = np.minimum(dom_u[ii],ub[ii]+tol*np.abs(ub[ii])+abstol)
                self.log.debug(f'adding {np.sum(ii & (ub==dom_u))} indices as converged.')
                converged = converged | (ii & (ub==dom_u))
            if np.sum(ind)==0:
                break
            else:
                ub[ind] = np.minimum(dom_u[ind], 2*ub[ind]-lb[ind])
        if not np.sum(ind)==0:
            raise RuntimeError(Errors.runtime_error("Could not determine upper bound. Increase maxBoundsIter!",self,"_numerical_prox"))
        r = fp(ub,**kwargs) + (ub - y)/tau 
        converged = converged | ((ub==dom_u) & (r<=0))
        if not np.all((r>=0) | converged): 
            raise RuntimeError(Errors.runtime_error(f'upper bound not satisfied for indices {np.where((r<0) & ~converged)}',self,"_numerical_prox"))
        if not np.all(lb>=dom_l):
            raise RuntimeError(Errors.runtime_error(f"Lower bound under the domains lower bound of the essential domain. "+"\n\t "+f"lb = {lb},"+"\n\t "+f"dom_l = {dom_l}", self,"_numerical_prox"))
        if not np.all(ub<=dom_u):
            raise RuntimeError(Errors.runtime_error(f"Uper bound above the domains uper bound of the essential domain. "+"\n\t "+f"ub = {ub},"+"\n\t "+f"dom_u = {dom_u}", self,"_numerical_prox"))

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
                conv = ((np.abs(ub-lb) < abstol) | (ub-lb <=tol*np.maximum(ub,-lb)))
                converged = converged | conv
                x = np.where(conv, x, xnew)
                self.log.debug(f'Newton it {iter}: {np.sum(converged==True)} out of {len(x.flatten())} converged.')
    
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
                nr_not_converged = np.sum(((ub-lb) >= abstol) & ((ub-lb) >= tol*np.maximum(ub,-lb)))
                self.log.debug(f'bisection it. {iter}: {nr_not_converged} not converged')
                if nr_not_converged == 0:
                    break
            x[mask] = 0.5*(lb+ub)
            if np.any((ub-lb>=abstol)  &  ((ub-lb) >= tol*np.maximum(ub,-lb))):
                ind = (ub-lb>= abstol) &  ((ub-lb) >= tol*np.maximum(ub,-lb))
                ff =fp(x[mask],mask=mask,**kwargs) + (x[mask]-y[mask])/tau
                raise RuntimeError(f'Could not satisfy either of the tolerance criteria (tol = {tol}, abstol = {abstol}) after {iter} steps of bisection algorithm at indices {np.where(ind)}. x values: {x[mask][ind]}, f values: {ff[ind]}, ub-lb {ub[ind]-lb[ind]}.')

        return x
    
class VectorIntegralFunctional(Functional):
    r"""
    Implements a vector-valued integral functional :math:`v\mapsto \sum_i f_i(\|v(x)\|)dx` on some domain in `regpy.vecsps.MeasureSpaceFcts`
    as a functional. Here, :math:`f_i` are scalar functions on the real line that are assumed to be even and convex, 
    and :math:`\|\cdot\|` some norm on the vector values. 

    The norm on the vector values can be given as a method taking the vector-valued function and the vector axis 
    as tuple and returning the norm values. By default, the Euclidean norm is used.  

    Parameters
    ----------
    vdomain : `regpy.vecsps.MeasureSpaceFcts`
        Domain of vector-valued functions on which it is defined. Needs some Measure therefore a `regpy.vecsps.MeasureSpaceFcts`
    scalar_func : `AbstractFunctional` or `IntegralFunctionalBase`
        Scalar separable functional to be used as (even!) functions :math:`f_i`.
        Either provides a specific separable Functional  
        or a AbstractFunctional which results in a `regpy.functionals.IntegralFunctionalBase`. Defaults to 'Lpp' with `p=2`. 
    scalar_func_args : dict, optional
        Additional arguments for the scalar_func if needed (e.g. 'p' for `Lpp`).
    vector_norm_p : int or float, optional
        This is the exponent in the norm of the vectors given by a :math:`p` norm :math:`(|x_1|^p+\dots+|x_n|^p)^{1/p}}`. 
        By default, the Euclidean norm is used with :math:`p=2`.
    Lipschitz, convexity_param: float (default: np.inf and 0., respectively)
        For the Euclidean norm as inner norm, :math:`Hess f_i(\|v\| )` has two eigenvalues: :math:`f_i''(\|v\|) `
        and :math:`f_i'(\|v\|)/\|v\|`. Using this fact, analytic expression may be computed for 
        Lipschitz = supremum of all possible EV and for convexity_param = infimum of all EV may be 
        computed for specific :math:`f_i` and passed as arguments to accelerate optimization methods.  
    """    

    def __init__(self, vdomain, scalar_func = None, scalar_func_args=None, 
                 vector_norm_p=2,
                 Lipschitz =np.inf, convexity_param = 0.,
                 methods = None, conj_methods = None
                 ):
        if not isinstance(vdomain,MeasureSpaceFcts) and vdomain.ndim_codomain>=1:
            raise ValueError(Errors.not_instance(vdomain,MeasureSpaceFcts,f"The vector domain needs to be an instance of MeasureSpaceFcts with ndim_codomain>1 since vector-valued functions need a measure and a vector domain."))
        self.vdomain = vdomain
        self.sdomain = vdomain.scalar_space()
        self.scalar_func_args = scalar_func_args if scalar_func_args is not None else {}

        if isinstance(scalar_func, AbstractFunctional):
            scalar_func = scalar_func(self.sdomain, **self.scalar_func_args)
        elif scalar_func is None:
            scalar_func = LppPower(self.sdomain, p=2., **self.scalar_func_args)
        if not scalar_func.separable or scalar_func.domain!=self.sdomain:
            raise ValueError(Errors.value_error(f'{scalar_func} need to be a callable giving a separable functional or already a separable functional with sdomain as domain.'))
        else:
            self.scalar_func = scalar_func

        if isinstance(vector_norm_p,(int,float)) and vector_norm_p >1 and vector_norm_p<np.inf:
            self.p = vector_norm_p
            self.q = vector_norm_p / (vector_norm_p - 1) 
            self.vector_norm = partial(np.linalg.norm, ord = self.p)
            self.dual_vector_norm = partial(np.linalg.norm, ord = self.q)
        else:
            raise ValueError(Errors.value_error("Not p-Norm", f"The provided p ={vector_norm_p} is not between 1< p < inf."))

        methods = methods if methods is not None else self.scalar_func.methods
        conj_methods = conj_methods if conj_methods is not None else self.scalar_func.conj.methods
        if self.p!=2:
            methods -= {'subgradient','hessian','proximal'}
            conj_methods =  {'subgradient','hessian','proximal'}

        super().__init__(vdomain, L2(vecsp=vdomain), 
                         convexity_param = convexity_param, 
                         Lipschitz = Lipschitz, 
                         separable= False, linear= False,
                         methods = methods, conj_methods = conj_methods)
        self._sbuf = self.sdomain.zeros()
        self._vbuf = vdomain.zeros()
        self._vaxes = tuple(range(-self.vdomain.ndim+self.vdomain.ndim_domain,0))

    @property
    def _sbuf_ext(self):
        return np.expand_dims(self._sbuf, axis=self._vaxes)

    def _eval(self, vec):
        self._sbuf = self.vector_norm(x = vec, axis=self._vaxes)
        return self.scalar_func(self._sbuf)

    def _conj(self, vec_star):
        self._sbuf = self.dual_vector_norm(x = vec_star, axis=self._vaxes)
        return self.scalar_func.conj(self._sbuf)

    def _subgradient(self, vec):
        r"""
        returns   
        
        .. math::
            [f_i'(\|v_i\|)\frac{v_i}{\|v_i\|}]_i
        """
        if self.p != 2:
            raise NotImplementedError('subgradient only implemented for L2 norm')
        self._sbuf = self.vector_norm(vec, axis=self._vaxes)
        np.divide(vec,self._sbuf_ext,where=self._sbuf_ext>0,out=self._vbuf)
        self._sbuf = self.scalar_func.subgradient(self._sbuf)
        self._vbuf *= self._sbuf_ext
        return self._vbuf.copy()
    
    def _conj_subgradient(self, vec_star):
        if self.p != 2:
            raise NotImplementedError('conjugate subgradient only implemented for L2 norm')
        self._sbuf = self.dual_vector_norm(vec_star, axis=self._vaxes)
        np.divide(vec_star,self._sbuf_ext,where=self._sbuf_ext>0,out=self._vbuf)
        self._sbuf = self.scalar_func.conj.subgradient(self._sbuf)
        self._vbuf *= self._sbuf_ext
        return self._vbuf.copy()

    def _proximal(self, vec, tau, mask=None):
        r"""
        returns   :math:`[prox_{\tau f_i}(\|v_i\|)\frac{v_i}{\|v_i\|}]_i`
        """
        if self.p != 2:
            raise NotImplementedError('proximal only implemented for L2 norm')
        self._sbuf = self.vector_norm(vec, axis=self._vaxes)
        np.divide(vec,self._sbuf_ext,where=self._sbuf_ext>0,out=self._vbuf)
        self._sbuf = self.scalar_func.proximal(self._sbuf, tau ,mask=None)
        self._vbuf *= self._sbuf_ext
        return self._vbuf.copy()

    def _conj_proximal(self, vec_star, tau, mask=None):
        if self.p != 2:
            raise NotImplementedError('conjugate proximal only implemented for L2 norm')
        self._sbuf = self.dual_vector_norm(vec_star, axis=self._vaxes)
        np.divide(vec_star,self._sbuf_ext,where=self._sbuf_ext>0,out=self._vbuf)
        self._sbuf = self.scalar_func.conj.proximal(self._sbuf, tau, mask=None)
        self._vbuf *= self._sbuf_ext
        return self._vbuf.copy()
    
    def _hessian(self, vec):
        if self.p != 2:
            raise NotImplementedError('proximal only implemented for L2 norm')
        self._sbuf = self.vector_norm(vec, axis=self._vaxes)
        np.divide(vec,self._sbuf_ext,where=self._sbuf_ext>0,out=self._vbuf)
        fp = self.scalar_func.subgradient(self._sbuf)
        fp = np.expand_dims(fp,self._vaxes)
        np.divide(fp,self._sbuf_ext, where=self._sbuf_ext>0, out= fp)
        fpp = self.scalar_func.hessian(self._sbuf)(self.sdomain.ones())
        # As f is even, f' is odd, so f'(0)=0 if f'' exists. Therefore, f'(v)/v tends to f''(v) as 
        # v tends to 0, so we can use this a places where ||v|| vanishes.
        fpp = np.expand_dims(fpp,self._vaxes)
        fp[~(self._sbuf_ext>0)] = fpp[~(self._sbuf_ext>0)]
        scal_mult =rop.PtwScalarMultiplication(self.domain,fp)
        proj = rop.PtwMatrixVectorMultiplication(self.vdomain,
                                             np.expand_dims(self._vbuf,axis=-2).copy()
                                             )
        # actually just a pointwise multiplication, but implemented multiplication with a 1x1 matrix 
        # to avoid conversions to scalar functions
        ptw_mult = rop.PtwMatrixVectorMultiplication(self.sdomain.vector_valued_space(1),
                                                 np.expand_dims(fpp-fp,axis=-2)
                                                )

        return scal_mult + proj.adjoint * ptw_mult * proj                                             
        
    def _conj_hessian(self, vec):
        if self.p != 2:
            raise NotImplementedError('proximal only implemented for L2 norm')
        self._sbuf = self.dual_vector_norm(vec, axis=self._vaxes)
        np.divide(vec,self._sbuf_ext,where=self._sbuf_ext>0,out=self._vbuf)
        fp = self.scalar_func.conj.subgradient(self._sbuf)
        fp = np.expand_dims(fp,self._vaxes)
        np.divide(fp,self._sbuf_ext, where=self._sbuf_ext>0, out= fp)
        fpp = self.scalar_func.conj.hessian(self._sbuf)(self.sdomain.ones())
        # see comments for hessian!
        fpp = np.expand_dims(fpp,self._vaxes)
        fp[~(self._sbuf_ext>0)] = fpp[~(self._sbuf_ext>0)]
        scal_mult = rop.PtwScalarMultiplication(self.domain,fp)
        proj = rop.PtwMatrixVectorMultiplication(self.vdomain,
                                             np.expand_dims(self._vbuf,axis=-2).copy()
                                             )
        ptw_mult = rop.PtwMatrixVectorMultiplication(self.sdomain.vector_valued_space(1),
                                                 np.expand_dims(fpp-fp,axis=-2)
                                                )

        return scal_mult + proj.adjoint * ptw_mult * proj   

class LppL2(VectorIntegralFunctional):
    """
    VectorIntegralFunctional with :math:`f_i(x):=(1/p)|x|^p`.

    Parameters
    ----------
    vdomain : `regpy.vecsps.MeasureSpaceFcts`
        A vector valued measure space.
    p: float (default: 2.)
        exponent of the :math:`L^p` norm. 
    """
    def __init__(self, vdomain,p=2.):
        sfunc = LppPower(vdomain.scalar_space(),p=p)
        super().__init__(vdomain, scalar_func=sfunc, 
                         Lipschitz=1. if p== 2 else np.inf,
                         convexity_param=1. if p==2 else 0.
                         )

class L1L2(VectorIntegralFunctional):
    """
    VectorIntegralFunctional with absolute value function as :math:`f_i`.

    Parameters
    ----------
    vdomain : `regpy.vecsps.MeasureSpaceFcts`
        A vector valued measure space.
    conj_tol : float
        A tolarance introduced into `L1` functional that allows also numbers slighly bigger then 1.0 in the conjugate.
    """
    def __init__(self, vdomain, conj_tol = 1e-16):
        sfunc = L1MeasureSpace(vdomain.scalar_space(), conj_tol= conj_tol)
        super().__init__(vdomain, scalar_func= sfunc, Lipschitz=1.)

class HuberL2(VectorIntegralFunctional):
    """
    VectorIntegralFunctional with Huber functional as :math:`f_i`.

    Parameters
    ----------
    vdomain : `regpy.vecsps.MeasureSpaceFcts`
        A vector valued measure space.
    sigma: float, default: 1.
        Parameter in Huber functional
    """
    def __init__(self, vdomain,sigma=1.):
        sfunc = Huber(vdomain.scalar_space(),sigma)
        super().__init__(vdomain, scalar_func=sfunc, Lipschitz=1.)

class LppPower(IntegralFunctionalBase):
    r"""
    Implements the norm power functional :math:`v\mapsto \frac{1}{p} \|v\|_{L^p}^p` power on some domain in `regpy.vecsps.MeasureSpaceFcts`
    as an integral functional. This corresponds to the function :math:`f(v):=\frac{1}{p}|v|^p`.

    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a `regpy.vecsps.MeasureSpaceFcts`
    p: float >1 [option]
        exponent
    constr_l,constr_u,lin_taylor_l, lin_taylor_u: None, np.isscalar or np.ndarray
        see IntegralFunctional  
    """

    def __init__(self, domain, p=2.,
                 constr_l=None, constr_u=None, lin_taylor_l=None, lin_taylor_u=None,
                  quad_taylor_l=None, quad_taylor_u=None,
                 **kwargs):
        if not np.isscalar(p): raise TypeError(Errors.not_instance(p,float,f"p in Lpp functional can only be a scalar. not {p} of type {type(p)}."))
        if p<= 1: raise ValueError(Errors.value_error(f"p in Lpp functional has to be bigger than one. was given p = {p}", self))
        self.p = p
        self.q = p/(p-1)

        dom_l = -np.inf if constr_l is None else constr_l        
        dom_u = np.inf if constr_u is None else constr_u
        dom2_l = dom_l
        dom2_u = dom_u
        taylor_l = -np.inf if lin_taylor_l is None else lin_taylor_l
        if quad_taylor_l is not None:
            taylor_l = quad_taylor_l
            dom2_l= taylor_l
        taylor_u = np.inf if lin_taylor_u is None else lin_taylor_u
        if quad_taylor_u is not None:
            taylor_u = quad_taylor_u
            dom2_u= taylor_u

        if p<2:
            aux = np.abs(np.max(np.maximum(-dom2_l,dom2_u)))
            convexity_param = (self.p-1) * aux**(self.p-2) if aux<np.inf else 0

            aux = np.min(np.maximum(taylor_l,-taylor_u))
            Lipschitz = (self.p-1) * np.abs(aux)**(self.p-2) if aux>0 else np.inf

        if p>2:
            aux = np.min(np.maximum(dom2_l, -dom2_u))
            convexity_param = (self.p-1) * aux**(self.p-2)  if aux>0 else 0

            aux =  np.abs(np.max(np.maximum(-taylor_l,taylor_u)))
            Lipschitz = (self.p-1) * aux**(self.p-2) if aux<np.inf else np.inf

        if p==2:
            convexity_param = 1. if lin_taylor_l is None and lin_taylor_u is None else 0.
            Lipschitz = 1. if constr_l is None and constr_u is None else np.inf

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
                                self._f_deriv,self._f_second_deriv,-np.inf, np.inf,
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
                                -np.inf, np.inf, 
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
    conj_tol : float
        A tolerance in the coputation of the conjugate allowing a slight deviation above from 1.0. Defaults: 1e-16
    """
    def __init__(self, domain,
                constr_l=None, constr_u=None, lin_taylor_l=None, lin_taylor_u=None,
                quad_taylor_l=None, quad_taylor_u=None, conj_tol = 1e-16,**kwargs):
        self.conj_tol = conj_tol
        super().__init__(domain,conj_dom_u=1.,conj_dom_l=-1.,
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                         quad_taylor_l=quad_taylor_l, quad_taylor_u=quad_taylor_u,
                         **kwargs
                         )

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
        ind = (res>1+self.conj_tol)
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

    def _ptw_dist_subdiff(self, vstar, x):
        diff = self.subgradient(x)
        diff -= vstar
        zeroind = (x==0)
        diff[zeroind] = np.maximum(np.abs(vstar[zeroind])-self.measure[zeroind],0.)
        return diff

    def _conj_ptw_dist_subdiff(self, v, xstar):
        w = v.copy()
        ind = np.where(np.isclose(xstar,self.measure))
        w[ind] = np.minimum(w[ind],0.)
        ind = np.where(np.isclose(xstar,-self.measure))
        w[ind] = np.maximum(w[ind],0.)
        return w

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
    data: explcitly declared data. 
        If data is given, then w is w+data; if the parameter w is not given, then w is just the data.
    """

    def __init__(self, domain, w=None,
                 constr_l=None, constr_u=None, lin_taylor_l=None, lin_taylor_u=None,
                 quad_taylor_l=None, quad_taylor_u=None, data = None,
                 **kwargs):
        if(w is None):
            w=domain.zeros()
        if not w in domain and not isinstance(w, (int,float,np.floating,np.integer)):
            raise ValueError(Errors.value_error('w not in domain and not scalar.'))
        if w in domain:
            self._w=w.copy()
        else:
            self._w= np.broadcast_to(w,domain.shape).copy()
        if(data is not None):
            if(data not in domain):
                raise ValueError(Errors.value_error('data not in domain.'))
            w=self._w+data#update parameter w to set correct constants everywhere
        if np.min(w)<0:
            raise ValueError(Errors.value_error('w must be non-negative.'))
        if constr_u is not None and np.any(constr_u<np.inf):
            Lipschitz = np.inf
        elif quad_taylor_l is not None or lin_taylor_l is not None:
            taylor_l = quad_taylor_l if quad_taylor_l is not None else lin_taylor_l
            Lipschitz =  np.max(w/taylor_l**2) if np.min(taylor_l)>0 else np.inf
        else: 
            Lipschitz = np.inf

        if lin_taylor_l is not None or lin_taylor_u is not None:
            convexity_param = 0 
        elif constr_u is not None or quad_taylor_u is not None:
            u = constr_u if constr_u is not None else quad_taylor_u
            convexity_param = np.min(w/u**2) 
        else:
            convexity_param = 0.
    
        super().__init__(domain,dom_l=1e-14*w,
                         conj_dom_u=domain.ones()-1e-14*w,
                         convexity_param = convexity_param, Lipschitz= Lipschitz,                              
                          constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                          quad_taylor_l=quad_taylor_l,quad_taylor_u=quad_taylor_u,
                         **kwargs
                         )
        self.data = data
        

    @memoized_property
    def w(self):
        if self.is_data_func:
            return self._w + self.data
        else:
            return self._w

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        self.log.warning("Setting new data outside of constructor currently does not update the convexity and Lipschitz constants and taylor or constraint parameters.")
        if new_data is None:
            self.is_data_func = False
            del self.w
        elif new_data in self.domain:
            self.is_data_func = True
            self._data = new_data
            del self.w
        else:
            raise ValueError(Errors.not_in_vecsp(new_data,self.domain,vec_name="new data vector",space_name="domain of functional"))
        
    @data.deleter
    def data(self):
        if self.is_data_func:
            del self._data
            del self.w
            self.is_data_func = False

    def as_data_func(self,data):
        self.data=data
        return self

    def _f(self, u,**kwargs):
        if 'w' in kwargs.keys():
            raise ValueError(Errors.value_error('second parameter w of KullbackLeibler must be fixed in constructor.'))
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
            raise ValueError(Errors.value_error('first parameter w of KullbackLeibler must be fixed in constructor.'))
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        if not np.all(np.logical_or(np.logical_not(u==0),wm==0)):
            raise ValueError(Errors.value_error('argument cannot be 0 at positions where w is not 0'))
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
            raise ValueError(Errors.value_error('second parameter w of KullbackLeibler must be fixed in constructor.'))
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
            raise ValueError(Errors.value_error('second parameter w of KullbackLeibler must be fixed in constructor.'))
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w    
        if not np.all(np.logical_or(np.logical_not(u_star==1),wm==0)):
            raise ValueError(Errors.value_error('argument cannot be 1 at positions where w is not 0.'))
        # memory efficient implementation of
        # toret = wm/(1-u_star)
        toret = np.subtract(1.,u_star)
        np.divide(wm,toret,out = toret)
        # end
        toret[u_star==1] = 0
        return toret 
    
    def _f_conj_second_deriv(self, u_star,**kwargs):
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        if not np.all(np.logical_or(np.logical_not(u_star==1),wm==0)):
            raise RuntimeError(Errors.runtime_error(f"Whenever u^ast is not 1 the first argument of initial Kullback-Leibler cannot be zero for the computation of second derivativ of conjugate functional.", self,"_f_conj_second_deriv"))
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
    r"""Relative Entropy divergence define by

    .. math::
        F_w(u) = KL(u,w) = \int (u(x)\ln \frac{u(x)}{w(x)}) \mathrm{d}x

    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        Domain on which to define the Relative Entropy divergence
    w: scalar or in domain [optional, default: 1]
        second argument of the Relative Entropy diverengence; reference value if used as penalty functional
    constr_l,constr_u,lin_taylor_l, lin_taylor_u: None, scalar or in domain
        see IntegralFunctional  
    """

    def  __init__(self, domain,w=1.,
                  constr_l=None, constr_u=None, lin_taylor_l=None, lin_taylor_u=None,
                  quad_taylor_l=None, quad_taylor_u=None,
                  **kwargs
                  ):
        if np.isscalar(w):
            w = np.broadcast_to(w,domain.shape)
        elif w not in domain:
            raise TypeError(Errors.not_in_vecsp(w,domain,vec_name="second argument",add_info=f" Error occured setting up the Relative Entropy functional."))
        if np.min(w)<=0:
            raise ValueError(Errors.value_error("The second argument in the RelativeEntropy needs to be positiv!", self))
        self.w= w.copy()

        if constr_u is not None and np.any(constr_u<np.inf):
            Lipschitz = np.inf
        elif quad_taylor_l is not None or lin_taylor_l is not None:
            taylor_l = quad_taylor_l if quad_taylor_l is not None else lin_taylor_l
            if not np.isscalar(taylor_l):
                taylor_l= min(taylor_l)
            Lipschitz = np.max(1/taylor_l) if np.min(taylor_l)>0 else np.inf
        else: 
            Lipschitz = np.inf

        if lin_taylor_u is not None or lin_taylor_l is not None:
            convexity_param = 0.
        elif constr_u is not None or quad_taylor_u is not None:
            u = constr_u if constr_u is not None else quad_taylor_u
            convexity_param = np.min(1./u) 
        else:
            convexity_param = 0.

        super().__init__(domain,dom_l = 1e-14*w,
                         convexity_param=convexity_param, Lipschitz= Lipschitz, 
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                         quad_taylor_l=quad_taylor_l,quad_taylor_u=quad_taylor_u,
                         **kwargs
                         )

    def _f(self, u,**kwargs):
        wm = self.w[kwargs['mask']] if 'mask' in kwargs.keys() else self.w
        res=np.zeros_like(u)
        # memory efficient implementation of 
        # res[ind_upos]=u[ind_upos] * np.log(u[ind_upos]/wm[ind_upos])
        np.divide(u,wm, out= res)
        np.log(res,out=res,where=res>0)
        res *= u
        # end
        res[u<0] = np.inf
        # res[u==0] = 0. # this is already the case due to initialization with zeros
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
        
        thres = np.log(np.finfo(v.dtype).max)-np.maximum(0.,np.log(1./tau))-np.maximum(0.,np.log(np.max(wm)))-1.
        v_mod = (v<=thres*tau) 
        # For v>=thres*tau an overflow occurs in the exponential (for standard doubles thres ~ 710) or in the subsequent multiplication and division.
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
        F(x) = 1/2 |x|^2                &\quadif  |x|\leq \sigma \\
        F(x) = \sigma |x|-\sigma^2/2    &\quadif  |x|>\sigma


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
        if not isinstance(domain,MeasureSpaceFcts): raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,"Huber domain needs to be a MeasureSPaceFcts isntance."))
        if not isinstance(sigma, (float,int)) and sigma not in domain:
            raise TypeError(Errors.type_error(f"Sigma in the HuberFunctional needs to be a scalar or elemnt in the domain! Given:"+"\n\t "+f"sigma = {sigma}",self))
        if isinstance(sigma,int):
            self.sigma = np.broadcast_to(np.real(float(sigma)),domain.shape)
        if isinstance(sigma, float):
            self.sigma = np.broadcast_to(np.real(sigma),domain.shape)
        else:
            self.sigma = np.real(sigma)
        if np.min(sigma)<=0:
            raise ValueError(Errors.value_error(f'sigma must be positive. min(sigma)={np.min(sigma)}',self))
        if as_primal:
            super().__init__(domain,Lipschitz=1.,
                             conj_dom_l=-self.sigma, conj_dom_u = self.sigma,                             
                             **kwargs)
            self.conjugate = QuadraticIntv(domain,as_primal=False,sigma=sigma,eps=eps)
        else:
            dual_domain = deepcopy(domain)
            dual_domain.measure = 1./domain.measure
            super().__init__(dual_domain, Lipschitz=1., **kwargs)
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
        F(x) = 1/2 |x|^2    &\quadif |x|\leq \sigma(x) \\
        F(x) = \infty    &\quadif |x|>\sigma(x)


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
        if not isinstance(domain,MeasureSpaceFcts): raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,"QuadraticIntv domain needs to be a MeasureSPaceFcts isntance."))
        if not isinstance(sigma, (float,int)) and sigma not in domain:
            raise TypeError(Errors.type_error(f"Sigma in the QuadraticIntv needs to be a scalar or elemnt in the domain! Given:"+"\n\t "+f"sigma = {sigma}",self))
        if np.min(sigma)<=0:
            raise ValueError(Errors.value_error("The interval width in the QuadtraticIntv needs to be positiv!", self))
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
            super().__init__(dual_domain, convexity_param=1,                                                     
                         **kwargs)   
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
    
    def _ptw_dist_subdiff(self, vstar, x):
        diff = vstar.copy()
        diff -= self.subgradient(x)
        ubind = (x==self.sigma)
        diff[ubind] = np.minimum(vstar[ubind] - self.sigma[ubind],0.)
        lbind = (x==-self.sigma)
        diff[lbind] = np.maximum(vstar[lbind] + self.sigma[lbind],0.)
        return diff


class QuadraticNonneg(IntegralFunctionalBase):
    r"""Functional 

    .. math::
        F(x) = 1/2 |x|^2    &\quadif x\geq 0 \\
        F(x) = \infty       &\quadif  x<0

    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        domain on which functional is defined 

    """

    def  __init__(self, domain,**kwargs):
        super().__init__(domain,convexity_param = 1.,dom_l=0.,
                         methods =  {'eval', 'subgradient', 'hessian', 'proximal'},
                         conj_methods =  {'eval', 'subgradient', 'hessian', 'proximal'},
                         **kwargs)

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
    
    def _ptw_dist_subdiff(self, vstar, x):
        raise NotImplementedError

class QuadraticBilateralConstraints(LinearCombination):
    r""" Returns `Functional` defined by 

    .. math::
        F(x) = \frac{\alpha}{2}\|x-x_0\|^2  &\quadif lb\leq x\leq ub \\
        F(x) = np.inf &\quadelse


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
        if not isinstance(domain,MeasureSpaceFcts): raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,"QuadraticBilateralConstraints domain needs to be a MeasureSPaceFcts isntance."))
        if isinstance(lb,(float,int)):
            lb = lb*domain.ones()
        elif lb is None:
            lb = domain.zeros()
        elif lb not in domain:
            raise TypeError(Errors.type_error(f"Lower bound lb in QuadraticBilateralConstraints needs to be a scalar or elemnt in the domain! Given:"+"\n\t "+f"lb = {lb}"))
        if isinstance(ub,(float,int)):
            ub = ub*domain.ones()
        elif ub is None:
            ub = domain.zeros()
        elif ub not in domain:
            raise TypeError(Errors.type_error(f"Upper bound ub in QuadraticBilateralConstraints needs to be a scalar or elemnt in the domain! Given:"+"\n\t "+f"ub = {ub}"))
        if np.any(lb>=ub):
            raise TypeError(Errors.type_error(f"The lower bound needs to be below the upper bound in QuadraticBilateralCOnstraint:"+"\n\t "+f"lb = {lb},"+"\n\t "+f"ub = {ub}"))
        if x0 is None:
            x0 =0.5*(lb+ub)
        elif isinstance(x0,(float,int)):
            x0 = x0*domain.ones()
        elif x0 not in domain:
            raise TypeError(Errors.type_error(f"Reference value x0 in QuadraticBilateralConstraints needs to be a scalar or elemnt in the domain! Given:"+"\n\t "+f"x0 = {x0}"))

        if not isinstance(alpha,(float,int)):
            raise TypeError(Errors.not_instance(alpha,float,add_info="Regularization parameter alpha in QuadraticBilateralConstraints needs to be a scalar!"))

        self.lb = lb; self.ub = ub; self.x0 =x0; self.alpha = alpha
        F = QuadraticIntv(domain,sigma=(ub-lb)/2.,**kwargs)
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

    .. math::
        F(x) = \frac{a}{2}\|x-x_0\|^2 &\quad\text{if } lb\leq x \\
        F(x) =  \inf  &\quad\text{else }


    Parameters
    ----------
    domain: regpy.vecsps.MeasureSpaceFcts
        domain on which the functional is defined
    lb: domain or float [default: None]
        lower bound (zero in the default case)
    x0: domain or float [default: None]
        lower bound (zero in the default case)
    a : float or int
        scaling factor
    """
    if not isinstance(domain,MeasureSpaceFcts): raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,"QuadraticLowerBound domain needs to be a MeasureSPaceFcts isntance."))
    if isinstance(lb,(float,int)):
        lb = lb*domain.ones()
    elif lb is None:
        lb = domain.zeros()
    elif lb not in domain:
        raise TypeError(Errors.type_error(f"Lower bound lb in QuadraticLowerBound needs to be a scalar or elemnt in the domain! Given:"+"\n\t "+f"lb = {lb}"))
    if isinstance(x0,(float,int)):
        x0 = x0*domain.ones()
    elif x0 is None:
        x0 = domain.zeros()
    elif x0 not in domain:
        raise TypeError(Errors.type_error(f"Reference value x0 in QuadraticLowerBound needs to be a scalar or elemnt in the domain! Given:"+"\n\t "+f"x0 = {x0}"))
    if not isinstance(a,(float,int)):
            raise TypeError(Errors.not_instance(a,float,add_info="Scaling factor a in QuadraticLowerBound needs to be a scalar!"))

    F = QuadraticNonneg(domain)
    lin = LinearFunctional(lb-x0,domain=domain,gradient_in_dual_space=False)
    offset = 0.5*(np.sum((x0**2-lb**2)*domain.measure))
    return a*HorizontalShiftDilation(F,shift=lb)+ a*lin + a*offset

class QuadraticPositiveSemidef(Functional):
    r"""Functional 

    .. math::
        F(x) = 1/2 ||x||_{HS}^2    &\quad\text{if } x\geq 0 \text{ and (optional) } tr(x)=c \\
        F(x) = \infty       &\quad\text{else}

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
        if not isinstance(domain,UniformGridFcts): raise TypeError(Errors.not_instance(domain,UniformGridFcts,"QuadraticLowerBound domain needs to be a UniformGridFcts isntance."))
        if domain.ndim!=2 or domain.shape[0]!=domain.shape[1] or domain.volume_elem!=1.:
            raise ValueError(Errors.value_error("The domain for QuadraticPositiveSemidef must be a two-dimensional domain with identical spread in the two domensions and uniform one measure! This is to represent vectors as a matrix."))
        if tol<0:
            raise ValueError(Errors.value_error("The tolerance for determinating PSD of a matrix needs to be non-negative"))
        if trace_val is not None and trace_val<=0:
            raise ValueError(Errors.value_error("The desired value of traces needs to be either None or positive!"))
        self.tol=tol
        if(trace_val is not None):
            self.has_trace_constraint=True
            self.trace_val=trace_val
        else:
            self.has_trace_constraint=False
        super().__init__(domain,Lipschitz=1.,convexity_param=1.,
                         methods = {'eval','subgradient','hessian','dist_subdiff','proximal'},
                         conj_methods = {'eval'},
                         **kwargs)

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
        if not isinstance(domain,NumPyVectorSpace):
            raise TypeError(Errors.not_instance(domain,NumPyVectorSpace,"To construct a L1Generic functional you need a NumPyVectorSpace"))
        super().__init__(domain,
                         methods = {'eval','subgradient','hessian','proximal'})

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

from regpy.operators import ForwardFDGradient
class TVUniformGridFcts(Composed):
    r"""Total Variation Norm: For :math:`C^1` functions the :math:`l^1`-norm of the gradient on a `UniformGrid`

    Parameters
    ----------
    domain : regpy.vecsps.UniformGridFcts
        Underlying domain. 
    h_domain : regpy.hilbert.HilbertSapce (default: L2)
        Underlying Hilbert space for proximal. 
    beta: float (>=0) [optional, default:0.]
        If beta>0, the L^1-norm is approximated by a Huber functional with this parameter. 
    boundary_condition: 'Neum', 'Diri' or 'per' (default: 'Neum')
        Boundary condition in the discretization of gradient, see ForwardFDGradient
    """

    def __init__(self, domain,beta=0.,boundary_condition='Neum'):
        if not isinstance(domain, UniformGridFcts):
            raise TypeError(Errors.not_instance(domain,UniformGridFcts,'only implemented for UniformGridFcts'))
        if not isinstance(beta,float) and (beta>=0.):
            raise ValueError(Errors.value_error('beta must be a non-negative float'))
        else:
            self.beta = beta
        self.grad = ForwardFDGradient(domain, boundary_condition=boundary_condition)
        if beta==0.:
            self.func = L1L2(self.grad.codomain)
        else:
            self.func = HuberL2(self.grad.codomain)

        super().__init__(self.func, op= self.grad, op_norm = self.grad.norm(),
                         methods = {'eval','proximal'},conj_methods = {'proximal'})

    def _proximal(self, x, tau, stepsize_safety=2., maxiter=1000,tol=0.01):
        r"""Prox computation after the method suggested by A. Chambolle (J. Math. Imaging and Vision 20: 89-97, 2004) 
        Parameters
        ----------
        x: np.array 
            First argument of prox
        tau: float >=0
            Second (scaling) argument of prox
        stepsize_safety: float [optional, default: 2.]
            Safety parameter for the stepsize. Convergence is guaranteed for values <=1, but empirically, 
            best results are optained for stepsize_safety =2.
        maxiter: int [optional: default: 1000]
            Maximum number of iterations
        tol: float>=0 [optional, default: 0.01]
            Tolerance parameter for stopping criterion. Iteration is stopped if two consecutive 
            iteratives differ by less than tol in the maxium norm. 
        """
        if self.beta!=0.:
            raise ValueError("Chambolle's method only works for beta=0.")
        p = self.grad.codomain.zeros()
        lastp = p
        stepsize = stepsize_safety/self.grad.norm()**2
        for i in range(maxiter):
            update = stepsize*self.grad( -self.grad.adjoint(p)-x/tau)
            p = (p+update) / np.expand_dims(1.+self.func.vector_norm(x = update, axis=self.func._vaxes),-1)
            if np.max(self.func.vector_norm(x = p-lastp, axis=self.func._vaxes))<tol:
                self.log.info(f'TV prox Chambolle terminated after {i} iterations.')
                break
            else:
                lastp=p
        return x+tau*self.grad.adjoint(p)