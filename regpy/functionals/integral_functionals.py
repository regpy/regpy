from copy import deepcopy

import numpy as np
from scipy.special import lambertw

from regpy.util import Errors, memoized_property
from regpy.vecsps.numpy import *
import regpy.operators as rop
from regpy.hilbert import L2

from .base import Functional, LinearFunctional, LinearCombination,HorizontalShiftDilation, NotInEssentialDomainError,NotTwiceDifferentiableError, as_functional

__all__ = ["IntegralFunctionalBase","LppPower","L1MeasureSpace","KullbackLeibler","RelativeEntropy","Huber","QuadraticIntv","QuadraticBilateralConstraints","QuadraticLowerBound","QuadraticNonneg","L1Generic"]


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
    
    (where _f_deriv, _f_prox, _f_conj, and _f_conj_prox must be able to handle out=input) since 

    .. math::
        F'[g]h = \int_\Omega h(x)(\partial_1 f)(g(x),x)

    is a functional of the same type, and

    .. math::
        \mathrm{prox}_{\tau F}(v)(x) = \mathrm{prox}_{\tau f(\cdot,x)}(v(x)).

    It allows to replace the function :math:`f(\cdot,x)` on the interval :math:`(-\infty,a(x))` by one of 
    the following terms, and analogously on :math:`(b(x),\infty)` with :math:`-\infty<=a(x) < b(x)\leq \infty`

    .. math::
        \infty &\text{hard constraint}\\
        f(a(x),x) + (x-a(x)) \partial_1 f(a(x),x) & \text{first order Taylor approximation}\\ 
        f(a(x),x) + (x-a(x)) \partial_1 f(a(x),x) +0.5(x-a(x))^2 \partial_1^2 f(a(x),x)& \text{second order Taylor approximation}\\ 

    An important use case of quadratic Taylor approximations is the improvement of the convexity constant 
    (and the Lipschitz constant of the conjugate functional) in convex optimization methods. 
    As opposed to hard constraints, this preserves Lipschitz continuity of derivative. 
    Note that hard constraints correspond to linear Taylor approximations of the conjugate functionals and vice versa, 
    whereas quadratic Taylor approximations lead to quadratic Taylor approximation of the conjugate. 

    Only one of the above replacements may for specified for all :math:`x` on each side.

    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a `regpy.vecsps.MeasureSpaceFcts`
    nat_dom_l,nat_dom_u : float or np.ndarray, default: -np.inf, np.inf
        lower and upper boundary of the natural essential domain of :math:`f`, i.e., the interval on which :math:`f` without artificial constraints or Taylor expansion is finite 
        If dom_l or dom_u are finite, :math:`f` should be finite at these points (possibly very large if :math:`f` tends to infinity there) 
        If the domain depends on the point `x` and/or arguments in `**kwargs`, this must be a NumPy array.
    conj_nat_dom_l,conj_nat_dom_u : float or np.ndarray [default: -np.inf and np.inf, rsp.]
        lower and upper boundary of the natural essential domain of the conjugate of :math:`f` 
        (as vector in primal space)
    constr_u: None or float or np.ndarray [default: None]
        If not None, an upper hard constraint is imposed, and constr_u specifies :math:`b(x)`
    constr_l: None or float or np.ndarray [default: None]
        If not None, a lower hard constraint is imposed, and constr_u specifies :math:`a(x)` 
    lin_taylor_u: None or float or np.ndarray [default: None]
        If not None, :math:`b(x)` for linear Taylor on the right-hand side
    lin_taylor_l: None or float or np.ndarray [default: None]
        If not None, :math:`a(x)` for linear Taylor on the left hand side
    quad_taylor_u: None or float or np.ndarray [default: None]
        If not None, :math:`b(x)` for quadratic Taylor on the right-hand side
    quad_taylor_l: None or float or np.ndarray [default: None]
        If not None, :math:`a(x)` for quadratic Taylor on the left hand side
    methods, conj_methods = set of strings or None [default: None]
        Names of the methods implemented by an IntegralFunctionalBase instance (see Functional!).
        In the default case all methods are indicated as being implemented. 
    """

    def __init__(self, domain,
                 nat_dom_l=-np.inf, nat_dom_u=np.inf, 
                 conj_nat_dom_l=-np.inf,conj_nat_dom_u=np.inf,
                 constr_l=None, constr_u=None,
                 lin_taylor_l=None, lin_taylor_u=None,
                 quad_taylor_l=None, quad_taylor_u= None,
                 Lipschitz = np.inf, convexity_param=0.,
                 methods = None, conj_methods = None,
                 is_data_func = False,
                 **kwargs):
        if not isinstance(domain,MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,add_info="IntegralFunctionalBase only implemented for real-valued MeasureSpaceFcts domains."))
        if not domain.shape_codomain == () or not np.issubdtype(domain.dtype, np.floating):
            raise RuntimeError(Errors.runtime_error(f'IntegralFunctionalBase only implemented for real-valued MeasureSpaceFcts. Got dtype={domain.dtype}, shape_codomain = {domain.shape_codomain}. Consider using  VectorIntegralFunctional instead.'))
        self.h_domain = L2(domain)
        self.measure = np.broadcast_to(domain.measure,domain.shape) if np.isscalar(domain.measure) else domain.measure
        self.domain = domain

        all_methods = {'eval', 'subgradient', 'hessian', 'proximal', 'dist_subdiff'}

        super().__init__(domain, Lipschitz=Lipschitz, convexity_param=convexity_param,
                         is_separable=True,
                         methods = methods if methods is not None else all_methods, 
                         conj_methods = conj_methods if conj_methods is not None else all_methods,
                         is_data_func = is_data_func
                         )

        self._buffer_status = {}  
        # dictionary with keys corresponding to buffer names and boolean values indicating if they are currently in use (True) 
        # or not (False). This dictionary helps to prevent accidental overwriting of buffers by recursions or multithreading. 
        self.kwargs = kwargs
        if 'logging_level' in kwargs.keys():
            self.log.setLevel(kwargs['logging_level'])
        
        if sum([trunc is not None for trunc in [constr_l,lin_taylor_l,quad_taylor_l]])>1:
            raise ValueError(Errors.value_error('At most one of the parameters constr_l,lin_taylor_l, quad_taylor_l may be specified.',self))
        if sum([trunc is not None for trunc in [constr_u,lin_taylor_u,quad_taylor_u]])>1:
            raise ValueError(Errors.value_error('At most one of the parameters constr_u,lin_taylor_u, quad_taylor_u may be specified.',self))     
       
        if nat_dom_l in domain: 
            self._nat_dom_l = nat_dom_l
        elif np.isscalar(nat_dom_l):
            self._nat_dom_l = np.broadcast_to(nat_dom_l,self.domain.shape)   
        else:
            raise TypeError(Errors.type_error(f"The lower bound of the essential domain can only be either a scalar or an element in the domain. Given nat_dom_l = {nat_dom_l}"))
        
        if nat_dom_u in domain: 
            self._nat_dom_u = nat_dom_u
        elif np.isscalar(nat_dom_u):
            self._nat_dom_u = np.broadcast_to(nat_dom_u,self.domain.shape)   
        else:
            raise TypeError(Errors.type_error(f"The upper bound of the essential domain can only be either a scalar or an element in the domain. Given nat_dom_u = {nat_dom_u}"))
        
        if np.isscalar(conj_nat_dom_l):
            if isinstance(domain,UniformGridFcts) or conj_nat_dom_l==-np.inf:
                self._conj_nat_dom_l = np.broadcast_to(conj_nat_dom_l*self.measure,domain.shape)
            else:
                self._conj_nat_dom_l = self.h_domain.gram(np.broadcast_to(conj_nat_dom_l,self.domain.shape))
        elif conj_nat_dom_l in domain:
            self._conj_nat_dom_l = self.h_domain.gram(conj_nat_dom_l)
        else:
            raise TypeError(Errors.type_error(f"The lower bound of the conjugate essential domain can only be either a scalar, an element in the domain or a MeasureSpaceFcts. Given conj_nat_dom_l = {conj_nat_dom_l} of type {type(conj_nat_dom_l)}."))

        if np.isscalar(conj_nat_dom_u):
            if isinstance(domain,UniformGridFcts) or conj_nat_dom_u==np.inf:
                self._conj_nat_dom_u = np.broadcast_to(conj_nat_dom_u*self.measure,domain.shape)
            else:
                self._conj_nat_dom_u = self.h_domain.gram(np.broadcast_to(conj_nat_dom_u,self.domain.shape))
        elif conj_nat_dom_u in domain:
            self._conj_nat_dom_u = self.h_domain.gram(conj_nat_dom_u)
        else:
            raise TypeError(Errors.type_error(f"The upper bound of the conjugate essential domain can only be either a scalar, an element in the domain or a MeasureSpaceFcts. Given conj_nat_dom_u = {conj_nat_dom_u} of type {type(conj_nat_dom_u)}."))
        
        self._constr_l, self._constr_u, self._lin_taylor_l, self._lin_taylor_u, self._quad_taylor_l, self._quad_taylor_u = None, None, None, None,  None, None

        if constr_l is not None: self.constr_l = constr_l
        if constr_u is not None: self.constr_u = constr_u
        if lin_taylor_l is not None: self.lin_taylor_l = lin_taylor_l
        if lin_taylor_u is not None: self.lin_taylor_u = lin_taylor_u
        if quad_taylor_l is not None: self.quad_taylor_l = quad_taylor_l
        if quad_taylor_u is not None: self.quad_taylor_u = quad_taylor_u
        
    @property
    def constr_l(self):
        return self._constr_l
    
    @constr_l.setter
    def constr_l(self, constr_l):
        if constr_l is None or (constr_l is not None and np.all(constr_l<self._nat_dom_l)):
            self._constr_l = None
            self._delete_f_l()
        elif self.lin_taylor_l is not None or self.quad_taylor_l is not None:
            raise ValueError(Errors.value_error("Cannot set lower constraint when Taylor expansion is already set. First set those to None.",self))
        elif constr_l in self.domain: 
            self._constr_l = constr_l
            self._precompute_f_l()
        elif np.isscalar(constr_l):
            self._constr_l = np.broadcast_to(constr_l,self.domain.shape)
            self._precompute_f_l()
        else:  
            raise TypeError(Errors.type_error(f"The lower constraint if defined, can only be either a scalar or an element in the domain. Given constr_l = {constr_l}"))
        
    @property
    def constr_u(self):
        return self._constr_u
    
    @constr_u.setter
    def constr_u(self, constr_u):
        if constr_u is None or (constr_u is not None and np.all(constr_u>self._nat_dom_u)):
            self._constr_u = None
            self._delete_f_u()
        elif self.lin_taylor_u is not None or self.quad_taylor_u is not None:
            raise ValueError(Errors.value_error("Cannot set upper constraint when Taylor expansion is already set. First set those to None.",self))
        elif constr_u in self.domain: 
            self._constr_u = constr_u
            self._precompute_f_u()
        elif np.isscalar(constr_u):
            self._constr_u = np.broadcast_to(constr_u,self.domain.shape)
            self._precompute_f_u()
        else:  
            raise TypeError(Errors.type_error(f"The upper constraint if defined, can only be either a scalar or an element in the domain. Given constr_u = {constr_u}"))

    @property
    def lin_taylor_l(self):
        return self._lin_taylor_l

    @lin_taylor_l.setter        
    def lin_taylor_l(self, lin_taylor_l):
        if lin_taylor_l is None or (lin_taylor_l is not None and np.all(lin_taylor_l<self._nat_dom_l)):
            self._lin_taylor_l = None
            self._delete_f_l()
        elif self.constr_l is not None or self.quad_taylor_l is not None:
            raise ValueError(Errors.value_error("Cannot set lower linear Taylor expansion when constraint or quadratic Taylor expansion is already set. First set those to None.",self))
        elif lin_taylor_l in self.domain:       
            self._lin_taylor_l = lin_taylor_l
            self._precompute_f_l()      
        elif np.isscalar(lin_taylor_l):
            self._lin_taylor_l = np.broadcast_to(lin_taylor_l,self.domain.shape)
            self._precompute_f_l()
        else: 
            raise TypeError(Errors.type_error(f"The lower point at which to replace with an Taylor first order expansion if defined, can only be either a scalar or an element in the domain. Given lin_taylor_l = {lin_taylor_l}"))
        
    @property
    def lin_taylor_u(self):
        return self._lin_taylor_u

    @lin_taylor_u.setter
    def lin_taylor_u(self, lin_taylor_u):
        if lin_taylor_u is None or (lin_taylor_u is not None and np.all(lin_taylor_u>self._nat_dom_u)):
            self._lin_taylor_u = None
            self._delete_f_u() 
        elif self.constr_u is not None or self.quad_taylor_u is not None:
            raise ValueError(Errors.value_error("Cannot set upper linear Taylor expansion when constraint or quadratic Taylor expansion is already set. First set those to None.",self)) 
        elif lin_taylor_u in self.domain: 
            self._lin_taylor_u = lin_taylor_u
            self._precompute_f_u()
        elif np.isscalar(lin_taylor_u):
            self._lin_taylor_u = np.broadcast_to(lin_taylor_u,self.domain.shape)
            self._precompute_f_u()
        else: 
            raise TypeError(Errors.type_error(f"The upper point at which to replace with an Taylor first order expansion if defined, can only be either a scalar or an element in the domain. Given lin_taylor_u = {lin_taylor_u}"))
    
    @property
    def quad_taylor_l(self):
        return self._quad_taylor_l
    
    @quad_taylor_l.setter
    def quad_taylor_l(self, quad_taylor_l):
        if quad_taylor_l is None or (quad_taylor_l is not None and np.all(quad_taylor_l<self._nat_dom_l)):
            self._quad_taylor_l = None
            self._delete_f_l()
        elif self.constr_l is not None or self.lin_taylor_l is not None:
            raise ValueError(Errors.value_error("Cannot set lower quadratic Taylor expansion when constraint or linear Taylor expansion is already set. First set those to None.",self))
        elif quad_taylor_l in self.domain: 
            self._quad_taylor_l = quad_taylor_l
            self._precompute_f_l()
        elif np.isscalar(quad_taylor_l):
            self._quad_taylor_l = np.broadcast_to(quad_taylor_l,self.domain.shape)
            self._precompute_f_l()
        else: 
            raise TypeError(Errors.type_error(f"The lower point at which to replace with an Taylor second order expansion if defined, can only be either a scalar or an element in the domain. Given quad_taylor_l = {quad_taylor_l}"))
        
    @property   
    def quad_taylor_u(self):
        return self._quad_taylor_u
    
    @quad_taylor_u.setter
    def quad_taylor_u(self, quad_taylor_u):
        if quad_taylor_u is None or (quad_taylor_u is not None and np.all(quad_taylor_u>self._nat_dom_u)):
            self._quad_taylor_u = None
            self._delete_f_u()
        elif self.constr_u is not None or self.lin_taylor_u is not None:
            raise ValueError(Errors.value_error("Cannot set upper quadratic Taylor expansion when constraint or linear Taylor expansion is already set. First set those to None.",self))
        elif quad_taylor_u is None or quad_taylor_u in self.domain: 
            self._quad_taylor_u = quad_taylor_u
            self._precompute_f_u()
        elif np.isscalar(quad_taylor_u):
            self._quad_taylor_u = np.broadcast_to(quad_taylor_u,self.domain.shape)
            self._precompute_f_u()
        else: 
            raise TypeError(Errors.type_error(f"The upper point at which to replace with an Taylor second order expansion if defined, can only be either a scalar or an element in the domain. Given quad_taylor_u = {quad_taylor_u}"))
  
    def _precompute_f_l(self):
        if self.constr_l is not None:
            active_ind_l = self.constr_l > self._nat_dom_l
            self.conj_taylor_l = np.full(self.domain.shape, -np.inf)
            self._f_deriv(
                self.dom_l, out=self.conj_taylor_l, mask=active_ind_l,
                **self.kwargs
            )
            np.multiply(
                self.conj_taylor_l, self.measure, out=self.conj_taylor_l,
                where=active_ind_l
            )
            self._if_constant_broadcast(self.conj_taylor_l)

            self.f_l = self.domain.zeros()
            self._f(self.dom_l, out=self.f_l, mask=active_ind_l, **self.kwargs)
            self._if_constant_broadcast(self.f_l, mask=active_ind_l)
        elif self.lin_taylor_l is not None:
            self.conj_f_l = self.domain.zeros()
            conj_active_ind_l = self._taylor_l > self.dom_l
            argument = np.empty_like(self.conj_dom_l)
            np.divide(
                self.conj_dom_l, self.measure, out=argument,
                where=conj_active_ind_l
            )
            self._f_conj(
                argument, out=self.conj_f_l, mask=conj_active_ind_l,
                **self.kwargs
            )
            self._if_constant_broadcast(self.conj_f_l, mask=conj_active_ind_l)
        elif self.quad_taylor_l is not None:
            taylor_active_l = self._taylor_l >= self.dom_l
            if np.any(taylor_active_l):
                self.conj_taylor_l = np.full(self.domain.shape, -np.inf)
                self._f_deriv(
                    self._taylor_l, out=self.conj_taylor_l,
                    mask=taylor_active_l, **self.kwargs
                )
                np.multiply(
                    self.conj_taylor_l, self.measure, out=self.conj_taylor_l,
                    where=taylor_active_l
                )
                self._if_constant_broadcast(self.conj_taylor_l)

                self.f_l = self.domain.zeros()
                self._f(
                    self._taylor_l, out=self.f_l, mask=taylor_active_l,
                    **self.kwargs
                )
                self._if_constant_broadcast(self.f_l, mask=taylor_active_l)

                argument = np.empty_like(self.conj_taylor_l)
                np.divide(
                    self.conj_taylor_l, self.measure, out=argument,
                    where=taylor_active_l
                )
                self.conj_f_l = self.domain.zeros()
                self._f_conj(
                    argument, out=self.conj_f_l, mask=taylor_active_l,
                    **self.kwargs
                )
                self._if_constant_broadcast(self.conj_f_l, mask=taylor_active_l)

                self.fpp_l = self.domain.zeros()
                self._f_second_deriv(
                    self._taylor_l, out=self.fpp_l, mask=taylor_active_l,
                    **self.kwargs
                )
                self._if_constant_broadcast(self.fpp_l, mask=taylor_active_l)

                self.fstarpp_l = self.domain.zeros()
                self._f_conj_second_deriv(
                    argument, out=self.fstarpp_l, mask=taylor_active_l,
                    **self.kwargs
                )
                self._if_constant_broadcast(self.fstarpp_l, mask=taylor_active_l)

    def _delete_f_l(self):
        if hasattr(self,'conj_taylor_l'):
            del self.conj_taylor_l
        if hasattr(self,'f_l'):
            del self.f_l
        if hasattr(self,'conj_f_l'):
            del self.conj_f_l
        if hasattr(self,'fpp_l'):
            del self.fpp_l
        if hasattr(self,'fstarpp_l'):
            del self.fstarpp_l

    def _precompute_f_u(self):
        if self.constr_u is not None:
            active_ind_u = self.constr_u < self._nat_dom_u
            self.conj_taylor_u = np.full(self.domain.shape, np.inf)
            self._f_deriv(
                self.dom_u, out=self.conj_taylor_u, mask=active_ind_u,
                **self.kwargs
            )
            np.multiply(
                self.conj_taylor_u, self.measure, out=self.conj_taylor_u,
                where=active_ind_u
            )
            self._if_constant_broadcast(self.conj_taylor_u)

            self.f_u = self.domain.zeros()
            self._f(self.dom_u, out=self.f_u, mask=active_ind_u, **self.kwargs)
            self._if_constant_broadcast(self.f_u, mask=active_ind_u)
        elif self.lin_taylor_u is not None:
            self.conj_f_u = self.domain.zeros()
            conj_active_ind_u = self._taylor_u < self.dom_u
            argument = np.empty_like(self.conj_dom_u)
            np.divide(
                self.conj_dom_u, self.measure, out=argument,
                where=conj_active_ind_u
            )
            self._f_conj(
                argument, out=self.conj_f_u, mask=conj_active_ind_u,
                **self.kwargs
            )
            self._if_constant_broadcast(self.conj_f_u, mask=conj_active_ind_u)
        elif self.quad_taylor_u is not None:
            taylor_active_u = self._taylor_u <= self.dom_u
            if np.any(taylor_active_u):
                self.conj_taylor_u = np.full(self.domain.shape, np.inf)
                self._f_deriv(
                    self._taylor_u, out=self.conj_taylor_u,
                    mask=taylor_active_u, **self.kwargs
                )
                np.multiply(
                    self.conj_taylor_u, self.measure, out=self.conj_taylor_u,
                    where=taylor_active_u
                )
                self._if_constant_broadcast(self.conj_taylor_u)

                self.f_u = self.domain.zeros()
                self._f(
                    self._taylor_u, out=self.f_u, mask=taylor_active_u,
                    **self.kwargs
                )
                self._if_constant_broadcast(self.f_u, mask=taylor_active_u)

                argument = np.empty_like(self.conj_taylor_u)
                np.divide(
                    self.conj_taylor_u, self.measure, out=argument,
                    where=taylor_active_u
                )
                self.conj_f_u = self.domain.zeros()
                self._f_conj(
                    argument, out=self.conj_f_u, mask=taylor_active_u,
                    **self.kwargs
                )
                self._if_constant_broadcast(self.conj_f_u, mask=taylor_active_u)

                self.fpp_u = self.domain.zeros()
                self._f_second_deriv(
                    self._taylor_u, out=self.fpp_u, mask=taylor_active_u,
                    **self.kwargs
                )
                self._if_constant_broadcast(self.fpp_u, mask=taylor_active_u)

                self.fstarpp_u = self.domain.zeros()
                self._f_conj_second_deriv(
                    argument, out=self.fstarpp_u, mask=taylor_active_u,
                    **self.kwargs
                )
                self._if_constant_broadcast(self.fstarpp_u, mask=taylor_active_u)

    def _delete_f_u(self):
        if hasattr(self,'conj_taylor_u'):
            del self.conj_taylor_u
        if hasattr(self,'f_u'):
            del self.f_u
        if hasattr(self,'conj_f_u'):
            del self.conj_f_u
        if hasattr(self,'fpp_u'):
            del self.fpp_u
        if hasattr(self,'fstarpp_u'):
            del self.fstarpp_u

    @property
    def _taylor_l(self)->np.ndarray:
        if self.lin_taylor_l is not None:
            return self.lin_taylor_l 
        elif self.quad_taylor_l is not None:
            return self.quad_taylor_l
        else:
            return None
        
    @property
    def _taylor_u(self)->np.ndarray:
        if self.lin_taylor_u is not None:
            return self.lin_taylor_u 
        elif self.quad_taylor_u is not None:
            return self.quad_taylor_u
        else:
            return None

    @property
    def _Lipschitz_exterior(self):
        return np.inf if (self.constr_l is not None or self.constr_u is not None) else 0.
    
    @property
    def Lipschitz(self):
        return max([super().Lipschitz,self._Lipschitz_exterior])

    @property
    def _convexity_param_exterior(self):
        return 0. if (self.lin_taylor_l is not None or self.lin_taylor_u is not None) else np.inf

    @property
    def convexity_param(self):
        return min([super().convexity_param,self._convexity_param_exterior])    

    @property
    def dom_l(self)->np.ndarray:
        toret = self._nat_dom_l.copy()
        if self._taylor_l is None:
            if not self.constr_l is None:
                toret = np.maximum(self._nat_dom_l, self.constr_l)
        else:
            toret[self._taylor_l>=self._nat_dom_l] = -np.inf
        return self._if_constant_broadcast(toret)
    
    @property
    def dom_u(self)->np.ndarray:
        toret = self._nat_dom_u.copy()
        if self._taylor_u is None:
            if self.constr_u is not None:
                toret = np.minimum(self._nat_dom_u, self.constr_u)
        else:
            toret[self._taylor_u<=self._nat_dom_u] = np.inf
        return self._if_constant_broadcast(toret)

    @property
    def conj_dom_l(self) -> np.ndarray:
        if self.constr_u is None and self._taylor_l is None:
            return self._conj_nat_dom_l
        toret = self._conj_nat_dom_l.copy()
        if self.lin_taylor_l is not None:
            active = self._taylor_l > self.dom_l
            self._f_deriv(self._taylor_l, out=toret, mask=active)
            np.multiply(toret, self.measure, out=toret, where=active)
        elif self.quad_taylor_l is not None:
            active = self._taylor_l > self._nat_dom_l
            np.copyto(toret, -np.inf, where=active)
        elif self.constr_l is not None:
            active = self.constr_l > self._nat_dom_l
            np.copyto(toret, -np.inf, where=active)
        return self._if_constant_broadcast(toret)

    @property
    def conj_dom_u(self) -> np.ndarray:
        if self.constr_u is None and self._taylor_u is None:
            return self._conj_nat_dom_u
        toret = self._conj_nat_dom_u.copy()
        if self.lin_taylor_u is not None:
            active = self._taylor_u < self.dom_u
            self._f_deriv(self._taylor_u, out=toret, mask=active)
            np.multiply(toret, self.measure, out=toret, where=active)
        elif self.quad_taylor_u is not None:
            active = self._taylor_u < self._nat_dom_u
            np.copyto(toret, np.inf, where=active)
        elif self.constr_u is not None:
            active = self.constr_u < self._nat_dom_u
            np.copyto(toret, np.inf, where=active)
        return self._if_constant_broadcast(toret)

    @property
    def everywhere_finite(self)->bool: 
        return np.all(self.dom_l==-np.inf) and np.all(self.dom_u==np.inf)
    
    @property
    def conj_everywhere_finite(self)->bool: 
        return np.all(self.conj_dom_l==-np.inf) and np.all(self.conj_dom_u==np.inf)

    """@property
    def constr_l(self):
        pass

    @property
    def lin_taylor_l(self):
        pass

    @property
    def quad_taylor_l(self):
        pass
    """

    def _if_constant_broadcast(self, x, mask=None):
        if not isinstance(x, np.ndarray) or x.shape != self.domain.shape:
            raise TypeError(Errors.type_error(
                "If constant use broadcasting in the IntegralFunctional init only allowed "
                "for NumPy arrays of same shape as domain." + "\n\t" + f"x = {x}"
            ))
        where = self._where(mask, x)
        if not np.any(where):
            return x
        first = x.flat[np.argmax(where)]
        close = np.isclose(x, first, rtol=1e-10, atol=1e-12)
        if np.all(close, where=where):
            x = np.broadcast_to(first, self.domain.shape)
        return x

    def _get_scratch(self, name, like):
        """Return a persistent scratch array (stored as ``self.<name>``) with the same
        shape and dtype as ``like``, reusing the previous allocation whenever possible.
        """
        if name in self._buffer_status and self._buffer_status[name] == True:
            raise RuntimeError(Errors.generic_message('Now request for buffer {name}, but this buffer is already in use.'))
        buf = getattr(self, name, None)
        if buf is None or buf.shape != like.shape or buf.dtype != like.dtype:
            buf = np.empty_like(like)            
            setattr(self, name, buf)
        self._buffer_status[name] = True
        return buf

    def _free_scratch(self,name):
        """Indicate that a buffer is no longer in use and hence not blocked.
        Does not free the memory!
        """
        self._buffer_status[name] = False

    @staticmethod
    def _prepare_out(out, like):
        """Return a writable output array compatible with ``like``.
        If ``out`` is None, a new array is allocated and returned.
        If ``out`` is an array, it is returned without initialization.
        This is used in pointwise operations to share code between allocating and
        in-place implementations.
        """
        if out is None:
            return np.empty_like(like)
        if not isinstance(out, np.ndarray):
            raise TypeError(Errors.not_instance(
                out, np.ndarray,
                add_info="The out argument of an integral functional must be a NumPy array."
            ))
        if out.shape != like.shape:
            raise ValueError(Errors.value_error(
                f"out and input must have identical shapes. Got {out.shape} and {like.shape}."
            ))
        if not out.flags.writeable:
            raise ValueError(Errors.value_error("The out array must be writable."))
        return out

    @staticmethod
    def _where(mask, like):
        """Return ``mask`` as a boolean array broadcast over the full input shape."""
        if mask is None:
            return np.broadcast_to(np.bool_(True), like.shape)
        where = np.asarray(mask, dtype=bool)
        if where.shape != like.shape:
            raise ValueError(Errors.value_error(
                f"mask and input must have identical shapes. Got {where.shape} and {like.shape}."
            ))
        return where

    def _apply_pointwise(self, func, x, out, mask=None, **kwargs):
        """Evaluate a pointwise method on full arrays using a NumPy ``where`` mask."""
        return func(x, out=out, mask=mask, **kwargs)

    def _assert_essential_domain(self, v, eps=1e-10, msg=None):
        buf = self._get_scratch('_buf',v)
        np.subtract(v, self.dom_u, out=buf)
        if np.any(self._buf>eps): raise  RuntimeError(Errors.generic_message(msg+f" argument too large in {self}."+"\n\t" + f"diff:{np.max(self._buf)},"+"\n\t"+ f"eps={eps}"))
        np.subtract(self.dom_l, v, out=buf)
        if np.any(self._buf>eps): raise  RuntimeError(Errors.generic_message(msg+f" argument too small in {self}."+"\n\t"+f"diff:{np.max(self._buf)},"+"\n\t"+f"eps={eps}"))
        self._free_scratch('_buf')

    def _assert_conj_essential_domain(self, vstar, eps=1e-10, msg=None):
        buf = self._get_scratch('_buf',vstar)        
        np.subtract(vstar, self.conj_dom_u, out=buf)
        if np.any(self._buf>eps): raise  RuntimeError(Errors.generic_message(msg+f" argument too large in {self}."+"\n\t"+f"diff:{np.max(self._buf)},"+"\n\t"+f"eps={eps}"))
        np.subtract(self.conj_dom_l, vstar, out=buf)
        if np.any(self._buf>eps): raise  RuntimeError(Errors.generic_message(msg+f" argument too small in {self}."+"\n\t"+f"diff:{np.max(self._buf)},"+"\n\t"+f"eps={eps}"))
        self._free_scratch('_buf')

    def _eval(self, v, func_vals=None):
        # See the comment in _conj.
        buf = self._get_scratch('_buf', v)
        try:
            mask = None
            if self._taylor_l is not None:
                v_small = v < self._taylor_l
                if np.any(v_small):
                    mask = ~v_small
                    buf[v_small] = v[v_small]
                    buf[v_small] *= (
                        self.conj_taylor_l[v_small]
                        if self.quad_taylor_l is not None
                        else self.conj_dom_l[v_small]
                    )
                    buf[v_small] /= self.measure[v_small]
                    buf[v_small] -= self.conj_f_l[v_small]
                    if self.quad_taylor_l is not None:
                        aux = v[v_small] - self._taylor_l[v_small]
                        aux *= aux
                        aux *= 0.5
                        aux *= self.fpp_l[v_small]
                        buf[v_small] += aux

            if self._taylor_u is not None:
                v_large = v > self._taylor_u
                if np.any(v_large):
                    mask = ~v_large if mask is None else mask & ~v_large
                    buf[v_large] = v[v_large]
                    buf[v_large] *= (
                        self.conj_taylor_u[v_large]
                        if self.quad_taylor_u is not None
                        else self.conj_dom_u[v_large]
                    )
                    buf[v_large] /= self.measure[v_large]
                    buf[v_large] -= self.conj_f_u[v_large]
                    if self.quad_taylor_u is not None:
                        aux = v[v_large] - self._taylor_u[v_large]
                        aux *= aux
                        aux *= 0.5
                        aux *= self.fpp_u[v_large]
                        buf[v_large] += aux

            self._apply_pointwise(self._f, v, buf, mask=mask, **self.kwargs)

            if self.constr_l is not None:
                buf[v < self.dom_l] = np.inf
            if self.constr_u is not None:
                buf[v > self.dom_u] = np.inf
            if func_vals is not None:
                np.copyto(func_vals, buf)
            buf *= self.measure
            return np.sum(buf)
        finally:
            self._free_scratch('_buf')

    def _conj(self, vstar, func_vals=None):
        # Using the definition of the conjugate, exterior values are given by
        # the corresponding Taylor extensions.
        buf = self._get_scratch('_buf', vstar)
        buf2 = self._get_scratch('_buf2', vstar)
        try:
            np.divide(vstar, self.measure, out=buf2)
            mask = None
            if self.constr_l is not None or self.quad_taylor_l is not None:
                vstar_small = vstar < self.conj_taylor_l
                if np.any(vstar_small):
                    mask = ~vstar_small
                    buf[vstar_small] = buf2[vstar_small]
                    buf[vstar_small] *= (
                        self._taylor_l[vstar_small]
                        if self.quad_taylor_l is not None
                        else self.dom_l[vstar_small]
                    )
                    buf[vstar_small] -= self.f_l[vstar_small]
                    if self.quad_taylor_l is not None:
                        aux = (
                            buf2[vstar_small]
                            - self.conj_taylor_l[vstar_small] / self.measure[vstar_small]
                        )
                        aux *= aux
                        aux *= 0.5
                        aux *= self.fstarpp_l[vstar_small]
                        buf[vstar_small] += aux

            if self.constr_u is not None or self.quad_taylor_u is not None:
                vstar_large = vstar > self.conj_taylor_u
                if np.any(vstar_large):
                    mask = ~vstar_large if mask is None else mask & ~vstar_large
                    buf[vstar_large] = buf2[vstar_large]
                    buf[vstar_large] *= (
                        self._taylor_u[vstar_large]
                        if self.quad_taylor_u is not None
                        else self.dom_u[vstar_large]
                    )
                    buf[vstar_large] -= self.f_u[vstar_large]
                    if self.quad_taylor_u is not None:
                        aux = (
                            buf2[vstar_large]
                            - self.conj_taylor_u[vstar_large] / self.measure[vstar_large]
                        )
                        aux *= aux
                        aux *= 0.5
                        aux *= self.fstarpp_u[vstar_large]
                        buf[vstar_large] += aux

            self._apply_pointwise(self._f_conj, buf2, buf, mask=mask, **self.kwargs)

            if self.lin_taylor_l is not None:
                buf[vstar < self.conj_dom_l] = np.inf
            if self.lin_taylor_u is not None:
                buf[vstar > self.conj_dom_u] = np.inf
            if func_vals is not None:
                np.copyto(func_vals, buf)
            buf *= self.measure
            return np.sum(buf)
        finally:
            self._free_scratch('_buf2')
            self._free_scratch('_buf')

    def _subgradient(self, v, out=None):
        if not self.everywhere_finite:
            self._assert_essential_domain(v, msg="_subgradient")

        out = self._prepare_out(out, v)

        source = v
        source_scratch = None
        if np.may_share_memory(out, v):
            source_scratch = self._get_scratch("_subgradient_input", v)
            np.copyto(source_scratch, v)
            source = source_scratch

        try:
            v_small = (
                source < self._taylor_l
                if self._taylor_l is not None else None
            )
            v_large = (
                source > self._taylor_u
                if self._taylor_u is not None else None
            )

            mask = None

            if v_small is not None and np.any(v_small):
                mask = ~v_small
                if self.quad_taylor_l is not None:
                    aux = source[v_small] - self._taylor_l[v_small]
                    aux *= self.fpp_l[v_small]
                    aux *= self.measure[v_small]
                    out[v_small] = self.conj_taylor_l[v_small]
                    out[v_small] += aux
                else:
                    out[v_small] = self.conj_dom_l[v_small]

            if v_large is not None and np.any(v_large):
                mask = ~v_large if mask is None else mask & ~v_large
                if self.quad_taylor_u is not None:
                    aux = source[v_large] - self._taylor_u[v_large]
                    aux *= self.fpp_u[v_large]
                    aux *= self.measure[v_large]
                    out[v_large] = self.conj_taylor_u[v_large]
                    out[v_large] += aux
                else:
                    out[v_large] = self.conj_dom_u[v_large]

            self._apply_pointwise(
                self._f_deriv, source, out,
                mask=mask, **self.kwargs
            )
            np.multiply(
                out, self.measure, out=out,
                where=self._where(mask, out)
            )
            return out
        finally:
            if source_scratch is not None:
                self._free_scratch("_subgradient_input")
 
    def dist_subdiff(self, vstar, x):
        self._assert_essential_domain(x, msg='dist_subdiff')
        diff = self._ptw_dist_subdiff(vstar, x)
        deriv = np.empty_like(x)

        ind = np.isclose(x, self.dom_u)
        if np.any(ind):
            self._f_deriv(x, out=deriv, mask=ind)
            np.multiply(deriv, self.measure, out=deriv, where=ind)
            np.subtract(vstar, deriv, out=deriv, where=ind)
            np.minimum(deriv, 0., out=diff, where=ind)

        ind = np.isclose(x, self.dom_l)
        if np.any(ind):
            self._f_deriv(x, out=deriv, mask=ind)
            np.multiply(deriv, self.measure, out=deriv, where=ind)
            np.subtract(vstar, deriv, out=deriv, where=ind)
            np.maximum(deriv, 0., out=diff, where=ind)
        return self.h_domain.dual_space().norm(diff)

    def _conj_subgradient(self, vstar, out=None):
        if not self.conj_everywhere_finite:
            self._assert_conj_essential_domain(vstar, msg='_conj_subgradient')
        out = self._prepare_out(out, vstar)
        buf2 = self._get_scratch('_buf2', vstar)
        try:
            np.divide(vstar, self.measure, out=buf2)
            vstar_small = (
                vstar < self.conj_taylor_l
                if self.constr_l is not None or self.quad_taylor_l is not None
                else None
            )
            vstar_large = (
                vstar > self.conj_taylor_u
                if self.constr_u is not None or self.quad_taylor_u is not None
                else None
            )
            mask = None

            if vstar_small is not None and np.any(vstar_small):
                mask = ~vstar_small
                if self.quad_taylor_l is not None:
                    aux = (
                        buf2[vstar_small]
                        - self.conj_taylor_l[vstar_small] / self.measure[vstar_small]
                    )
                    aux *= self.fstarpp_l[vstar_small]
                    out[vstar_small] = self._taylor_l[vstar_small]
                    out[vstar_small] += aux
                else:
                    out[vstar_small] = self.dom_l[vstar_small]

            if vstar_large is not None and np.any(vstar_large):
                mask = ~vstar_large if mask is None else mask & ~vstar_large
                if self.quad_taylor_u is not None:
                    aux = (
                        buf2[vstar_large]
                        - self.conj_taylor_u[vstar_large] / self.measure[vstar_large]
                    )
                    aux *= self.fstarpp_u[vstar_large]
                    out[vstar_large] = self._taylor_u[vstar_large]
                    out[vstar_large] += aux
                else:
                    out[vstar_large] = self.dom_u[vstar_large]

            self._apply_pointwise(
                self._f_conj_deriv, buf2, out, mask=mask, **self.kwargs
            )
            return out
        finally:
            self._free_scratch('_buf2')

    def _conj_dist_subdiff(self, v, xstar):
        self._assert_conj_essential_domain(xstar, msg='_conj_dist_subdiff')
        diff = self._conj_ptw_dist_subdiff(v, xstar)
        argument = np.empty_like(xstar)
        deriv = np.empty_like(xstar)
        np.divide(xstar, self.measure, out=argument)

        ind = np.isclose(xstar, self.conj_dom_u)
        if np.any(ind):
            self._f_conj_deriv(argument, out=deriv, mask=ind)
            np.subtract(v, deriv, out=deriv, where=ind)
            np.minimum(deriv, 0., out=diff, where=ind)

        ind = np.isclose(xstar, self.conj_dom_l)
        if np.any(ind):
            self._f_conj_deriv(argument, out=deriv, mask=ind)
            np.subtract(v, deriv, out=deriv, where=ind)
            np.maximum(deriv, 0., out=diff, where=ind)
        return self.h_domain.norm(diff)

    def _hessian(self, v):
        if not self.everywhere_finite:
            self._assert_essential_domain(v, msg='_hessian')
        buf = self._get_scratch('_buf', v)
        try:
            buf.fill(np.inf)
            if self.lin_taylor_l is not None:
                buf[v < self._taylor_l] = 0.
            if self.quad_taylor_l is not None:
                v_small = v < self._taylor_l
                buf[v_small] = self.fpp_l[v_small]
            if self.lin_taylor_u is not None:
                buf[v > self._taylor_u] = 0.
            if self.quad_taylor_u is not None:
                v_large = v > self._taylor_u
                buf[v_large] = self.fpp_u[v_large]

            mask = None
            if self._taylor_l is not None or self._taylor_u is not None:
                mask = buf == np.inf
            self._apply_pointwise(
                self._f_second_deriv, v, buf, mask=mask, **self.kwargs
            )
            buf *= self.measure
            return rop.PtwMultiplication(self.domain, buf.copy())
        finally:
            self._free_scratch('_buf')

    def _conj_hessian(self, vstar):
        if not self.conj_everywhere_finite:
            self._assert_conj_essential_domain(vstar, msg='_conj_hessian')
        buf = self._get_scratch('_buf', vstar)
        buf2 = self._get_scratch('_buf2', vstar)
        try:
            np.divide(vstar, self.measure, out=buf2)
            buf.fill(np.inf)
            if self.constr_l is not None:
                buf[buf2 < self.conj_taylor_l] = 0.
            if self.constr_u is not None:
                buf[buf2 > self.conj_taylor_u] = 0.
            if self.quad_taylor_l is not None:
                vstar_small = vstar < self.conj_taylor_l
                buf[vstar_small] = self.fstarpp_l[vstar_small]
            if self.quad_taylor_u is not None:
                vstar_large = vstar > self.conj_taylor_u
                buf[vstar_large] = self.fstarpp_u[vstar_large]

            mask = None
            if (
                self.constr_l is not None or self.constr_u is not None
                or self.quad_taylor_l is not None or self.quad_taylor_u is not None
            ):
                mask = buf == np.inf
            self._apply_pointwise(
                self._f_conj_second_deriv, buf2, buf,
                mask=mask, **self.kwargs
            )
            buf /= self.measure
            return rop.PtwMultiplication(self.domain, buf.copy())
        finally:
            self._free_scratch('_buf2')
            self._free_scratch('_buf')
    
    def _proximal(self, v, tau, mask=None, out=None):
        out = self._prepare_out(out, v)
        where = self._where(mask, v)
        needs_source = any(
            item is not None for item in (
                self.lin_taylor_l, self.lin_taylor_u,
                self.quad_taylor_l, self.quad_taylor_u,
            )
        )
        source = v
        source_scratch = None
        if needs_source and np.may_share_memory(out, v):
            source_scratch = self._get_scratch('_prox_input', v)
            np.copyto(source_scratch, v, where=where)
            source = source_scratch

        candidate = aux = None
        if needs_source:
            candidate = self._get_scratch('_prox_candidate', v)
            aux = self._get_scratch('_prox_aux', v)
        try:
            self._f_prox(v, tau, out=out, mask=mask, **self.kwargs)

            if self.constr_l is not None:
                np.maximum(out, self.dom_l, out=out, where=where)
            if self.constr_u is not None:
                np.minimum(out, self.dom_u, out=out, where=where)

            if self.lin_taylor_l is not None:
                np.multiply(tau, self.conj_dom_l, out=aux, where=where)
                np.divide(aux, self.measure, out=aux, where=where)
                np.subtract(source, aux, out=candidate, where=where)
                active = where & (candidate <= self._taylor_l)
                np.copyto(out, candidate, where=active)
            if self.lin_taylor_u is not None:
                np.multiply(tau, self.conj_dom_u, out=aux, where=where)
                np.divide(aux, self.measure, out=aux, where=where)
                np.subtract(source, aux, out=candidate, where=where)
                active = where & (candidate >= self._taylor_u)
                np.copyto(out, candidate, where=active)
            if self.quad_taylor_l is not None:
                np.multiply(tau, self.fpp_l, out=aux, where=where)
                np.multiply(aux, self._taylor_l, out=candidate, where=where)
                np.add(source, candidate, out=candidate, where=where)
                np.multiply(tau, self.conj_taylor_l, out=aux, where=where)
                np.divide(aux, self.measure, out=aux, where=where)
                np.subtract(candidate, aux, out=candidate, where=where)
                np.multiply(tau, self.fpp_l, out=aux, where=where)
                np.add(aux, 1., out=aux, where=where)
                np.divide(candidate, aux, out=candidate, where=where)
                active = where & (candidate <= self._taylor_l)
                np.copyto(out, candidate, where=active)
            if self.quad_taylor_u is not None:
                np.multiply(tau, self.fpp_u, out=aux, where=where)
                np.multiply(aux, self._taylor_u, out=candidate, where=where)
                np.add(source, candidate, out=candidate, where=where)
                np.multiply(tau, self.conj_taylor_u, out=aux, where=where)
                np.divide(aux, self.measure, out=aux, where=where)
                np.subtract(candidate, aux, out=candidate, where=where)
                np.multiply(tau, self.fpp_u, out=aux, where=where)
                np.add(aux, 1., out=aux, where=where)
                np.divide(candidate, aux, out=candidate, where=where)
                active = where & (candidate >= self._taylor_u)
                np.copyto(out, candidate, where=active)
            return out
        finally:
            if aux is not None:
                self._free_scratch('_prox_aux')
                self._free_scratch('_prox_candidate')
            if source_scratch is not None:
                self._free_scratch('_prox_input')
    
    def _conj_proximal(self, vstar, tau, mask=None, out=None):
        out = self._prepare_out(out, vstar)
        where = self._where(mask, vstar)
        needs_source = any(
            item is not None for item in (
                self.lin_taylor_l, self.lin_taylor_u,
                self.constr_l, self.constr_u,
                self.quad_taylor_l, self.quad_taylor_u,
            )
        )
        source = vstar
        source_scratch = None
        if needs_source and np.may_share_memory(out, vstar):
            source_scratch = self._get_scratch('_prox_input', vstar)
            np.copyto(source_scratch, vstar, where=where)
            source = source_scratch

        candidate = aux = None
        if needs_source:
            candidate = self._get_scratch('_prox_candidate', vstar)
            aux = self._get_scratch('_prox_aux', vstar)
        try:
            np.divide(vstar, self.measure, out=out, where=where)
            self._f_conj_prox(out, tau, out=out, mask=mask, **self.kwargs)
            np.multiply(out, self.measure, out=out, where=where)

            if self.lin_taylor_l is not None:
                np.maximum(out, self.conj_dom_l, out=out, where=where)
            if self.lin_taylor_u is not None:
                np.minimum(out, self.conj_dom_u, out=out, where=where)

            if self.constr_l is not None:
                np.multiply(tau, self.dom_l, out=aux, where=where)
                np.multiply(aux, self.measure, out=aux, where=where)
                np.subtract(source, aux, out=candidate, where=where)
                active = where & (candidate <= self.conj_taylor_l)
                np.copyto(out, candidate, where=active)
            if self.constr_u is not None:
                np.multiply(tau, self.dom_u, out=aux, where=where)
                np.multiply(aux, self.measure, out=aux, where=where)
                np.subtract(source, aux, out=candidate, where=where)
                active = where & (candidate >= self.conj_taylor_u)
                np.copyto(out, candidate, where=active)
            if self.quad_taylor_l is not None:
                np.multiply(tau, self.fstarpp_l, out=aux, where=where)
                np.multiply(aux, self.conj_taylor_l, out=candidate, where=where)
                np.add(source, candidate, out=candidate, where=where)
                np.multiply(tau, self._taylor_l, out=aux, where=where)
                np.multiply(aux, self.measure, out=aux, where=where)
                np.subtract(candidate, aux, out=candidate, where=where)
                np.multiply(tau, self.fstarpp_l, out=aux, where=where)
                np.add(aux, 1., out=aux, where=where)
                np.divide(candidate, aux, out=candidate, where=where)
                active = where & (candidate <= self.conj_taylor_l)
                np.copyto(out, candidate, where=active)
            if self.quad_taylor_u is not None:
                np.multiply(tau, self.fstarpp_u, out=aux, where=where)
                np.multiply(aux, self.conj_taylor_u, out=candidate, where=where)
                np.add(source, candidate, out=candidate, where=where)
                np.multiply(tau, self._taylor_u, out=aux, where=where)
                np.multiply(aux, self.measure, out=aux, where=where)
                np.subtract(candidate, aux, out=candidate, where=where)
                np.multiply(tau, self.fstarpp_u, out=aux, where=where)
                np.add(aux, 1., out=aux, where=where)
                np.divide(candidate, aux, out=candidate, where=where)
                active = where & (candidate >= self.conj_taylor_u)
                np.copyto(out, candidate, where=active)
            return out
        finally:
            if aux is not None:
                self._free_scratch('_prox_aux')
                self._free_scratch('_prox_candidate')
            if source_scratch is not None:
                self._free_scratch('_prox_input')

    def _f(self, v, out=None, mask=None, **kwargs):
        raise NotImplementedError
    
    def _f_deriv(self, v, out=None, mask=None, **kwargs):
        raise NotImplementedError

    def _ptw_dist_subdiff(self,vstar, w, **kwargs):
        return vstar-self.subgradient(w, **kwargs)

    def _f_second_deriv(self, v, out=None, mask=None, **kwargs):
        raise NotImplementedError

    def _f_prox(self, v, tau, out=None, mask=None, tol=1e-12, abstol=1e-12,
                maxNewtonIter=15, maxBisecIter=300, maxBoundsIter=100,
                **kwargs):
        if self.__class__.__dict__.get("_f_deriv") is not None:
            f_second_deriv = self._f_second_deriv if self._f_second_deriv is not None else None
            return self._numerical_prox(
                v, tau, self._f_deriv, f_second_deriv, self.dom_l, self.dom_u,
                out=out, mask=mask, tol=tol, abstol=abstol,
                maxNewtonIter=maxNewtonIter, maxBisecIter=maxBisecIter,
                maxBoundsIter=maxBoundsIter, **kwargs
            )
        raise NotImplementedError('Need first derivative for numerical prox operator')
    
    def _f_conj(self, vstar, out=None, mask=None, **kwargs):
        raise NotImplementedError
    
    def _f_conj_deriv(self, vstar, out=None, mask=None, **kwargs):
        raise NotImplementedError

    def _conj_ptw_dist_subdiff(self, v, wstar, **kwargs):
        return v-self.conj.subgradient(wstar, **kwargs)

    def _f_conj_second_deriv(self, vstar, out=None, mask=None, **kwargs):
        raise NotImplementedError

    def _f_conj_prox(self, vstar, tau, out=None, mask=None,
                     tol=1e-12, abstol=1e-12,
                     maxNewtonIter=15, maxBisecIter=300, maxBoundsIter=100,
                     **kwargs):
        if self.__class__.__dict__.get("_f_conj_deriv") is not None:
            f_conj_second_deriv = (
                self._f_conj_second_deriv
                if self._f_conj_second_deriv is not None else None
            )
            return self._numerical_prox(
                vstar, tau, self._f_conj_deriv, f_conj_second_deriv,
                self.conj_dom_l / self.measure,
                self.conj_dom_u / self.measure,
                out=out, mask=mask, tol=tol, abstol=abstol,
                maxNewtonIter=maxNewtonIter, maxBisecIter=maxBisecIter,
                maxBoundsIter=maxBoundsIter, **kwargs
            )
        raise NotImplementedError(
            'Need first derivative of conjugate functional for numerical conjugate prox operator'
        )

    def _numerical_prox(self, y, tau, fp, fpp, dom_l, dom_u,
                        start=None, ub=None, lb=None, out=None, mask=None,
                        tol=1e-15, abstol=1e-15,
                        maxNewtonIter=15, maxBisecIter=300,
                        maxBoundsIter=100, **kwargs):
        """Vectorized proximal operator of a smooth convex scalar function
        implemented by Newton's method with automatic fallback to bisection.

        Parameters
        ----------
        y : np.ndarray
            Input array.
        tau : float
            Prox parameter (>0).
        f, fp, fpp : callables
            f, f', f'' (accept NumPy arrays).
        dom_l, dom_u : floats
            Domain bounds (can be -np.inf, np.inf).
        start: np.ndarray or None
            starting point
        out: np.ndarray or None
            If not None, this is used as the iterate and is returned
        mask: np.ndarray[bool] or None
            If not None, this specifies the subset of entries to which 
            the Newton-bisection method is applied 
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
        out : ndarray
            Proximal points, same shape as y.
        """
    
        if not isinstance(y, np.ndarray):
            raise TypeError(Errors.not_instance(
                y, np.ndarray,
                add_info="The numerical prox in IntegralFunctionals can only be computed for y an ndarray"
            ))
        if tau <= 0:
            raise ValueError(Errors.value_error('tau must be positive.', self, '_numerical_prox'))

        where = self._where(mask, y)
        out = self._prepare_out(out, y)
        y_input = y.copy() if np.may_share_memory(out, y) else y

        if np.isscalar(dom_u):
            dom_u = np.broadcast_to(dom_u, y.shape)
        if np.isscalar(dom_l):
            dom_l = np.broadcast_to(dom_l, y.shape)
        if np.any(where & (dom_l > dom_u)):
            raise ValueError(Errors.value_error(
                'Upper/lower bounds of essential domain invalid. dom_l not less or equal to dom_u!',
                self, '_numerical_prox'
            ))

        np.copyto(out, y_input if start is None else start, where=where)
        np.maximum(dom_l, out, out=out, where=where)
        np.minimum(out, dom_u, out=out, where=where)
        x = out

        r = np.empty_like(y)
        tmp = np.empty_like(y)
        candidate = np.empty_like(y)

        def residual(z, result, active):
            fp(z, out=result, mask=active, **kwargs)
            np.subtract(z, y_input, out=tmp, where=active)
            np.divide(tmp, tau, out=tmp, where=active)
            np.add(result, tmp, out=result, where=active)
            return result

        residual(x, r, where)
        width = dom_u - dom_l
        scale = np.maximum(dom_u, -dom_l)
        converged = ~where | (
            (width < abstol)
            | (width < tol * scale)
            | (r == 0)
            | ((x == dom_l) & (r >= 0))
            | ((x == dom_u) & (r <= 0))
        )

        self.log.debug(f'initially {np.sum(converged & where)} converged.')
        ub = x.copy() if ub is None else np.array(ub, copy=True)
        lb = x.copy() if lb is None else np.array(lb, copy=True)

        x_too_small = where & (r < 0) & ~converged
        np.multiply(r, -tau, out=candidate, where=x_too_small)
        np.minimum(candidate, tau, out=candidate, where=x_too_small)
        np.add(x, candidate, out=candidate, where=x_too_small)
        np.minimum(candidate, dom_u, out=candidate, where=x_too_small)
        np.copyto(ub, candidate, where=x_too_small)

        x_too_large = where & (r > 0) & ~converged
        np.multiply(r, tau, out=candidate, where=x_too_large)
        np.minimum(candidate, tau, out=candidate, where=x_too_large)
        np.subtract(x, candidate, out=candidate, where=x_too_large)
        np.maximum(candidate, dom_l, out=candidate, where=x_too_large)
        np.copyto(lb, candidate, where=x_too_large)

        bad = where & (lb > ub)
        if np.any(bad):
            raise RuntimeError(Errors.runtime_error(
                f'lb<=ub violated for input values {y_input[bad]}. '
                f'lb: {lb[bad]}, ub: {ub[bad]}', self, '_numerical_prox'
            ))

        ind = np.zeros_like(where)
        for it in range(maxBoundsIter):
            active = where & ~converged
            residual(lb, r, active)
            ind = active & (r > 0) & (lb > dom_l)
            if it == 0:
                width = ub - lb
                scale = np.maximum(ub, -lb)
                ii = ind & (width <= tol * scale)
                np.abs(lb, out=candidate, where=ii)
                np.multiply(candidate, tol, out=candidate, where=ii)
                np.add(candidate, abstol, out=candidate, where=ii)
                np.subtract(lb, candidate, out=candidate, where=ii)
                np.maximum(candidate, dom_l, out=candidate, where=ii)
                np.copyto(lb, candidate, where=ii)
                converged |= ii & (lb == dom_l)
                ind &= ~converged
            if not np.any(ind):
                break
            np.multiply(lb, 2., out=candidate, where=ind)
            np.subtract(candidate, ub, out=candidate, where=ind)
            np.maximum(candidate, dom_l, out=candidate, where=ind)
            np.copyto(lb, candidate, where=ind)
        if np.any(ind):
            raise RuntimeError(Errors.runtime_error(
                'Could not determine lower bound. Increase maxBoundsIter!',
                self, '_numerical_prox'
            ))

        active = where & ~converged
        residual(lb, r, active)
        converged |= where & (lb == dom_l) & (r >= 0)
        violation = where & ~converged & (r > 0)
        if np.any(violation):
            raise RuntimeError(Errors.runtime_error(
                f'lower bound not satisfied for indices {np.where(violation)}',
                self, '_numerical_prox'
            ))

        for it in range(maxBoundsIter):
            active = where & ~converged
            residual(ub, r, active)
            ind = active & (r < 0) & (ub < dom_u)
            if it == 0:
                width = ub - lb
                scale = np.maximum(ub, -lb)
                ii = ind & (width <= tol * scale)
                np.abs(ub, out=candidate, where=ii)
                np.multiply(candidate, tol, out=candidate, where=ii)
                np.add(candidate, abstol, out=candidate, where=ii)
                np.add(ub, candidate, out=candidate, where=ii)
                np.minimum(candidate, dom_u, out=candidate, where=ii)
                np.copyto(ub, candidate, where=ii)
                converged |= ii & (ub == dom_u)
                ind &= ~converged
            if not np.any(ind):
                break
            np.multiply(ub, 2., out=candidate, where=ind)
            np.subtract(candidate, lb, out=candidate, where=ind)
            np.minimum(candidate, dom_u, out=candidate, where=ind)
            np.copyto(ub, candidate, where=ind)
        if np.any(ind):
            raise RuntimeError(Errors.runtime_error(
                'Could not determine upper bound. Increase maxBoundsIter!',
                self, '_numerical_prox'
            ))

        active = where & ~converged
        residual(ub, r, active)
        converged |= where & (ub == dom_u) & (r <= 0)
        violation = where & ~converged & (r < 0)
        if np.any(violation):
            raise RuntimeError(Errors.runtime_error(
                f'upper bound not satisfied for indices {np.where(violation)}',
                self, '_numerical_prox'
            ))

        if fpp is not None:
            d = np.empty_like(y)
            xnew = np.empty_like(y)
            for iteration in range(maxNewtonIter):
                active = where & ~converged
                if not np.any(active):
                    return x
                residual(x, r, active)
                fpp(x, out=d, mask=active, **kwargs)
                np.add(d, 1. / tau, out=d, where=active)
                np.divide(r, d, out=tmp, where=active)
                np.subtract(x, tmp, out=xnew, where=active)
                np.maximum(xnew, lb, out=xnew, where=active)
                np.minimum(xnew, ub, out=xnew, where=active)

                nonpositive = active & (r <= 0)
                np.copyto(lb, x, where=nonpositive)
                positive = active & (r > 0)
                np.multiply(r, tau, out=candidate, where=positive)
                np.subtract(x, candidate, out=candidate, where=positive)
                np.maximum(candidate, lb, out=candidate, where=positive)
                np.copyto(lb, candidate, where=positive)

                nonnegative = active & (r >= 0)
                np.copyto(ub, x, where=nonnegative)
                negative = active & (r < 0)
                np.multiply(r, tau, out=candidate, where=negative)
                np.subtract(x, candidate, out=candidate, where=negative)
                np.minimum(candidate, ub, out=candidate, where=negative)
                np.copyto(ub, candidate, where=negative)

                width = np.abs(ub - lb)
                scale = np.maximum(ub, -lb)
                conv = active & ((width < abstol) | (width <= tol * scale))
                converged |= conv
                np.copyto(x, xnew, where=active & ~conv)
                self.log.debug(
                    f'Newton it {iteration}: {np.sum(converged & where)} '
                    f'out of {np.sum(where)} converged.'
                )
                if np.all(converged):
                    return x

        bisect_mask = where & ~converged
        active = bisect_mask.copy()
        not_converged = active.copy()
        mid = np.empty_like(y)
        fmid = np.empty_like(y)
        if np.any(active):
            for iteration in range(maxBisecIter):
                np.add(lb, ub, out=mid, where=active)
                np.multiply(mid, 0.5, out=mid, where=active)
                residual(mid, fmid, active)
                right = active & (fmid > 0)
                np.copyto(ub, mid, where=right)
                np.copyto(lb, mid, where=active & ~right)
                width = ub - lb
                scale = np.maximum(ub, -lb)
                not_converged = active & (
                    (width >= abstol) & (width >= tol * scale)
                )
                if not np.any(not_converged):
                    break
                active = not_converged
            np.add(lb, ub, out=x, where=bisect_mask)
            np.multiply(x, 0.5, out=x, where=bisect_mask)
            if np.any(not_converged):
                raise RuntimeError(
                    f'Could not satisfy either tolerance criterion after '
                    f'{iteration + 1} bisection steps.'
                )
        return x
    
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

        super().__init__(domain, 
                         is_quadratic = (p==2.),
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                         quad_taylor_l=quad_taylor_l, quad_taylor_u=quad_taylor_u,
                         **kwargs
                         )

    @property
    def is_norm_squared_type(self):
        return self.p==2 and all([x is None for x in (self.constr_l, self.lin_taylor_l, self.quad_taylor_l,
                                                      self.constr_u, self.lin_taylor_u, self.quad_taylor_u)])

    @property
    def Lipschitz(self):
        taylor_l = -np.inf if self.lin_taylor_l is None else self.lin_taylor_l
        if self.quad_taylor_l is not None:
            taylor_l = self.quad_taylor_l
        taylor_u = np.inf if self.lin_taylor_u is None else self.lin_taylor_u
        if self.quad_taylor_u is not None:
            taylor_u = self.quad_taylor_u

        if self.p<2:
            aux = np.min(np.maximum(taylor_l,-taylor_u))
            Lip= (self.p-1) * np.abs(aux)**(self.p-2) if aux>0 else np.inf
        if self.p>2:
            aux =  np.abs(np.max(np.maximum(-taylor_l,taylor_u)))
            Lip = (self.p-1) * aux**(self.p-2) if aux<np.inf else np.inf
        if self.p==2:
            Lip = 1. if self.constr_l is None and self.constr_u is None else np.inf
        return np.max([Lip,self._Lipschitz_exterior])

    @property
    def convexity_param(self):
        dom2_l, dom2_u = self.dom_l, self.dom_u
        taylor_l = -np.inf if self.lin_taylor_l is None else self.lin_taylor_l
        if self.quad_taylor_l is not None:
            taylor_l = self.quad_taylor_l
            dom2_l= taylor_l
        taylor_u = np.inf if self.lin_taylor_u is None else self.lin_taylor_u
        if self.quad_taylor_u is not None:
            taylor_u = self.quad_taylor_u
            dom2_u= taylor_u

        if self.p<2:
            aux = np.abs(np.max(np.maximum(-dom2_l,dom2_u)))
            conv = (self.p-1) * aux**(self.p-2) if aux<np.inf else 0
        if self.p>2:
            aux = np.min(np.maximum(dom2_l, -dom2_u))
            conv = (self.p-1) * aux**(self.p-2)  if aux>0 else 0
        if self.p==2:
            conv = 1. if self.lin_taylor_l is None and self.lin_taylor_u is None else 0.
        return np.min([conv,self._convexity_param_exterior])

    def _f(self, v, out=None, mask=None, **kwargs):
        # out = np.abs(v)**self.p/self.p
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.abs(v, out=out, where=where)
        np.power(out, self.p, out=out, where=where)
        np.divide(out, self.p, out=out, where=where)
        return out
    
    def _f_deriv(self, v, out=None, mask=None, **kwargs):
        # out = np.abs(v)**(self.p-1)*np.sign(v)
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        aux = self._get_scratch('_aux', v)
        try:
            np.sign(v, out=aux, where=where)
            np.abs(v, out=out, where=where)
            np.power(out, self.p - 1, out=out, where=where)
            np.multiply(out, aux, out=out, where=where)
        finally:
            self._free_scratch('_aux')
        return out
    
    def _f_second_deriv(self, v, out=None, mask=None, **kwargs):
        # out = (self.p-1)*np.abs(v)**(self.p-2)
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.abs(v, out=out, where=where)
        np.power(out, self.p - 2, out=out, where=where)
        np.multiply(out, self.p - 1, out=out, where=where)
        return out
    
    def _f_prox(self, v, tau, out=None, mask=None, **kwargs):
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        if self.p == 2:
            np.divide(v, 1 + tau, out=out, where=where)
            return out
        return self._numerical_prox(
            v, tau, self._f_deriv, self._f_second_deriv,
            -np.inf, np.inf, out=out, mask=mask,
            tol=1e-12, maxNewtonIter=10, maxBisecIter=300,
            maxBoundsIter=300, **kwargs
        )
    
    def _f_conj(self, vstar, out=None, mask=None, **kwargs):
        # out = np.abs(vstar)**self.q/self.q
        where = self._where(mask, vstar)
        out = self._prepare_out(out, vstar)
        np.abs(vstar, out=out, where=where)
        np.power(out, self.q, out=out, where=where)
        np.divide(out, self.q, out=out, where=where)
        return out

    def _f_conj_deriv(self, vstar, out=None, mask=None, **kwargs):
        # out = np.abs(vstar)**(self.q-1)*np.sign(vstar)
        where = self._where(mask, vstar)
        out = self._prepare_out(out, vstar)
        aux = self._get_scratch('_aux', vstar)
        try:
            np.sign(vstar, out=aux, where=where)
            np.abs(vstar, out=out, where=where)
            np.power(out, self.q - 1, out=out, where=where)
            np.multiply(out, aux, out=out, where=where)
        finally:
            self._free_scratch('_aux')
        return out
    
    def _f_conj_second_deriv(self, vstar, out=None, mask=None, **kwargs):
        # out = (self.q-1)*np.abs(vstar)**(self.q-2)
        where = self._where(mask, vstar)
        out = self._prepare_out(out, vstar)
        np.abs(vstar, out=out, where=where)
        np.power(out, self.q - 2, out=out, where=where)
        np.multiply(out, self.q - 1, out=out, where=where)
        return out
    
    def _f_conj_prox(self, v_star, tau, out=None, mask=None, **kwargs):
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        if self.p == 2:
            np.divide(v_star, 1 + tau, out=out, where=where)
            return out
        return self._numerical_prox(
            v_star, tau, self._f_conj_deriv, self._f_conj_second_deriv,
            -np.inf, np.inf, out=out, mask=mask,
            tol=1e-12, maxNewtonIter=10, maxBisecIter=300,
            maxBoundsIter=300, **kwargs
        )

class Quadratic(IntegralFunctionalBase):
    r"""Quadratic functional :math:`v\mapsto \int\frac{a}{2}v(x)^2+b(x)v(x)+c(x)dx` on some domain in `regpy.vecsps.MeasureSpaceFcts` as an integral functional.

    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a `regpy.vecsps.MeasureSpaceFcts`
    a,b,c: scalar or domain element
        Coefficients of the quadratic functional. Defaults are 1., 0., and 0., respectively. If scalar, they are broadcasted to the shape of the domain.         
    constr_l,constr_u,lin_taylor_l, lin_taylor_u: None, np.isscalar or np.ndarray
        see IntegralFunctional  
    """
    def __init__(self, domain,
                 a=1., b=0., c=0.,
                 constr_l=None, constr_u=None, lin_taylor_l=None, lin_taylor_u=None,
                 quad_taylor_l=None, quad_taylor_u=None,
                 **kwargs):
        super().__init__(domain, is_quadratic=True,
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                         quad_taylor_l=quad_taylor_l, quad_taylor_u=quad_taylor_u,
                         **kwargs
                         )
        if not np.isscalar(a) and a not in domain:
            raise ValueError(Errors.value_error(f'a needs to be either scalar or an element of the domain. Given a: {a} of type {type(a)}', self))
        self._a = np.broadcast_to(a, domain.shape) if np.isscalar(a) else a
        if not np.isscalar(b) and b not in domain:  
            raise ValueError(Errors.value_error(f'b needs to be either scalar or an element of the domain. Given b: {b} of type {type(b)}', self))
        self._b = np.broadcast_to(b, domain.shape) if np.isscalar(b) else b
        if not np.isscalar(c) and c not in domain:  
            raise ValueError(Errors.value_error(f'c needs to be either scalar or an element of the domain. Given c: {c} of type {type(c)}', self))
        self._c = np.broadcast_to(c, domain.shape) if np.isscalar(c) else c

    @property
    def a(self):
        return self._a
    
    @a.setter
    def a(self, value):
        if not np.isscalar(value) and value not in self.domain:
            raise ValueError(Errors.value_error(f'a needs to be either scalar or an element of the domain. Given a: {value} of type {type(value)}', self))
        self._a = np.broadcast_to(value, self.domain.shape) if np.isscalar(value) else value    
        self._precompute_f_l(); self._precompute_f_u()
        
    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value): 
        if not np.isscalar(value) and value not in self.domain:
            raise ValueError(Errors.value_error(f'b needs to be either scalar or an element of the domain. Given b: {value} of type {type(value)}', self))
        self._b = np.broadcast_to(value, self.domain.shape) if np.isscalar(value) else value    
        self._precompute_f_l(); self._precompute_f_u()

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        if not np.isscalar(value) and value not in self.domain:
            raise ValueError(Errors.value_error(f'c needs to be either scalar or an element of the domain. Given c: {value} of type {type(value)}', self))
        self._c = np.broadcast_to(value, self.domain.shape) if np.isscalar(value) else value    
        self._precompute_f_l(); self._precompute_f_u()

    @property
    def Lipschitz(self):
        return max([self._Lipschitz_exterior,np.max(np.abs(self._a))]) 
    
    @property
    def convexity_param(self):
        return min([self._convexity_param_exterior,np.min(self._a)])
    
    @property
    def is_convex(self):
        return self.convexity_param>=0
    
    @property
    def conj_is_quadratic(self):
        return (np.min(self.a)>0)
     
    @property
    def is_norm_squared_type(self):
        a0 = self.a.flat[0]
        return (a0>0 and np.allclose(self.a, a0) and
                all([x is None for x in (self.constr_l, self.lin_taylor_l, self.quad_taylor_l,
                                         self.constr_u, self.lin_taylor_u, self.quad_taylor_u)])             
               )
   
    def _f(self, v, out=None, mask=None, **kwargs):
        # out = 0.5*self._a*v**2 + self._b*v + self._c
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        source = v
        scratch = None
        if np.may_share_memory(out, v):
            scratch = self._get_scratch('_aux', v)
            np.copyto(scratch, v, where=where)
            source = scratch
        try:
            np.multiply(self._a, source, out=out, where=where)
            np.multiply(out, 0.5, out=out, where=where)
            np.add(out, self._b, out=out, where=where)
            np.multiply(out, source, out=out, where=where)
            np.add(out, self._c, out=out, where=where)
        finally:
            if scratch is not None:
                self._free_scratch('_aux')
        return out
    
    def _f_deriv(self, v, out=None, mask=None, **kwargs):
        # out = a*v +b
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.multiply(self._a, v, out=out, where=where)
        np.add(out, self._b, out=out, where=where)
        return out
    
    def _f_second_deriv(self, v, out=None, mask=None, **kwargs):
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.copyto(out, self._a, where=where)
        return out
    
    def _f_prox(self, v, tau, out=None, mask=None, **kwargs):
        # out = (v - tau*b) / (1. + tau*a)
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        aux = self._get_scratch('_aux', v)
        try:
            np.multiply(tau, self._b, out=aux, where=where)
            np.subtract(v, aux, out=out, where=where)
            np.multiply(tau, self._a, out=aux, where=where)
            np.add(aux, 1., out=aux, where=where)
            np.divide(out, aux, out=out, where=where)
        finally:
            self._free_scratch('_aux')
        return out
    
    def _f_conj(self, v_star, out=None, mask=None, **kwargs):
        # out = 0.5*(v_star - b)**2 / a - c
        if np.min(self.a) <= 0:
            raise ValueError(Errors.value_error(
                f'Conjugate of quadratic functional is only quadratic if a is positive. Given a: {self.a}', self
            ))
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        np.subtract(v_star, self._b, out=out, where=where)
        np.square(out, out=out, where=where)
        np.multiply(out, 0.5, out=out, where=where)
        np.divide(out, self._a, out=out, where=where)
        np.subtract(out, self._c, out=out, where=where)
        return out

    def _f_conj_deriv(self, v_star, out=None, mask=None, **kwargs):
        #  out = (v_star - b) / a
        if np.min(self.a) <= 0:
            raise ValueError(Errors.value_error(
                f'Conjugate of quadratic functional is only quadratic if a is positive. Given a: {self.a}', self
            ))
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        np.subtract(v_star, self._b, out=out, where=where)
        np.divide(out, self._a, out=out, where=where)
        return out
    
    def _f_conj_second_deriv(self, v_star, out=None, mask=None, **kwargs):
        if np.min(self.a) <= 0:
            raise ValueError(Errors.value_error(
                f'Conjugate of quadratic functional is only quadratic if a is positive. Given a: {self.a}', self
            ))
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        np.divide(1., self._a, out=out, where=where)
        return out
        
    def _f_conj_prox(self, v_star, tau, out=None, mask=None, **kwargs):
        # out =  (a* v_star + tau*b) / (a + tau)
        if np.min(self.a) <= 0:
            raise ValueError(Errors.value_error(
                f'Conjugate of quadratic functional is only quadratic if a is positive. Given a: {self.a}', self
            ))
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        aux = self._get_scratch('_aux', v_star)
        try:
            np.multiply(self._a, v_star, out=out, where=where)
            np.multiply(tau, self._b, out=aux, where=where)
            np.add(out, aux, out=out, where=where)
            np.add(self._a, tau, out=aux, where=where)
            np.divide(out, aux, out=out, where=where)
        finally:
            self._free_scratch('_aux')
        return out

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
                quad_taylor_l=None, quad_taylor_u=None, conj_tol = 1e-10, **kwargs):
        self.conj_tol = conj_tol
        super().__init__(domain,conj_nat_dom_u=1.,conj_nat_dom_l=-1.,
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                         quad_taylor_l=quad_taylor_l, quad_taylor_u=quad_taylor_u,
                         **kwargs
                         )

    def _f(self, v, out=None, mask=None, **kwargs):
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.abs(v, out=out, where=where)
        return out

    def _f_deriv(self, v, out=None, mask=None, **kwargs):
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.sign(v, out=out, where=where)
        return out

    def _f_second_deriv(self, v, out=None, mask=None, **kwargs):
        where = self._where(mask, v)
        if np.any(where & (v == 0)):
            raise NotTwiceDifferentiableError('L1')
        out = self._prepare_out(out, v)
        np.copyto(out, 0., where=where)
        return out

    def _f_prox(self, v, tau, out=None, mask=None, **kwargs):
        # out = np.maximum(0, np.abs(v)-tau)*np.sign(v)
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        aux = self._get_scratch('_aux', v)
        try:
            np.sign(v, out=aux, where=where)
            np.abs(v, out=out, where=where)
            np.subtract(out, tau, out=out, where=where)
            np.maximum(0., out, out=out, where=where)
            np.multiply(out, aux, out=out, where=where)
        finally:
            self._free_scratch('_aux')
        return out

    def _f_conj(self, v_star, out=None, mask=None, **kwargs):
        # out = 0 if abs(v_star) <= 1, and np.inf otherwise
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        np.abs(v_star, out=out, where=where)
        outside = where & (out > 1 + self.conj_tol)
        np.copyto(out, 0., where=where)
        np.copyto(out, np.inf, where=outside)
        return out
    
    def _f_conj_deriv(self, v_star, out=None, mask=None, **kwargs):
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        np.copyto(out, 0., where=where)
        return out

    def _f_conj_second_deriv(self, v_star, out=None, mask=None, **kwargs):
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        np.copyto(out, 0., where=where)
        return out

    def _f_conj_prox(self, vstar, tau, out=None, mask=None, **kwargs):
        where = self._where(mask, vstar)
        out = self._prepare_out(out, vstar)
        np.maximum(vstar, -1., out=out, where=where)
        np.minimum(out, 1., out=out, where=where)
        return out

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
    w: domain or scalar or None (default: None) 
        First argument of Kullback-Leibler divergence. In the case of None, w is set to zero. If w is scalar, it is broadcasted to the shape of the domain. 
    constr_l,constr_u,lin_taylor_l, lin_taylor_u: None, np.isscalar or np.ndarray
        see IntegralFunctional
    data: explcitly declared data (default:None) 
        If data is given, then w is w+data; if the parameter w is not given, then w is just the data.
    omit_normalization:bool (default:False)
        If true, the term :math:`w(x)\ln w(x)` is omitted in the definition of F_w. (Then :math:`F_w(w)\neq 0`.)
        This is sometimes done if KullbackLeibler is used as Poisson log-likelihood since the omitted term is not well-defined for 
        a continuous Poisson process, and it is irrelevant and large for large discrete data. 
    """

    def __init__(self, domain, w=None,
                 constr_l=None, constr_u=None, lin_taylor_l=None, lin_taylor_u=None,
                 quad_taylor_l=None, quad_taylor_u=None, data = None,
                 omit_normalization:bool = False,
                 **kwargs):
        if(w is None):
            w=0.
        if w in domain:
            self._w=w.copy()
        elif  isinstance(w, (int,float,np.floating,np.integer)):
            self._w= np.broadcast_to(float(w),domain.shape)
        else:
            raise ValueError(Errors.value_error(f'w not in domain and not scalar. {w} of type {type(w)}.', self))
  
        if(data is not None):
            if(data not in domain):
                raise ValueError(Errors.value_error('data not in domain.'))
            self._data=data
            self.is_data_func = True            
        if np.min(w)<0:
            raise ValueError(Errors.value_error('w must be non-negative.'))
        
        if not isinstance(omit_normalization,bool):
            raise ValueError(Errors.value_error(f"{omit_normalization} not of type bool."))
        self.omit_normalization = omit_normalization

        super().__init__(domain,nat_dom_l=1e-14*w,
                         conj_nat_dom_u=domain.ones()-1e-14*w,
                          constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                          quad_taylor_l=quad_taylor_l,quad_taylor_u=quad_taylor_u,
                          is_data_func = (data is not None),
                         **kwargs
                         )
        

    @property
    def w(self):
        if self.is_data_func:
            return self._w + self.data
        else:
            return self._w
    @w.setter
    def w(self, new_w):
        if new_w in self.domain or (isinstance(new_w, (int,float,np.floating,np.integer))): 
            if new_w in self.domain:
                self._w = new_w.copy()
            else:
                self._w = np.broadcast_to(float(new_w),self.domain.shape)
            if np.min(self._w+ self._data if self.is_data_func else self._w)<0:
                raise ValueError(Errors.value_error('w must be non-negative.'))
            self._precompute_f_l(); self._precompute_f_u()
        else:
            raise ValueError(Errors.not_in_vecsp(new_w,self.domain,vec_name="new w vector",space_name="domain of functional"))

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        if new_data is None:
            self.is_data_func = False
            if self.is_data_func:
                del self._data
                self._precompute_f_l(); self._precompute_f_u()
        elif new_data in self.domain:
            self.is_data_func = True
            self._data = new_data
            if np.min(self._w+ self._data)<0:
                raise ValueError(Errors.value_error('w+data must be non-negative.'))
            self._precompute_f_l(); self._precompute_f_u()
        else:
            raise ValueError(Errors.not_in_vecsp(new_data,self.domain,vec_name="new data vector",space_name="domain of functional"))
        
    @data.deleter
    def data(self):
        if self.is_data_func:
            del self._data
            self.is_data_func = False
            self._precompute_f_l(); self._precompute_f_u()

    def as_data_func(self,data):
        self.data=data
        return self

    @property
    def Lipschitz(self):
        if self.constr_u is not None:
            return np.inf
        elif self._taylor_l is not None:
            return  max([self._Lipschitz_exterior,np.max(self.w/self._taylor_l**2) if np.min(self._taylor_l)>0 else np.inf])
        else: 
            return np.inf

    @property
    def convexity_param(self):
        if self.lin_taylor_l is not None or self.lin_taylor_u is not None:
            return 0. 
        elif self.constr_u is not None or self.quad_taylor_u is not None:
            u = self.constr_u if self.constr_u is not None else self.quad_taylor_u
            return np.min([self._convexity_param_exterior,np.min(self.w/u**2) if np.min(u)>0 else np.inf]) 
        else:
            return 0

    def _f(self, u, out=None, mask=None, **kwargs):
        # out = u - w - w * log(u / w)
        if 'w' in kwargs:
            raise ValueError(Errors.value_error(
                'second parameter w of KullbackLeibler must be fixed in constructor.'
            ))
        where = self._where(mask, u)
        wm = self.w
        out = self._prepare_out(out, u)
        source = u
        scratch = None
        if np.may_share_memory(out, u):
            scratch = self._get_scratch('_aux', u)
            np.copyto(scratch, u, where=where)
            source = scratch
        invalid = where & ((source < 0) | ((source == 0) & (wm > 0)))
        try:
            if self.omit_normalization:
                np.copyto(out, source, where=where)
            else:
                np.divide(source, wm, out=out, where=where)
            with np.errstate(invalid='ignore', divide='ignore'):
                np.log(out, out=out, where=where)
            np.multiply(out, wm, out=out, where=where)
            np.negative(out, out=out, where=where)
            np.add(out, source, out=out, where=where)
            np.subtract(out, wm, out=out, where=where)
            np.copyto(out, np.inf, where=invalid)
        finally:
            if scratch is not None:
                self._free_scratch('_aux')
        return out
   
    def _f_deriv(self, u, out=None, mask=None, **kwargs):
        # out =  1 - w / u
        if 'w' in kwargs:
            raise ValueError(Errors.value_error(
                'first parameter w of KullbackLeibler must be fixed in constructor.'
            ))
        where = self._where(mask, u)
        wm = self.w
        zero = where & (u == 0)
        if np.any(zero & (wm != 0)):
            raise ValueError(Errors.value_error(
                'argument cannot be 0 at positions where w is not 0'
            ))
        out = self._prepare_out(out, u)
        regular = where & ~zero
        np.divide(wm, u, out=out, where=regular)
        np.negative(out, out=out, where=regular)
        np.add(out, 1., out=out, where=regular)
        np.copyto(out, 1., where=zero)
        return out

    def _f_second_deriv(self, u, out=None, mask=None, **kwargs):
        # out = w / u**2
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.square(u, out=out, where=where)
        np.divide(self.w, out, out=out, where=where)
        return out

    def _f_conj(self, u_star, out=None, mask=None, **kwargs):
        # out = - w * log(1-u_star)
        if 'w' in kwargs:
            raise ValueError(Errors.value_error(
                'second parameter w of KullbackLeibler must be fixed in constructor.'
            ))
        where = self._where(mask, u_star)
        wm = self.w
        invalid = where & ((u_star > 1) | ((u_star == 1) & (wm > 0)))
        out = self._prepare_out(out, u_star)
        np.subtract(1., u_star, out=out, where=where)
        with np.errstate(invalid='ignore', divide='ignore'):
            np.log(out, out=out, where=where)
        np.multiply(out, wm, out=out, where=where)
        np.negative(out, out=out, where=where)
        if self.omit_normalization:
            aux = self._get_scratch('_aux', u_star)
            try:
                positive = where & (wm > 0)
                np.log(wm, out=aux, where=positive)
                np.multiply(aux, wm, out=aux, where=positive)
                np.add(out, aux, out=out, where=positive)
            finally:
                self._free_scratch('_aux')
        np.copyto(out, np.inf, where=invalid)
        return out

    def _f_conj_deriv(self, u_star, out=None, mask=None, **kwargs):
        # out = w / (1-u_star)
        if 'w' in kwargs:
            raise ValueError(Errors.value_error(
                'second parameter w of KullbackLeibler must be fixed in constructor.'
            ))
        where = self._where(mask, u_star)
        wm = self.w
        at_one = where & (u_star == 1)
        if np.any(at_one & (wm != 0)):
            raise ValueError(Errors.value_error(
                'argument cannot be 1 at positions where w is not 0.'
            ))
        out = self._prepare_out(out, u_star)
        regular = where & ~at_one
        np.subtract(1., u_star, out=out, where=regular)
        np.divide(wm, out, out=out, where=regular)
        np.copyto(out, 0., where=at_one)
        return out
    
    def _f_conj_second_deriv(self, u_star, out=None, mask=None, **kwargs):
        #  out = w / (1-u_star)**2
        where = self._where(mask, u_star)
        wm = self.w
        at_one = where & (u_star == 1)
        if np.any(at_one & (wm != 0)):
            raise RuntimeError(Errors.runtime_error(
                "Whenever u^ast is 1 the first argument of initial Kullback-Leibler "
                "must be zero for the computation of the second derivative of the "
                "conjugate functional.", self, "_f_conj_second_deriv"
            ))
        out = self._prepare_out(out, u_star)
        regular = where & ~at_one
        np.subtract(1., u_star, out=out, where=regular)
        np.square(out, out=out, where=regular)
        np.divide(wm, out, out=out, where=regular)
        np.copyto(out, 0., where=at_one)
        return out

    def _f_prox(self, v, tau, out=None, mask=None, **kwargs):
        # out = -0.5 * (tau - v) + sqrt(0.25 * (tau - v)**2 + tau * w)
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        aux = self._get_scratch('_aux', v)
        try:
            np.subtract(tau, v, out=aux, where=where)
            np.square(aux, out=out, where=where)
            np.multiply(out, 0.25, out=out, where=where)
            np.add(out, tau * self.w, out=out, where=where)
            np.sqrt(out, out=out, where=where)
            np.multiply(aux, 0.5, out=aux, where=where)
            np.subtract(out, aux, out=out, where=where)
        finally:
            self._free_scratch('_aux')
        return out

    def _f_conj_prox(self, vstar, tau, out=None, mask=None, **kwargs):
        # out =  0.5 * (1 + vstar) - sqrt(0.25 * (1 + vstar)**2 + tau * w - vstar)
        where = self._where(mask, vstar)
        out = self._prepare_out(out, vstar)
        aux = self._get_scratch('_aux', vstar)
        try:
            np.add(1., vstar, out=aux, where=where)
            np.square(aux, out=out, where=where)
            np.multiply(out, 0.25, out=out, where=where)
            np.add(out, tau * self.w, out=out, where=where)
            np.subtract(out, aux, out=out, where=where)
            np.add(out, 1., out=out, where=where)
            np.sqrt(out, out=out, where=where)
            np.negative(out, out=out, where=where)
            np.multiply(aux, 0.5, out=aux, where=where)
            np.add(out, aux, out=out, where=where)
        finally:
            self._free_scratch('_aux')
        return out


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
            raise ValueError(Errors.value_error("The second argument in the RelativeEntropy needs to be positive!", self))
        self._w= w.copy()

        super().__init__(domain,nat_dom_l = 1e-14*w,
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                         quad_taylor_l=quad_taylor_l,quad_taylor_u=quad_taylor_u,
                         **kwargs
                         )

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, new_w):
        if new_w in self.domain or (isinstance(new_w, (int,float,np.floating,np.integer))): 
            if new_w in self.domain:
                self._w = new_w.copy()
            else:
                self._w = np.broadcast_to(float(new_w),self.domain.shape)
            if np.min(self._w)<0:
                raise ValueError(Errors.value_error('w must be non-negative.'))
            self._precompute_f_l(); self._precompute_f_u()
        else:
            raise ValueError(Errors.not_in_vecsp(new_w,self.domain,vec_name="new w vector",space_name="domain of functional"))

    @property
    def Lipschitz(self):
        if self.constr_u is not None:
            return np.inf
        elif self._taylor_l is not None:
            return np.max([self._Lipschitz_exterior,np.max(1/self._taylor_l) if np.min(self._taylor_l)>0 else np.inf])
        else: 
            return np.inf

    @property
    def convexity_param(self):
        if self.lin_taylor_u is not None or self.lin_taylor_l is not None:
            return 0.
        elif self.constr_u is not None or self.quad_taylor_u is not None:
            u = self.constr_u if self.constr_u is not None else self.quad_taylor_u
            return np.min([self._convexity_param_exterior,np.min(1./u) if np.min(u)>0 else np.inf]) 
        else:
            return 0.

    def _f(self, u, out=None, mask=None, **kwargs):
        # out = u * log(u / w)
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        source = u
        scratch = None
        if np.may_share_memory(out, u):
            scratch = self._get_scratch('_aux', u)
            np.copyto(scratch, u, where=where)
            source = scratch
        negative = where & (source < 0)
        try:
            np.divide(source, self.w, out=out, where=where)
            positive = where & (out > 0)
            np.log(out, out=out, where=positive)
            np.multiply(out, source, out=out, where=where)
            np.copyto(out, np.inf, where=negative)
        finally:
            if scratch is not None:
                self._free_scratch('_aux')
        return out
   
    def _f_deriv(self, u, out=None, mask=None, **kwargs):
        # out = 1 + log(u / w)
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.divide(u, self.w, out=out, where=where)
        np.log(out, out=out, where=where)
        np.add(out, 1., out=out, where=where)
        return out

    def _f_second_deriv(self, u, out=None, mask=None, **kwargs):
        # out = 1 / u
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.divide(1., u, out=out, where=where)
        return out

    def _f_prox(self, v, tau, out=None, mask=None, **kwargs):
        # out = tau * lambertW( (w / tau) * exp(v / tau - 1) )
        # For v >= thres * tau an overflow occurs in the exponential (for standard doubles thres ~ 710) or
        # in the subsequent multiplication and division.
        # For such values we use an approximation via linearization (= one Newon step) instead of the exact formula above.
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        source = v
        source_scratch = None
        if np.may_share_memory(out, v):
            source_scratch = self._get_scratch('_aux', v)
            np.copyto(source_scratch, v, where=where)
            source = source_scratch
        scaled = self._get_scratch('_prox_scaled', v)
        try:
            max_w = np.max(self.w, where=where, initial=0.)
            thres = (
                np.log(np.finfo(source.dtype).max)
                - np.maximum(0., np.log(1. / tau))
                - np.maximum(0., np.log(max_w)) - 1.
            )
            regular = where & (source <= thres * tau)
            large = where & ~regular

            np.divide(source, tau, out=scaled, where=regular)
            np.subtract(scaled, 1., out=scaled, where=regular)
            np.exp(scaled, out=scaled, where=regular)
            np.multiply(scaled, self.w, out=scaled, where=regular)
            np.divide(scaled, tau, out=scaled, where=regular)
            np.copyto(scaled, 0., where=~regular)
            lambert = lambertw(scaled).real
            np.multiply(lambert, tau, out=out, where=regular)

            np.divide(source, self.w, out=scaled, where=large)
            np.log(scaled, out=scaled, where=large)
            np.multiply(scaled, tau, out=scaled, where=large)
            np.subtract(source, scaled, out=out, where=large)
            np.divide(tau, source, out=scaled, where=large)
            np.add(scaled, 1., out=scaled, where=large)
            np.divide(out, scaled, out=out, where=large)
        finally:
            self._free_scratch('_prox_scaled')
            if source_scratch is not None:
                self._free_scratch('_aux')
        return out

    def _f_conj_prox(self, vstar, tau, out=None, mask=None, **kwargs):
        # out = vstar - tau * f_prox(vstar / tau, 1 / tau)
        where = self._where(mask, vstar)
        out = self._prepare_out(out, vstar)
        source = vstar
        scratch = None
        if np.may_share_memory(out, vstar):
            scratch = self._get_scratch('_conj_prox_input', vstar)
            np.copyto(scratch, vstar, where=where)
            source = scratch
        try:
            np.divide(source, tau, out=out, where=where)
            self._f_prox(out, 1. / tau, out=out, mask=mask, **kwargs)
            np.multiply(out, tau, out=out, where=where)
            np.subtract(source, out, out=out, where=where)
        finally:
            if scratch is not None:
                self._free_scratch('_conj_prox_input')
        return out

    def _f_conj(self, u_star, out=None, mask=None, **kwargs):
        # out = w * (exp(u_star - 1))
        where = self._where(mask, u_star)
        out = self._prepare_out(out, u_star)
        np.subtract(u_star, 1., out=out, where=where)
        np.exp(out, out=out, where=where)
        np.multiply(out, self.w, out=out, where=where)
        return out

    def _f_conj_deriv(self, u_star, out=None, mask=None, **kwargs):
        return self._f_conj(u_star, out=out, mask=mask, **kwargs)
    
    def _f_conj_second_deriv(self, u_star, out=None, mask=None, **kwargs):
        return self._f_conj(u_star, out=out, mask=mask, **kwargs)

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
    # This could also be implemented as special case of Quadratic

    def  __init__(self, domain, as_primal=True, sigma = 1., eps=1e-10, **kwargs):
        if not isinstance(domain,MeasureSpaceFcts): raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,"Huber domain needs to be a MeasureSpaceFcts instance."))
        if not isinstance(sigma, (float,int)) and sigma not in domain:
            raise TypeError(Errors.type_error(f"Sigma in the HuberFunctional needs to be a scalar or element in the domain! Given:"+"\n\t "+f"sigma = {sigma}",self))
        if isinstance(sigma,int):
            self.sigma = np.broadcast_to(np.real(float(sigma)),domain.shape)
        elif isinstance(sigma, float):
            self.sigma = np.broadcast_to(np.real(sigma),domain.shape)
        else:
            self.sigma = np.real(sigma)
        if np.min(sigma)<=0:
            raise ValueError(Errors.value_error(f'sigma must be positive. min(sigma)={np.min(sigma)}',self))
        if as_primal:
            super().__init__(domain,Lipschitz=1.,
                             conj_nat_dom_l=-self.sigma, conj_nat_dom_u = self.sigma,                             
                             **kwargs)
            self.conjugate = QuadraticIntv(domain,as_primal=False,sigma=sigma,eps=eps)
        else:
            dual_domain = deepcopy(domain)
            dual_domain.measure = 1./domain.measure
            super().__init__(dual_domain, Lipschitz=1., **kwargs)
        # auxiliary vectors
        self._abs_u = self.domain.zeros() 
        self._small = np.zeros(self.domain.shape,dtype=bool)

    def _f(self, u, out=None, mask=None, **kwargs):
        # out = 1/2 * abs(u)**2                            if abs(u) <= sigma
        # out = sigma * abs(x) - 1/2 * sigma**2            else
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        abs_u = self._get_scratch('_abs_u', u)
        try:
            np.abs(u, out=abs_u, where=where)
            small = where & (abs_u <= self.sigma)
            large = where & ~small
            np.square(abs_u, out=out, where=small)
            np.multiply(out, 0.5, out=out, where=small)
            np.multiply(self.sigma, abs_u, out=out, where=large)
            np.square(self.sigma, out=abs_u, where=large)
            np.multiply(abs_u, 0.5, out=abs_u, where=large)
            np.subtract(out, abs_u, out=out, where=large)
        finally:
            self._free_scratch('_abs_u')
        return out

    def _f_deriv(self, u, out=None, mask=None, **kwargs):
        # out = u                      if abs(u) <= sigma
        # out = sigma * sign(u)        else
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.maximum(u, -self.sigma, out=out, where=where)
        np.minimum(out, self.sigma, out=out, where=where)
        return out

    def _f_second_deriv(self, u, out=None, mask=None, **kwargs):
        # out = 1   if abs(u) <= sigma
        # out = 0   else
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        inside = np.abs(u) <= self.sigma
        np.copyto(out, inside, where=where, casting='unsafe')
        return out

    def _f_conj(self, ustar, out=None, mask=None, **kwargs):
        return self.conjugate._f(ustar, out=out, mask=mask, **kwargs)
   
    def _f_conj_deriv(self, ustar, out=None, mask=None, **kwargs):
        return self.conjugate._f_deriv(ustar, out=out, mask=mask, **kwargs)

    def _f_conj_second_deriv(self, ustar, out=None, mask=None, **kwargs):
        return self.conjugate._f_second_deriv(ustar, out=out, mask=mask, **kwargs)

    def _f_conj_prox(self, ustar, tau, out=None, mask=None, **kwargs):
        return self.conjugate._f_prox(ustar, tau, out=out, mask=mask, **kwargs)


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
    # This could also be implemented as special case of Quadratic

    def  __init__(self, domain, as_primal=True, sigma=1., eps=1e-10, **kwargs):
        if not isinstance(domain,MeasureSpaceFcts): 
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,"QuadraticIntv domain needs to be a MeasureSpaceFcts instance."))
        if not isinstance(sigma, (float,int)) and sigma not in domain:
            raise TypeError(Errors.type_error(f"Sigma in the QuadraticIntv needs to be a scalar or element in the domain! Given:"+"\n\t "+f"sigma = {sigma}",self))
        if np.min(sigma)<=0:
            raise ValueError(Errors.value_error("The interval width in the QuadraticIntv needs to be positive!", self))
        self.eps=eps
        if isinstance(sigma, (float,int)):
            self.sigma = np.broadcast_to(np.real(sigma), domain.shape)
            self.sigmaeps = np.broadcast_to(sigma*(1+eps), domain.shape)           
        else:
            self.sigma = sigma 
            self.sigmaeps = self.sigma*(1+eps) if eps>0 else self.sigma            
        if as_primal:
            super().__init__(domain,convexity_param=1,
                             nat_dom_l=-self.sigmaeps,nat_dom_u=self.sigmaeps,
                             **kwargs)
            self.conjugate = Huber(domain,as_primal=False,sigma=sigma)
        else:
            dual_domain = deepcopy(domain)
            dual_domain.measure = 1./domain.measure
            super().__init__(dual_domain, convexity_param=1,                                                     
                         **kwargs)   

    def _f(self, u, out=None, mask=None, **kwargs):
        # out = 0.5 * abs(u)**2    if abs(u)<=sigmaeps
        # out = np.inf             else
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        outside = where & (np.abs(u) > self.sigmaeps)
        np.square(u, out=out, where=where)
        np.multiply(out, 0.5, out=out, where=where)
        np.copyto(out, np.inf, where=outside)
        return out
   
    def _f_deriv(self, u, out=None, mask=None, **kwargs):
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.copyto(out, u, where=where)
        return out

    def _f_prox(self, u, tau, out=None, mask=None, **kwargs):
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.divide(u, 1. + tau, out=out, where=where)
        aux = self._get_scratch('_aux', u)
        try:
            np.abs(out, out=aux, where=where)
            np.divide(aux, self.sigma, out=aux, where=where)
            np.maximum(aux, 1., out=aux, where=where)
            np.divide(out, aux, out=out, where=where)
        finally:
            self._free_scratch('_aux')
        return out

    def _f_second_deriv(self, u, out=None, mask=None, **kwargs):
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.copyto(out, 1., where=where)
        return out

    def _f_conj(self, ustar, out=None, mask=None, **kwargs):
        return self.conjugate._f(ustar, out=out, mask=mask, **kwargs)
   
    def _f_conj_deriv(self, ustar, out=None, mask=None, **kwargs):
        return self.conjugate._f_deriv(ustar, out=out, mask=mask, **kwargs)

    def _f_conj_second_deriv(self, ustar, out=None, mask=None, **kwargs):
        return self.conjugate._f_second_deriv(ustar, out=out, mask=mask, **kwargs)

    def _f_conj_prox(self, ustar, tau, out=None, mask=None, **kwargs):
        return self.conjugate._f_prox(ustar, tau, out=out, mask=mask, **kwargs)
    
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
    # This could also be implemented as special case of Quadratic

    def  __init__(self, domain, **kwargs):
        super().__init__(domain,convexity_param = 1., nat_dom_l=0.,
                         methods =  {'eval', 'subgradient', 'hessian', 'proximal'},
                         conj_methods =  {'eval', 'subgradient', 'hessian', 'proximal'},
                         **kwargs)

    def _f(self, u, out=None, mask=None, **kwargs):
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        negative = where & (u < 0)
        np.square(u, out=out, where=where)
        np.multiply(out, 0.5, out=out, where=where)
        np.copyto(out, np.inf, where=negative)
        return out

    def _f_deriv(self, u, out=None, mask=None, **kwargs):
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.copyto(out, u, where=where)
        return out

    def _f_prox(self, u, tau, out=None, mask=None, **kwargs):
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.divide(u, 1. + tau, out=out, where=where)
        np.maximum(out, 0., out=out, where=where)
        return out

    def _f_second_deriv(self, u, out=None, mask=None, **kwargs):
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.copyto(out, 1., where=where)
        return out

    def _f_conj(self, ustar, out=None, mask=None, **kwargs):
        where = self._where(mask, ustar)
        out = self._prepare_out(out, ustar)
        negative = where & (ustar < 0)
        np.square(ustar, out=out, where=where)
        np.multiply(out, 0.5, out=out, where=where)
        np.copyto(out, 0., where=negative)
        return out

    def _f_conj_deriv(self, ustar, out=None, mask=None, **kwargs):
        where = self._where(mask, ustar)
        out = self._prepare_out(out, ustar)
        np.copyto(out, ustar, where=where)
        np.copyto(out, 0., where=where & (ustar < 0))
        return out

    def _f_conj_second_deriv(self, ustar, out=None, mask=None, **kwargs):
        where = self._where(mask, ustar)
        out = self._prepare_out(out, ustar)
        nonnegative = ustar >= 0
        np.copyto(out, nonnegative, where=where, casting='unsafe')
        return out

    def _f_conj_prox(self, ustar, tau, out=None, mask=None, **kwargs):
        where = self._where(mask, ustar)
        out = self._prepare_out(out, ustar)
        np.copyto(out, ustar, where=where)
        positive = where & (ustar > 0)
        np.divide(ustar, 1. + tau, out=out, where=positive)
        return out
    
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
    lb: domain, optional
        lower bound, default:0
    ub: domain, optional
        upper bound, default:1
    x0: domain, optional
        reference value default 0.5*(lb+ub)
    alpha: float [default: 1]
        regularization parameter
    eps: real [default: 0]
        Tolerance parameter for violations of the hard constraints (which may occur due to rounding errors).
        If constraints are violated by less then eps times the interval width, the polynomial is evaluated, rather than returning np.inf.
    """
    # This could also be implemented as special case of Quadratic

    def __init__(self,domain:MeasureSpaceFcts, 
                 lb=None, ub=None, x0=None, alpha:float=1., **kwargs):
        if not isinstance(domain,MeasureSpaceFcts): raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,"QuadraticBilateralConstraints domain needs to be a MeasureSpaceFcts instance."))
        if isinstance(lb,(float,int)):
            lb = lb*domain.ones()
        elif lb is None:
            lb = domain.zeros()
        elif lb not in domain:
            raise TypeError(Errors.type_error(f"Lower bound lb in QuadraticBilateralConstraints needs to be a scalar or element in the domain! Given:"+"\n\t "+f"lb = {lb}"))
        if isinstance(ub,(float,int)):
            ub = ub*domain.ones()
        elif ub is None:
            ub = domain.ones()
        elif ub not in domain:
            raise TypeError(Errors.type_error(f"Upper bound ub in QuadraticBilateralConstraints needs to be a scalar or element in the domain! Given:"+"\n\t "+f"ub = {ub}"))
        if np.any(lb>=ub):
            raise TypeError(Errors.type_error(f"The lower bound needs to be below the upper bound in QuadraticBilateralCOnstraint:"+"\n\t "+f"lb = {lb},"+"\n\t "+f"ub = {ub}"))
        if x0 is None:
            x0 =0.5*(lb+ub)
        elif isinstance(x0,(float,int)):
            x0 = x0*domain.ones()
        elif x0 not in domain:
            raise TypeError(Errors.type_error(f"Reference value x0 in QuadraticBilateralConstraints needs to be a scalar or element in the domain! Given:"+"\n\t "+f"x0 = {x0}"))

        if not isinstance(alpha,(float,int)):
            raise TypeError(Errors.not_instance(alpha,float,add_info="Regularization parameter alpha in QuadraticBilateralConstraints needs to be a scalar!"))

        self.lb = lb; self.ub = ub; self.x0 =x0; self.alpha = alpha
        F = QuadraticIntv(domain,sigma=(ub-lb)/2., **kwargs)
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

def QuadraticLowerBound(domain, lb=None, x0=None, a=1.):
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
        raise TypeError(Errors.type_error(f"Lower bound lb in QuadraticLowerBound needs to be a scalar or element in the domain! Given:"+"\n\t "+f"lb = {lb}"))
    if isinstance(x0,(float,int)):
        x0 = x0*domain.ones()
    elif x0 is None:
        x0 = domain.zeros()
    elif x0 not in domain:
        raise TypeError(Errors.type_error(f"Reference value x0 in QuadraticLowerBound needs to be a scalar or element in the domain! Given:"+"\n\t "+f"x0 = {x0}"))
    if not isinstance(a,(float,int)):
            raise TypeError(Errors.not_instance(a,float,add_info="Scaling factor a in QuadraticLowerBound needs to be a scalar!"))

    F = QuadraticNonneg(domain)
    lin = LinearFunctional(lb-x0,domain=domain,gradient_in_dual_space=False)
    offset = 0.5*(np.sum((x0**2-lb**2)*domain.measure))
    return a*HorizontalShiftDilation(F,shift=lb)+ a*lin + a*offset
 
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

    def _subgradient(self, x, out=None, **subgradient_par):
        if out is None:
            out = np.empty_like(x)
        np.sign(x, out=out)
        return out

    def _hessian(self, x):
        # Even approximate Hessians don't work here.
        raise NotImplementedError

    def _proximal(self, x, tau, out=None, **proximal_par):
        if out is None:
            out = np.empty_like(x)
            source = x
        else:
            source = x.copy() if np.may_share_memory(out, x) else x
        np.abs(source, out=out)
        np.subtract(out, tau, out=out)
        np.maximum(out, 0, out=out)
        np.copysign(out, source, out=out)
        return out
