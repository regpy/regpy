from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable
    from typing_extensions import Self
    from types import NotImplementedType
    from numpy.typing import ArrayLike
    from regpy.vecsps.base import Scalar, Vector
    from regpy.vecsps.numpy import MeasureSpaceFcts
    from regpy.operators.base import Operator
    from regpy._typing import Mask
from copy import deepcopy

import numpy as np
from scipy.special import lambertw

from regpy.util import Errors, memoized_property
from regpy.vecsps import ReadOnlyVector
from regpy.vecsps.numpy import *
import regpy.operators as rop
from regpy.hilbert import L2

from .base import Functional, LinearFunctional, LinearCombination,HorizontalShiftDilation, NotInEssentialDomainError,NotTwiceDifferentiableError, as_functional

__all__ = ["IntegralFunctionalBase","LppPower","L1MeasureSpace","KullbackLeibler","KullbackLeiblerOffset","RelativeEntropy","Huber","Huber_old","Quadratic","QuadraticIntv","QuadraticBilateralConstraints","QuadraticLowerBound","QuadraticNonneg","L1Generic"]


class IntegralFunctionalBase(Functional):
    r"""This class provides a general framework for integral functionals of the type
    
    .. math::
        F\colon X &\to \mathbb{R} \\
        v&\mapsto \int_\Omega f(v(x),x)\mathrm{d}x
    
    with :math:`f\colon \mathbb{R}^2\to \mathbb{R}`. 
    
    Subclasses defining explicit functionals of this type have to implement
     * `_f` evaluation the function :math:`f`
     * `_f_deriv` the derivative :math:`\partial_v f`
     * `_f_second_deriv' the second derivative :math:`\partial_v^2 f` (often not needed!)
     * `_f_prox` giving the proximal function :math:`\mathrm{prox}_{\tau f(.,x)}` for each :math:`x\in\Omega`
     * `_f_conj` evaluation the Fenchel conjugate function :math:`f^*(v^*,x)`
     * `_f_conj_deriv` the derivative :math:`\partial_{v^*}f^*`
     * `_f_conj_second_deriv' the second derivative :math:`\partial_{v^*}^2 f^*` (often not needed!)
     * `_f_conj_prox` giving the proximal  function :math:`\mathrm{prox}_{\tau f^*(\cdot,x)}`     
    
    (where `_f_deriv`, `_f_prox`, `_f_conj`, and `_f_conj_prox` must be able to handle out=input) since 
    
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
    domain: MeasureSpaceFcts
        Domain on which it is defined. Needs some Measure therefore a `regpy.vecsps.MeasureSpaceFcts`
    nat_dom_l: float | ReadOnlyVector [default: -np.inf]
        lower and upper boundary of the natural essential domain of :math:`f`, i.e., the interval on which :math:`f` without artificial constraints or Taylor expansion is finite 
        If dom_l or dom_u are finite, :math:`f` should be finite at these points (possibly very large if :math:`f` tends to infinity there) 
        If the domain depends on the point `x` and/or arguments in `**kwargs`, this must be a NumPy array.
    nat_dom_u: float | ReadOnlyVector [default: np.inf]
        lower and upper boundary of the natural essential domain of :math:`f`, i.e., the interval on which :math:`f` without artificial constraints or Taylor expansion is finite 
        If dom_l or dom_u are finite, :math:`f` should be finite at these points (possibly very large if :math:`f` tends to infinity there) 
        If the domain depends on the point `x` and/or arguments in `**kwargs`, this must be a NumPy array.
    conj_nat_dom_l: float | ReadOnlyVector [default: -np.inf]
        lower and upper boundary of the natural essential domain of the conjugate of :math:`f` 
        (as vector in primal space)
    conj_nat_dom_u: float | ReadOnlyVector [default: np.inf]
        lower and upper boundary of the natural essential domain of the conjugate of :math:`f` 
        (as vector in primal space)
    constr_l: float | ReadOnlyVector | None [default: None]
        If not None, a lower hard constraint is imposed, and constr_u specifies :math:`a(x)`
    constr_u: float | ReadOnlyVector | None [default: None]
        If not None, an upper hard constraint is imposed, and constr_u specifies :math:`b(x)`
    lin_taylor_l: float | ReadOnlyVector | None [default: None]
        If not None, :math:`a(x)` for linear Taylor on the left hand side
    lin_taylor_u: float | ReadOnlyVector | None [default: None]
        If not None, :math:`a(x)` for linear Taylor on the right-hand side
    quad_taylor_l: float | ReadOnlyVector | None [default: None]
        If not None, :math:`a(x)` for quadratic Taylor on the left hand side
    quad_taylor_u: float | ReadOnlyVector | None [default: None]
        If not None, :math:`a(x)` for quadratic Taylor on the right-hand side
    Lipschitz: float [default: np.inf]
        Lipschitz bound for the derivative.
    convexity_param: float [default: 0.0]
        Strong-convexity parameter.
    methods: set[str] | None [default: None]
        Names of the implemented functional operations.
    conj_methods: set[str] | None [default: None]
        Names of the implemented functional operations.
    is_data_func: bool [default: False]
        Is data func.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    """

    def __init__(self, domain: MeasureSpaceFcts,
                 nat_dom_l: float | ReadOnlyVector=-np.inf, nat_dom_u: float | ReadOnlyVector=np.inf, 
                 conj_nat_dom_l: float | ReadOnlyVector=-np.inf,conj_nat_dom_u: float | ReadOnlyVector=np.inf,
                 constr_l: float | ReadOnlyVector | None=None, constr_u: float | ReadOnlyVector | None=None,
                 lin_taylor_l: float | ReadOnlyVector | None=None, lin_taylor_u: float | ReadOnlyVector | None=None,
                 quad_taylor_l: float | ReadOnlyVector | None=None, quad_taylor_u: float | ReadOnlyVector | None= None,
                 Lipschitz: float = np.inf, convexity_param: float=0.,
                 methods: set[str] | None = None, conj_methods: set[str] | None = None,
                 is_data_func: bool = False,
                 **kwargs: Any) -> None:
        if not isinstance(domain,MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,add_info="IntegralFunctionalBase only implemented for real-valued MeasureSpaceFcts domains."))
        if not domain.shape_codomain == () or not np.issubdtype(domain.dtype, np.floating):
            raise RuntimeError(Errors.runtime_error(f'IntegralFunctionalBase only implemented for real-valued MeasureSpaceFcts. Got dtype={domain.dtype}, shape_codomain = {domain.shape_codomain}. Consider using  VectorIntegralFunctional instead.'))
        self.h_domain = L2(domain)
        r""":math:`L^2` Hilbert space on self.domain with respect to self.measure."""
        self.measure = np.broadcast_to(domain.measure,domain.shape) if np.isscalar(domain.measure) else domain.measure
        r"""The measure of self.domain"""
        self.domain = domain
        r"""The vector space on which the functional is defined."""

        all_methods = {'eval', 'subgradient', 'hessian', 'proximal', 'dist_subdiff'}

        super().__init__(domain, Lipschitz=Lipschitz, convexity_param=convexity_param,
                         is_separable=True,
                         methods = methods if methods is not None else all_methods, 
                         conj_methods = conj_methods if conj_methods is not None else all_methods,
                         is_data_func = is_data_func
                         )

        self._ESS_DOM_TOL = 1e-10
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
    def constr_l(self) -> np.ndarray | None:
        r"""Constr l.
        
        
        Returns
        -------
        np.ndarray | None
            Result described above.
        """
        return self._constr_l
    
    @constr_l.setter
    def constr_l(self, constr_l: float | ReadOnlyVector) -> None:
        r"""Constr l.
        
        Parameters
        ----------
        constr_l: float | ReadOnlyVector
            Lower pointwise constraint; None leaves this side unconstrained.
        """
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
    def constr_u(self) -> np.ndarray | None:
        r"""Constr u.
        
        
        Returns
        -------
        np.ndarray | None
            Result described above.
        """
        return self._constr_u
    
    @constr_u.setter
    def constr_u(self, constr_u: float | ReadOnlyVector) -> None:
        r"""Constr u.
        
        Parameters
        ----------
        constr_u: float | ReadOnlyVector
            Upper pointwise constraint; None leaves this side unconstrained.
        """
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
    def lin_taylor_l(self) -> np.ndarray | None:
        r"""Lin taylor l.
        
        
        Returns
        -------
        np.ndarray | None
            Result described above.
        """
        return self._lin_taylor_l

    @lin_taylor_l.setter        
    def lin_taylor_l(self, lin_taylor_l: float | ReadOnlyVector) -> None:
        r"""Lin taylor l.
        
        Parameters
        ----------
        lin_taylor_l: float | ReadOnlyVector
            Lower threshold for a linear extension; None disables this extension.
        """
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
    def lin_taylor_u(self) -> np.ndarray | None:
        r"""Lin taylor u.
        
        
        Returns
        -------
        np.ndarray | None
            Result described above.
        """
        return self._lin_taylor_u

    @lin_taylor_u.setter
    def lin_taylor_u(self, lin_taylor_u: float | ReadOnlyVector) -> None:
        r"""Lin taylor u.
        
        Parameters
        ----------
        lin_taylor_u: float | ReadOnlyVector
            Upper threshold for a linear extension; None disables this extension.
        """
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
    def quad_taylor_l(self) -> np.ndarray | None:
        r"""Quad taylor l.
        
        
        Returns
        -------
        np.ndarray | None
            Result described above.
        """
        return self._quad_taylor_l
    
    @quad_taylor_l.setter
    def quad_taylor_l(self, quad_taylor_l: float | ReadOnlyVector) -> None:
        r"""Quad taylor l.
        
        Parameters
        ----------
        quad_taylor_l: float | ReadOnlyVector
            Lower threshold for a quadratic extension; None disables this extension.
        """
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
    def quad_taylor_u(self) -> np.ndarray | None:
        r"""Quad taylor u.
        
        
        Returns
        -------
        np.ndarray | None
            Result described above.
        """
        return self._quad_taylor_u
    
    @quad_taylor_u.setter
    def quad_taylor_u(self, quad_taylor_u: float | ReadOnlyVector) -> None:
        r"""Quad taylor u.
        
        Parameters
        ----------
        quad_taylor_u: float | ReadOnlyVector
            Upper threshold for a quadratic extension; None disables this extension.
        """
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
  
    def _precompute_f_l(self) -> None:
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
            taylor_active_l = np.isfinite(self._taylor_l) & (self._taylor_l >= self.dom_l)
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

    def _delete_f_l(self) -> None:
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

    def _precompute_f_u(self) -> None:
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
            taylor_active_u = np.isfinite(self._taylor_u) & (self._taylor_u <= self.dom_u)
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

    def _delete_f_u(self) -> None:
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
    def _taylor_l(self)->np.ndarray | None:
        if self.lin_taylor_l is not None:
            return self.lin_taylor_l 
        elif self.quad_taylor_l is not None:
            return self.quad_taylor_l
        else:
            return None
        
    @property
    def _taylor_u(self)->np.ndarray | None:
        if self.lin_taylor_u is not None:
            return self.lin_taylor_u 
        elif self.quad_taylor_u is not None:
            return self.quad_taylor_u
        else:
            return None

    @property
    def _Lipschitz_exterior(self) -> float:
        return np.inf if (self.constr_l is not None or self.constr_u is not None) else 0.
    
    @property
    def Lipschitz(self) -> float:
        r"""Lipschitz.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return max([super().Lipschitz,self._Lipschitz_exterior])

    @property
    def _convexity_param_exterior(self) -> float:
        return 0. if (self.lin_taylor_l is not None or self.lin_taylor_u is not None) else np.inf

    @property
    def convexity_param(self) -> float:
        r"""Convexity param.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return min([super().convexity_param,self._convexity_param_exterior])    

    @property
    def dom_l(self)->np.ndarray:
        r"""Dom l.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        toret = self._nat_dom_l.copy()
        if self._taylor_l is None:
            if not self.constr_l is None:
                toret = np.maximum(self._nat_dom_l, self.constr_l)
        else:
            toret[self._taylor_l>=self._nat_dom_l] = -np.inf
        return self._if_constant_broadcast(toret)
    
    @property
    def dom_u(self)->np.ndarray:
        r"""Dom u.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        toret = self._nat_dom_u.copy()
        if self._taylor_u is None:
            if self.constr_u is not None:
                toret = np.minimum(self._nat_dom_u, self.constr_u)
        else:
            toret[self._taylor_u<=self._nat_dom_u] = np.inf
        return self._if_constant_broadcast(toret)

    @property
    def conj_dom_l(self) -> np.ndarray:
        r"""Conj dom l.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
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
        r"""Conj dom u.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
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
        r"""Everywhere finite.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return np.all(self.dom_l==-np.inf) and np.all(self.dom_u==np.inf)
    
    @property
    def conj_everywhere_finite(self)->bool: 
        r"""Conj everywhere finite.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
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

    def _if_constant_broadcast(self, x: ReadOnlyVector, mask: Mask | None=None) -> ReadOnlyVector:
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

    def _get_scratch(self, name: str, like: np.ndarray) -> np.ndarray:
        r"""Return a persistent scratch array (stored as ``self.<name>``) with the same
        shape and dtype as ``like``, reusing the previous allocation whenever possible.
        
        Parameters
        ----------
        name: str
            Name.
        like: np.ndarray
            Like.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        if name in self._buffer_status and self._buffer_status[name] == True:
            raise RuntimeError(Errors.generic_message('Now request for buffer {name}, but this buffer is already in use.'))
        buf = getattr(self, name, None)
        if buf is None or buf.shape != like.shape or buf.dtype != like.dtype:
            buf = np.empty_like(like)            
            setattr(self, name, buf)
        self._buffer_status[name] = True
        return buf

    def _free_scratch(self,name: str) -> None:
        r"""Indicate that a buffer is no longer in use and hence not blocked.
        Does not free the memory!
        
        Parameters
        ----------
        name: str
            Name.
        """
        self._buffer_status[name] = False

    @staticmethod
    def _prepare_out(out: Vector, like: np.ndarray) -> np.ndarray:
        r"""Return a writable output array compatible with ``like``.
        If ``out`` is None, a new array is allocated and returned.
        If ``out`` is an array, it is returned without initialization.
        This is used in pointwise operations to share code between allocating and
        in-place implementations.
        
        Parameters
        ----------
        out: Vector
            Writable output buffer. If None, allocate a new result.
        like: np.ndarray
            Like.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
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
    def _where(mask: Mask, like: np.ndarray) -> np.ndarray | bool:
        r"""Return ``mask`` as a boolean array broadcast over the full input shape.
        
        Parameters
        ----------
        mask: Mask
            Coordinate-selection mask in the format used by the vector space.
        like: np.ndarray
            Like.
        
        
        Returns
        -------
        np.ndarray | bool
            Result described above.
        """
        if mask is None:
            return np.broadcast_to(np.bool_(True), like.shape)
        where = np.asarray(mask, dtype=bool)
        if where.shape != like.shape:
            raise ValueError(Errors.value_error(
                f"mask and input must have identical shapes. Got {where.shape} and {like.shape}."
            ))
        return where

    def _apply_pointwise(self, func: Callable[..., np.ndarray], x: ReadOnlyVector, out: Vector, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        r"""Evaluate a pointwise method on full arrays using a NumPy ``where`` mask.
        
        Parameters
        ----------
        func: Callable[..., np.ndarray]
            Functional to evaluate or transform.
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        out: Vector
            Writable output buffer. If None, allocate a new result.
        mask: Mask | None [default: None]
            Coordinate-selection mask in the format used by the vector space.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        return func(x, out=out, mask=mask, **kwargs)

    def _assert_interval(self, v: ReadOnlyVector, lower: float | np.ndarray, upper: float | np.ndarray, msg: str | None=None) -> None:
        r"""Check bounds without subtracting equal infinite endpoints.
        
        Infinite arguments are accepted at matching unbounded endpoints for
        internal limiting-value calculations. NaN is never admissible.
        
        Parameters
        ----------
        v: ReadOnlyVector
            Input vector; its coefficients are only read.
        lower: float | np.ndarray
            Lower.
        upper: float | np.ndarray
            Upper.
        msg: str | None [default: None]
            Msg.
        """
        msg = "" if msg is None else msg
        if np.any(np.isnan(v)):
            raise NotInEssentialDomainError(Errors.generic_message(
                msg + f" NaN argument in {self}."))
        # Adding a finite tolerance leaves infinite bounds unchanged.
        if np.any(v > np.asarray(upper) + self._ESS_DOM_TOL):
            raise NotInEssentialDomainError(Errors.generic_message(
                msg + f" argument too large in {self}."))
        if np.any(v < np.asarray(lower) - self._ESS_DOM_TOL):
            raise NotInEssentialDomainError(Errors.generic_message(
                msg + f" argument too small in {self}."))

    def _assert_essential_domain(self, v: ReadOnlyVector, msg: str | None=None) -> None:
        self._assert_interval(v, self.dom_l, self.dom_u, msg)

    def _assert_conj_essential_domain(self, vstar: ReadOnlyVector, msg: str | None=None) -> None:
        self._assert_interval(vstar, self.conj_dom_l, self.conj_dom_u, msg)

    def bregman(self, x: ReadOnlyVector, y: ReadOnlyVector | None,
                ystar: ReadOnlyVector | None = None) -> float:
        r"""Evaluate the Bregman divergence by summing pointwise differences.
        
        Computes F(x)-F(y)-<ystar,x-y>, with ystar in the
        subdifferential at y. If omitted, ystar is computed by
        subgradient(y). Constraints and linear/quadratic extensions are included. 
        
        With y=None, ystar is required and the pointwise Fenchel--Young
        expression F(x)+F*(ystar)-<ystar,x> is integrated instead.
        
        Subtraction precedes the final integration, reducing cancellation
        between large global values. 
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        y: ReadOnlyVector | None
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return self._pointwise_bregman(x, y, ystar, conjugate=False)

    def conj_bregman(self, xstar: ReadOnlyVector, ystar: ReadOnlyVector | None,
                     y: ReadOnlyVector | None = None) -> float:
        r"""Evaluate the conjugate Bregman divergence before integration.
        
        ystar=None uses F*(xstar)+F(y)-<y,xstar> and requires y.
        
        Parameters
        ----------
        xstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None
            Input vector; its coefficients are only read.
        y: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return self._pointwise_bregman(xstar, ystar, y, conjugate=True)

    def _pointwise_bregman(self, x: ReadOnlyVector, x0: ReadOnlyVector, pstar: ReadOnlyVector, *, conjugate: bool) -> float:
        r"""Shared pointwise reduction for the primal and conjugate.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        x0: ReadOnlyVector
            Input vector; its coefficients are only read.
        pstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        conjugate: bool
            Conjugate.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        if x0 is None:
            if pstar is None:
                raise ValueError("bregman requires x0 or pstar.")
            for name, value in (("x", x), ("pstar", pstar)):
                if value not in self.domain:
                    raise ValueError(Errors.not_in_vecsp(
                        value, self.domain, space_name="domain",
                        add_info=f"{name} of bregman must belong to the domain."
                    ))
            values = self.domain.empty()
            conjugate_values = self.domain.empty()
            evaluate = self._conj if conjugate else self._eval
            evaluate_conjugate = self._eval if conjugate else self._conj
            evaluate(x, func_vals=values, sum_values=False)
            evaluate_conjugate(pstar, func_vals=conjugate_values, sum_values=False)
            values += conjugate_values
            values *= self.measure
            values -= np.real(np.conjugate(pstar) * x)
            return float(np.sum(values))
        for name, value in (("x", x), ("x0", x0)):
            if value not in self.domain:
                raise ValueError(Errors.not_in_vecsp(
                    value, self.domain, space_name="domain",
                    add_info=f"{name} of bregman must belong to the domain."
                ))
        if pstar is None:
            pstar = self.conj_subgradient(x0) if conjugate else self.subgradient(x0)
        else:
            if pstar not in self.domain:
                raise ValueError(Errors.not_in_vecsp(
                    pstar, self.domain, space_name="domain",
                    add_info="The Bregman subgradient must belong to the domain."
                ))
            distance = (self._conj_dist_subdiff(pstar, x0) if conjugate
                        else self.dist_subdiff(pstar, x0))
            if distance > 0:
                raise ValueError(Errors.value_error(
                    "Given subgradient not in subdifferential",
                    (distance, pstar, x0)
                ))

        values = self.domain.empty()
        reference = self.domain.empty()
        evaluate = self._conj if conjugate else self._eval
        evaluate(x, func_vals=values, sum_values=False)
        evaluate(x0, func_vals=reference, sum_values=False)
        values -= reference
        values *= self.measure
        values -= np.real(np.conjugate(pstar) * (x - x0))
        return float(np.sum(values))

    def _eval(self, v: ReadOnlyVector, func_vals: Vector | None=None, *, sum_values: bool=True) -> float | None:
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
                buf[v < self.dom_l - self._ESS_DOM_TOL] = np.inf
            if self.constr_u is not None:
                buf[v > self.dom_u + self._ESS_DOM_TOL] = np.inf
            if func_vals is not None:
                np.copyto(func_vals, buf)
            if not sum_values:
                return None
            buf *= self.measure
            return np.sum(buf)
        finally:
            self._free_scratch('_buf')

    def _conj(self, vstar: ReadOnlyVector, func_vals: Vector | None=None, *, sum_values: bool=True) -> float | None:
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
                buf[vstar < self.conj_dom_l - self._ESS_DOM_TOL] = np.inf
            if self.lin_taylor_u is not None:
                buf[vstar > self.conj_dom_u + self._ESS_DOM_TOL] = np.inf
            if func_vals is not None:
                np.copyto(func_vals, buf)
            if not sum_values:
                return None
            buf *= self.measure
            return np.sum(buf)
        finally:
            self._free_scratch('_buf2')
            self._free_scratch('_buf')

    def _subgradient(self, v: ReadOnlyVector, out: Vector | None=None) -> Vector:
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
 
    @staticmethod
    def _at_boundary(x: ReadOnlyVector, bound: float | np.ndarray) -> np.ndarray:
        r"""Recognize a finite boundary up to floating-point roundoff.
        
        Coordinate transformations can move an exact boundary a few ulps
        into the interior. Use a small relative machine-precision tolerance,
        not NumPy's default isclose tolerance or the essential-domain
        tolerance. No absolute floor is used, so small weighted dual bounds
        retain the same relative precision as large ones.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        bound: float | np.ndarray
            Bound.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        eps = max(np.finfo(np.result_type(np.asarray(value), 1.)).eps
                  for value in (x, bound))
        return np.isfinite(bound) & np.isclose(x, bound, rtol=8 * eps, atol=0.)

    def dist_subdiff(self, vstar: ReadOnlyVector, x: ReadOnlyVector, best_approx: Vector | None=None) -> float:
        r"""Dual distance, optionally writing the projection to best_approx.
        
        best_approx must be a pre-allocated vector in self.domain and may
        alias either input. Pointwise hooks return the signed residual
        ``vstar - projection``; active bounds add their normal cones.
        Only points within roundoff of a bound acquire its normal cone from
        the interior. Points just outside a bound, accepted by the
        essential-domain tolerance, also use that bound.
        
        Parameters
        ----------
        vstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        best_approx: Vector | None [default: None]
            Writable vector used by this operation.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        self._validate_out(best_approx)
        self._assert_essential_domain(x, msg='dist_subdiff')
        diff = self._ptw_dist_subdiff(vstar, x)

        ind = (x >= self.dom_u) | self._at_boundary(x, self.dom_u)
        np.minimum(diff, 0., out=diff, where=ind)

        ind = (x <= self.dom_l) | self._at_boundary(x, self.dom_l)
        np.maximum(diff, 0., out=diff, where=ind)
        distance = self.h_domain.dual_space().norm(diff)
        if best_approx is not None:
            self._copy_vector(vstar - diff, out=best_approx)
        return distance

    def _conj_subgradient(self, vstar: ReadOnlyVector, out: Vector | None=None) -> Vector:
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

    def _conj_dist_subdiff(self, v: ReadOnlyVector, xstar: ReadOnlyVector, best_approx: Vector | None=None) -> float:
        r"""Primal distance and optional projection onto the conjugate subdiff.
        
        Parameters
        ----------
        v: ReadOnlyVector
            Input vector; its coefficients are only read.
        xstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        best_approx: Vector | None [default: None]
            Writable vector used by this operation.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        self._validate_out(best_approx)
        self._assert_conj_essential_domain(xstar, msg='_conj_dist_subdiff')
        diff = self._conj_ptw_dist_subdiff(v, xstar)

        ind = ((xstar >= self.conj_dom_u)
               | self._at_boundary(xstar, self.conj_dom_u))
        np.minimum(diff, 0., out=diff, where=ind)

        ind = ((xstar <= self.conj_dom_l)
               | self._at_boundary(xstar, self.conj_dom_l))
        np.maximum(diff, 0., out=diff, where=ind)
        distance = self.h_domain.norm(diff)
        if best_approx is not None:
            self._copy_vector(v - diff, out=best_approx)
        return distance

    def _hessian(self, v: ReadOnlyVector) -> Operator:
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

    def _conj_hessian(self, vstar: ReadOnlyVector) -> Operator:
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
    
    def _proximal(self, v: ReadOnlyVector, tau: float, mask: Mask | None=None, out: Vector | None=None) -> Vector:
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
    
    def _conj_proximal(self, vstar: ReadOnlyVector, tau: float, mask: Mask | None=None, out: Vector | None=None) -> Vector:
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

    def _f(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError
    
    def _f_deriv(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def _ptw_dist_subdiff(self,vstar: ReadOnlyVector, w: Scalar | ArrayLike, **kwargs: Any) -> np.ndarray:
        r"""Return the signed residual vstar minus its subdiff projection.
        
        Parameters
        ----------
        vstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        w: Scalar | ArrayLike
            W.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        return vstar-self.subgradient(w, **kwargs)

    def _f_second_deriv(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def _f_prox(self, v: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, tol: float=1e-12, abstol: float=1e-12,
                maxNewtonIter: int=15, maxBisecIter: int=300, maxBoundsIter: int=100,
                **kwargs: Any) -> np.ndarray:
        if self.__class__.__dict__.get("_f_deriv") is not None:
            f_second_deriv = self._f_second_deriv if self._f_second_deriv is not None else None
            return self._numerical_prox(
                v, tau, self._f_deriv, f_second_deriv, self.dom_l, self.dom_u,
                out=out, mask=mask, tol=tol, abstol=abstol,
                maxNewtonIter=maxNewtonIter, maxBisecIter=maxBisecIter,
                maxBoundsIter=maxBoundsIter, **kwargs
            )
        raise NotImplementedError('Need first derivative for numerical prox operator')
    
    def _f_conj(self, vstar: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError
    
    def _f_conj_deriv(self, vstar: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def _conj_ptw_dist_subdiff(self, v: ReadOnlyVector, wstar: ReadOnlyVector, **kwargs: Any) -> np.ndarray:
        r"""Return v minus its projection onto the conjugate subdifferential.
        
        Parameters
        ----------
        v: ReadOnlyVector
            Input vector; its coefficients are only read.
        wstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        return v-self.conj.subgradient(wstar, **kwargs)

    def _f_conj_second_deriv(self, vstar: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def _f_conj_prox(self, vstar: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None,
                     tol: float=1e-12, abstol: float=1e-12,
                     maxNewtonIter: int=15, maxBisecIter: int=300, maxBoundsIter: int=100,
                     **kwargs: Any) -> np.ndarray:
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

    def _numerical_prox(self, y: ReadOnlyVector, tau: float, fp: Callable[..., np.ndarray], fpp: Callable[..., np.ndarray], dom_l: float | ReadOnlyVector, dom_u: float | ReadOnlyVector,
                        start: np.ndarray | None=None, ub: float | ReadOnlyVector | None=None, lb: float | ReadOnlyVector | None=None, out: Vector | None=None, mask: Mask | None=None,
                        tol: float=1e-15, abstol: float=1e-15,
                        maxNewtonIter: int=15, maxBisecIter: int=300,
                        maxBoundsIter: int=100, **kwargs: Any) -> np.ndarray:
        r"""Vectorized proximal operator of a smooth convex scalar function
        implemented by Newton's method with automatic fallback to bisection.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input array.
        tau: float
            Prox parameter (>0).
        fp: Callable[..., np.ndarray]
            f, f', f'' (accept NumPy arrays).
        fpp: Callable[..., np.ndarray]
            f, f', f'' (accept NumPy arrays).
        dom_l: float | ReadOnlyVector
            Domain bounds (can be -np.inf, np.inf).
        dom_u: float | ReadOnlyVector
            Domain bounds (can be -np.inf, np.inf).
        start: np.ndarray | None [default: None]
            starting point
        ub: float | ReadOnlyVector | None [default: None]
            guesses for lower and upper bounds on solution (must not be valid!)
        lb: float | ReadOnlyVector | None [default: None]
            guesses for lower and upper bounds on solution (must not be valid!)
        out: Vector | None [default: None]
            If not None, this is used as the iterate and is returned
        mask: Mask | None [default: None]
            If not None, this specifies the subset of entries to which 
            the Newton-bisection method is applied
        tol: float [default: 1e-15]
            Root-finding tolerance for Newton/bisection.
        abstol: float [default: 1e-15]
            Absolute tolerance.
        maxNewtonIter: int [default: 15]
            Maxnewtoniter.
        maxBisecIter: int [default: 300]
            Maxbiseciter.
        maxBoundsIter: int [default: 100]
            Maxboundsiter.
        **kwargs: Any
            Passed to f,fp and fpp
        
        Other Parameters
        ----------------
        f: callables
            f, f', f'' (accept NumPy arrays).
        maxiter: int
            Max Newton iterations before falling back.
        
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

        def residual(z: ReadOnlyVector, result: np.ndarray, active: np.ndarray) -> np.ndarray:
            r"""Residual.
            
            Parameters
            ----------
            z: ReadOnlyVector
                Input vector; its coefficients are only read.
            result: np.ndarray
                Result.
            active: np.ndarray
                Active.
            
            
            Returns
            -------
            np.ndarray
                Computed array.
            """
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
    r"""Implements the norm power functional :math:`v\mapsto \frac{1}{p} \|v\|_{L^p}^p` power on some domain in `regpy.vecsps.MeasureSpaceFcts`
    as an integral functional. This corresponds to the function :math:`f(v):=\frac{1}{p}|v|^p`.
    
    Parameters
    ----------
    domain: MeasureSpaceFcts
        Domain on which it is defined. Needs some Measure therefore a `regpy.vecsps.MeasureSpaceFcts`
    p: float [default: 2.0]
        exponent
    constr_l: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    constr_u: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    lin_taylor_l: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    lin_taylor_u: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    quad_taylor_l: float | ReadOnlyVector | None [default: None]
        Lower threshold for a quadratic extension; None disables this extension.
    quad_taylor_u: float | ReadOnlyVector | None [default: None]
        Upper threshold for a quadratic extension; None disables this extension.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    """

    @property
    def is_strictly_convex(self) -> bool:
        r"""Strict convexity without linear tails; constraints are allowed.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(
            True
            and self.lin_taylor_l is None
            and self.lin_taylor_u is None
        )

    @property
    def conj_is_strictly_convex(self) -> bool:
        r"""Strict convexity of the conjugate without primal hard constraints.
        
        Primal constraints can generate affine tails in the conjugate.
        Linear primal extensions instead restrict its essential domain.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(
            True
            and self.constr_l is None
            and self.constr_u is None
        )

    def __init__(self, 
                domain: MeasureSpaceFcts, 
                p: float=2.,
                constr_l: float | ReadOnlyVector | None=None, 
                constr_u: float | ReadOnlyVector | None=None, 
                lin_taylor_l: float | ReadOnlyVector | None=None, 
                lin_taylor_u: float | ReadOnlyVector | None=None,
                quad_taylor_l: float | ReadOnlyVector | None=None, 
                quad_taylor_u: float | ReadOnlyVector | None=None,
                **kwargs: Any
                ) -> None:       
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
    def is_norm_squared_type(self) -> bool:
        r"""Whether the functional is norm squared type.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.p==2 and all([x is None for x in (self.constr_l, self.lin_taylor_l, self.quad_taylor_l,
                                                      self.constr_u, self.lin_taylor_u, self.quad_taylor_u)])

    @property
    def Lipschitz(self) -> float:
        r"""Lipschitz.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
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
    def convexity_param(self) -> float:
        r"""Convexity param.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
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

    def _f(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = np.abs(v)**self.p/self.p
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.abs(v, out=out, where=where)
        np.power(out, self.p, out=out, where=where)
        np.divide(out, self.p, out=out, where=where)
        return out
    
    def _f_deriv(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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
    
    def _f_second_deriv(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = (self.p-1)*np.abs(v)**(self.p-2)
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.abs(v, out=out, where=where)
        np.power(out, self.p - 2, out=out, where=where)
        np.multiply(out, self.p - 1, out=out, where=where)
        return out
    
    def _f_prox(self, v: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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
    
    def _f_conj(self, vstar: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = np.abs(vstar)**self.q/self.q
        where = self._where(mask, vstar)
        out = self._prepare_out(out, vstar)
        np.abs(vstar, out=out, where=where)
        np.power(out, self.q, out=out, where=where)
        np.divide(out, self.q, out=out, where=where)
        return out

    def _f_conj_deriv(self, vstar: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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
    
    def _f_conj_second_deriv(self, vstar: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = (self.q-1)*np.abs(vstar)**(self.q-2)
        where = self._where(mask, vstar)
        out = self._prepare_out(out, vstar)
        np.abs(vstar, out=out, where=where)
        np.power(out, self.q - 2, out=out, where=where)
        np.multiply(out, self.q - 1, out=out, where=where)
        return out
    
    def _f_conj_prox(self, v_star: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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
    domain: MeasureSpaceFcts
        Domain on which it is defined. Needs some Measure therefore a `regpy.vecsps.MeasureSpaceFcts`
    a: float [default: 1.0]
        First coefficient of the quadratic functional.  If scalar, it is broadcasted to the shape of the domain.
    b: float [default: 0.0]
        Second coefficient of the quadratic functional. 
    c: float [default: 0.0]
        Third coefficient of the quadratic functional.
    constr_l: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    constr_u: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    lin_taylor_l: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    lin_taylor_u: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    quad_taylor_l: float | ReadOnlyVector | None [default: None]
        Lower threshold for a quadratic extension; None disables this extension.
    quad_taylor_u: float | ReadOnlyVector | None [default: None]
        Upper threshold for a quadratic extension; None disables this extension.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    """
    def __init__(self, 
                 domain: MeasureSpaceFcts,
                 a: float=1., b: float=0., c: float=0.,
                 constr_l: float | ReadOnlyVector | None=None, 
                 constr_u: float | ReadOnlyVector | None=None, 
                 lin_taylor_l: float | ReadOnlyVector | None=None, 
                 lin_taylor_u: float | ReadOnlyVector | None=None,
                 quad_taylor_l: float | ReadOnlyVector | None=None, 
                 quad_taylor_u: float | ReadOnlyVector | None=None,
                 **kwargs: Any
                 ) -> None:
        if not np.isscalar(a) and a not in domain:
            raise ValueError(Errors.value_error(f'a needs to be either scalar or an element of the domain. Given a: {a} of type {type(a)}', self))
        self._a = np.broadcast_to(a, domain.shape) if np.isscalar(a) else a
        if not np.isscalar(b) and b not in domain:  
            raise ValueError(Errors.value_error(f'b needs to be either scalar or an element of the domain. Given b: {b} of type {type(b)}', self))
        self._b = np.broadcast_to(b, domain.shape) if np.isscalar(b) else b
        if not np.isscalar(c) and c not in domain:  
            raise ValueError(Errors.value_error(f'c needs to be either scalar or an element of the domain. Given c: {c} of type {type(c)}', self))
        self._c = np.broadcast_to(c, domain.shape) if np.isscalar(c) else c

        super().__init__(domain, is_quadratic=True,
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                         quad_taylor_l=quad_taylor_l, quad_taylor_u=quad_taylor_u,
                         **kwargs
                         )
    def bregman_func(self, y: ReadOnlyVector,
                     ystar: ReadOnlyVector | None = None) -> Functional:
        r"""Return a Quadratic with the Bregman affine correction.
        
        Constraints, extensions, measure, and data metadata are preserved.
        The dual subgradient is divided by the integration measure to obtain
        its pointwise density. Constants are subtracted pointwise, before
        integration, rather than distributed from a global functional value.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        if not self.is_convex:
            return super().bregman_func(y, ystar)
        y, ystar, _ = self._bregman_reference(y, ystar)
        if np.any(self.measure <= 0) or not np.all(np.isfinite(self.measure)):
            raise ValueError('Quadratic.bregman_func requires finite positive integration weights.')
        density = ystar / self.measure
        values = self.domain.empty()
        self._eval(y, func_vals=values, sum_values=False)
        result = self._arithmetic_copy()
        result._b = self._b - density
        result._c = self._c - values + density * y
        result._refresh_quadratic()
        return result


    def _arithmetic_copy(self) -> Quadratic:
        r"""Copy mutable state without cloning the domain or Hilbert geometry.
        
        
        Returns
        -------
        Quadratic
            Constructed functional.
        """
        result = deepcopy(self, {id(self.domain): self.domain,
                                 id(self.h_domain): self.h_domain})
        result.__dict__.pop('conj', None)
        return result

    def _refresh_quadratic(self) -> None:
        self._precompute_f_l()
        self._precompute_f_u()

    def __mul__(self, other: float | Functional) -> Functional | NotImplementedType:
        if np.isscalar(other):
            return self.__rmul__(other)
        return super().__mul__(other)

    def __rmul__(self, other: float | Functional) -> Functional | NotImplementedType:
        r"""Scale coefficients, preserving compatible constraints and tails.
        
        Nonpositive scaling with constraints or extensions uses the generic
        representation, whose extended-real semantics cannot in general be
        represented by the convex quadratic extension machinery.
        
        Parameters
        ----------
        other: float | Functional
            Other.
        
        
        Returns
        -------
        Functional | NotImplementedType
            Result described above.
        """
        if not np.isscalar(other) or not np.isrealobj(other):
            return NotImplemented
        if not np.isfinite(other):
            raise ValueError("Quadratic scaling requires a finite real scalar.")
        if other <= 0 and any(getattr(self, name) is not None for name in (
                'constr_l', 'constr_u', 'lin_taylor_l', 'lin_taylor_u',
                'quad_taylor_l', 'quad_taylor_u')):
            return super().__rmul__(other)
        result = self._arithmetic_copy()
        result._a = other * self._a
        result._b = other * self._b
        result._c = other * self._c
        result._refresh_quadratic()
        return result

    def __imul__(self, other: float | Functional) -> Self:
        r"""Scale in place when the result remains a Quadratic.
        
        Parameters
        ----------
        other: float | Functional
            Other.
        
        
        Returns
        -------
        Self
            Result described above.
        """
        result = self.__rmul__(other)
        if not isinstance(result, Quadratic):
            return result
        # Existing conjugate views must continue to point to this object.
        conjugate = self.__dict__.get('conj')
        self.__dict__.update(result.__dict__)
        if conjugate is not None:
            self.__dict__['conj'] = conjugate
        return self

    def __add__(self, other: float | Functional) -> Functional | NotImplementedType:
        r"""Add a constant, linear functional, or compatible quadratic.
        
        Constants are functional values, so their pointwise density is
        divided by the total measure. A linear functional's dual gradient
        is divided pointwise by the measure. Without extensions, constraint
        intervals are intersected. Matching extensions are preserved;
        incompatible extension joins or empty intersections fall back to
        LinearCombination.
        
        Parameters
        ----------
        other: float | Functional
            Other.
        
        
        Returns
        -------
        Functional | NotImplementedType
            Result described above.
        """
        if np.isscalar(other):
            if not np.isrealobj(other):
                return NotImplemented
            mass = np.sum(self.measure)
            if not np.isfinite(other) or not np.isfinite(mass) or mass <= 0:
                return super().__add__(other)
            result = self._arithmetic_copy()
            result._c = self._c + other / mass
        elif isinstance(other, LinearFunctional):
            if self.domain != other.domain:
                raise ValueError("Functional domains must agree for addition.")
            if np.any(self.measure <= 0) or not np.all(np.isfinite(self.measure)):
                return super().__add__(other)
            result = self._arithmetic_copy()
            result._b = self._b + other.gradient / self.measure
        elif isinstance(other, Quadratic):
            if self.domain != other.domain:
                raise ValueError("Functional domains must agree for addition.")
            names = ('_nat_dom_l', '_nat_dom_u', 'constr_l', 'constr_u',
                     'lin_taylor_l', 'lin_taylor_u', 'quad_taylor_l', 'quad_taylor_u')
            compatible = np.array_equal(self.measure, other.measure)
            for name in names:
                left, right = getattr(self, name), getattr(other, name)
                compatible = compatible and (
                    left is right or (left is not None and right is not None
                                      and np.array_equal(left, right))
                )
            extension_names = ('lin_taylor_l', 'lin_taylor_u',
                               'quad_taylor_l', 'quad_taylor_u')
            without_extensions = all(
                getattr(func, name) is None
                for func in (self, other) for name in extension_names
            )
            intersect = without_extensions and np.array_equal(self.measure, other.measure)
            if intersect:
                lower = np.maximum(self.dom_l, other.dom_l)
                upper = np.minimum(self.dom_u, other.dom_u)
                if np.any(lower > upper):
                    return super().__add__(other)
            elif not compatible:
                return super().__add__(other)
            result = self._arithmetic_copy()
            if intersect:
                result._nat_dom_l = np.maximum(self._nat_dom_l, other._nat_dom_l)
                result._nat_dom_u = np.minimum(self._nat_dom_u, other._nat_dom_u)
                result._constr_l = None if np.all(np.isneginf(lower)) else lower
                result._constr_u = None if np.all(np.isposinf(upper)) else upper
            result._a = self._a + other._a
            result._b = self._b + other._b
            result._c = self._c + other._c
        else:
            return super().__add__(other)
        result._refresh_quadratic()
        return result

    def __radd__(self, other: float | Functional) -> Functional | NotImplementedType:
        return self.__add__(other)

    @property
    def data(self) -> ReadOnlyVector:
        r"""Current data shift, exposed read-only.
        
        
        Returns
        -------
        ReadOnlyVector
            Read-only vector view.
        """
        return self.domain.read_only(self._data)

    @data.setter
    def data(self, value: ReadOnlyVector | None) -> None:
        r"""Data.
        
        Parameters
        ----------
        value: ReadOnlyVector | None
            Input vector; its coefficients are only read.
        """
        if value is not None and value not in self.domain:
            raise ValueError("Quadratic data must be an element of its domain.")
        old = getattr(self, '_data', self.domain.zeros())
        new = self.domain.zeros() if value is None else value.copy()
        delta = new - old
        # Translate the current polynomial by delta, without accumulating
        # repeated assignments of the same data vector.
        b = self._b
        self._c = self._c - b * delta + 0.5 * self._a * delta**2
        self._b = b - self._a * delta
        for name in ('_nat_dom_l', '_nat_dom_u', '_constr_l', '_constr_u',
                     '_lin_taylor_l', '_lin_taylor_u',
                     '_quad_taylor_l', '_quad_taylor_u'):
            bound = getattr(self, name)
            if bound is not None:
                setattr(self, name, bound + delta)
        self._data = new
        self.is_data_func = value is not None
        self._precompute_f_l()
        self._precompute_f_u()

    @data.deleter
    def data(self) -> None:
        r"""Data.
        """
        self.data = None

    def as_data_func(self, data: ReadOnlyVector | None = None) -> "Quadratic":
        r"""Shift by data in place and return this Quadratic instance.
        
        For an unshifted polynomial a*x**2/2+b*x+c, the shifted coefficients
        are b-a*data and c-b*data+a*data**2/2. All constraints and extension
        points are translated by data as well. Subsequent calls replace the
        previous data shift. None preserves an existing shift, or installs
        zero data on a functional that is not yet a data functional.
        Assigning data=None or deleting data removes the shift.
        
        Parameters
        ----------
        data: ReadOnlyVector | None [default: None]
            Observed data.
        
        
        Returns
        -------
        Quadratic
            Constructed functional.
        """
        if data is not None:
            self.data = data
        elif not self.is_data_func:
            self.data = self.domain.zeros()
        return self

    @property
    def a(self) -> np.ndarray:
        r"""A.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        return self._a
    
    @a.setter
    def a(self, value: Scalar | ArrayLike) -> None:
        r"""A.
        
        Parameters
        ----------
        value: Scalar | ArrayLike
            Value.
        """
        if not np.isscalar(value) and value not in self.domain:
            raise ValueError(Errors.value_error(f'a needs to be either scalar or an element of the domain. Given a: {value} of type {type(value)}', self))
        self._a = np.broadcast_to(value, self.domain.shape) if np.isscalar(value) else value    
        self._precompute_f_l(); self._precompute_f_u()
        
    @property
    def b(self) -> np.ndarray:
        r"""B.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        return self._b

    @b.setter
    def b(self, value: Scalar | ArrayLike) -> None: 
        r"""B.
        
        Parameters
        ----------
        value: Scalar | ArrayLike
            Value.
        """
        if not np.isscalar(value) and value not in self.domain:
            raise ValueError(Errors.value_error(f'b needs to be either scalar or an element of the domain. Given b: {value} of type {type(value)}', self))
        self._b = np.broadcast_to(value, self.domain.shape) if np.isscalar(value) else value    
        self._precompute_f_l(); self._precompute_f_u()

    @property
    def c(self) -> np.ndarray:
        r"""C.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        return self._c

    @c.setter
    def c(self, value: Scalar | ArrayLike) -> None:
        r"""C.
        
        Parameters
        ----------
        value: Scalar | ArrayLike
            Value.
        """
        if not np.isscalar(value) and value not in self.domain:
            raise ValueError(Errors.value_error(f'c needs to be either scalar or an element of the domain. Given c: {value} of type {type(value)}', self))
        self._c = np.broadcast_to(value, self.domain.shape) if np.isscalar(value) else value    
        self._precompute_f_l(); self._precompute_f_u()

    @property
    def Lipschitz(self) -> float:
        r"""Lipschitz.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return max([self._Lipschitz_exterior,np.max(np.abs(self._a))]) 
    
    @property
    def convexity_param(self) -> float:
        r"""Convexity param.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return min([self._convexity_param_exterior,np.min(self._a)])
    
    @property
    def is_convex(self) -> bool:
        r"""Whether the functional is convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.convexity_param>=0
    
    @property
    def conj_is_quadratic(self) -> bool:
        r"""Conj is quadratic.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return (np.min(self.a)>0)
     
    @property
    def is_norm_squared_type(self) -> bool:
        r"""Whether the functional is norm squared type.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        a0 = self.a.flat[0]
        return (a0>0 and np.allclose(self.a, a0) and
                all([x is None for x in (self.constr_l, self.lin_taylor_l, self.quad_taylor_l,
                                         self.constr_u, self.lin_taylor_u, self.quad_taylor_u)])             
               )
   
    def _f(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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
    
    def _f_deriv(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = a*v +b
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.multiply(self._a, v, out=out, where=where)
        np.add(out, self._b, out=out, where=where)
        return out
    
    def _f_second_deriv(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.copyto(out, self._a, where=where)
        return out
    
    def _f_prox(self, v: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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
    
    def _f_conj(self, v_star: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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

    def _f_conj_deriv(self, v_star: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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
    
    def _f_conj_second_deriv(self, v_star: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        if np.min(self.a) <= 0:
            raise ValueError(Errors.value_error(
                f'Conjugate of quadratic functional is only quadratic if a is positive. Given a: {self.a}', self
            ))
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        np.divide(1., self._a, out=out, where=where)
        return out
        
    def _f_conj_prox(self, v_star: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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
    domain: MeasureSpaceFcts
        Domain on which to define the generic L1.
    constr_l: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    constr_u: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    lin_taylor_l: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    lin_taylor_u: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    quad_taylor_l: float | ReadOnlyVector | None [default: None]
        Lower threshold for a quadratic extension; None disables this extension.
    quad_taylor_u: float | ReadOnlyVector | None [default: None]
        Upper threshold for a quadratic extension; None disables this extension.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    """
    def __init__(self, domain: MeasureSpaceFcts,
                constr_l: float | ReadOnlyVector | None=None, constr_u: float | ReadOnlyVector | None=None, lin_taylor_l: float | ReadOnlyVector | None=None, lin_taylor_u: float | ReadOnlyVector | None=None,
                quad_taylor_l: float | ReadOnlyVector | None=None, quad_taylor_u: float | ReadOnlyVector | None=None, **kwargs: Any) -> None:

        super().__init__(domain,conj_nat_dom_u=1.,conj_nat_dom_l=-1.,
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                         quad_taylor_l=quad_taylor_l, quad_taylor_u=quad_taylor_u,
                         **kwargs
                         )

    def _f(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.abs(v, out=out, where=where)
        return out

    def _f_deriv(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, v)
        out = self._prepare_out(out, v)
        np.sign(v, out=out, where=where)
        return out

    def _f_second_deriv(self, v: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, v)
        if np.any(where & (v == 0)):
            raise NotTwiceDifferentiableError('L1')
        out = self._prepare_out(out, v)
        np.copyto(out, 0., where=where)
        return out

    def _f_prox(self, v: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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

    def _f_conj(self, v_star: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = 0 if abs(v_star) <= 1, and np.inf otherwise
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        np.abs(v_star, out=out, where=where)
        outside = where & (out > 1 + self._ESS_DOM_TOL)
        np.copyto(out, 0., where=where)
        np.copyto(out, np.inf, where=outside)
        return out
    
    def _f_conj_deriv(self, v_star: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        np.copyto(out, 0., where=where)
        return out

    def _f_conj_second_deriv(self, v_star: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, v_star)
        out = self._prepare_out(out, v_star)
        np.copyto(out, 0., where=where)
        return out

    def _f_conj_prox(self, vstar: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, vstar)
        out = self._prepare_out(out, vstar)
        np.maximum(vstar, -1., out=out, where=where)
        np.minimum(out, 1., out=out, where=where)
        return out

    def _ptw_dist_subdiff(self, vstar: ReadOnlyVector, x: ReadOnlyVector) -> np.ndarray:
        diff = vstar - self.subgradient(x)
        zeroind = (x==0)
        diff[zeroind] = (np.sign(vstar[zeroind])
                         * np.maximum(np.abs(vstar[zeroind])-self.measure[zeroind],0.))
        return diff

    def _conj_ptw_dist_subdiff(self, v: ReadOnlyVector, xstar: ReadOnlyVector) -> np.ndarray:
        w = v.copy()
        ind = np.where((xstar >= self.measure)
                       | self._at_boundary(xstar, self.measure))
        w[ind] = np.minimum(w[ind],0.)
        ind = np.where((xstar <= -self.measure)
                       | self._at_boundary(xstar, -self.measure))
        w[ind] = np.maximum(w[ind],0.)
        return w

class KullbackLeibler(IntegralFunctionalBase):
    r"""Kullback-Leiber divergence defined by
    
    .. math::
        F_w(u) = KL(w,u) = \int (u(x) -w(x) - w(x)\ln \frac{u(x)}{w(x)}) \mathrm{d}x
    
    
    Parameters
    ----------
    domain: MeasureSpaceFcts
        Domain on which to define the Kullback-Leibler divergence
    w: Scalar | ArrayLike | None [default: None]
        First argument of Kullback-Leibler divergence. In the case of None, w is set to zero. If w is scalar, it is broadcasted to the shape of the domain.
    constr_l: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    constr_u: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    lin_taylor_l: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    lin_taylor_u: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    quad_taylor_l: float | ReadOnlyVector | None [default: None]
        Lower threshold for a quadratic extension; None disables this extension.
    quad_taylor_u: float | ReadOnlyVector | None [default: None]
        Upper threshold for a quadratic extension; None disables this extension.
    data: ReadOnlyVector | None [default: None]
        If data is given, then w is w+data; if the parameter w is not given, then w is just the data.
    omit_normalization: bool [default: False]
        If true, the term :math:`w(x)\ln w(x)` is omitted in the definition of F_w. (Then :math:`F_w(w)\neq 0`.)
        This is sometimes done if KullbackLeibler is used as Poisson log-likelihood since the omitted term is not well-defined for 
        a continuous Poisson process, and it is irrelevant and large for large discrete data.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    """

    @property
    def is_strictly_convex(self) -> bool:
        r"""Strict convexity without linear tails; constraints are allowed.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(
            np.all(self.w > 0)
            and self.lin_taylor_l is None
            and self.lin_taylor_u is None
        )

    @property
    def conj_is_strictly_convex(self) -> bool:
        r"""Strict convexity of the conjugate without primal hard constraints.
        
        Primal constraints can generate affine tails in the conjugate.
        Linear primal extensions instead restrict its essential domain.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(
            np.all(self.w > 0)
            and self.constr_l is None
            and self.constr_u is None
        )

    def __init__(self, domain: MeasureSpaceFcts, w: Scalar | ArrayLike | None=None,
                 constr_l: float | ReadOnlyVector | None=None, constr_u: float | ReadOnlyVector | None=None, lin_taylor_l: float | ReadOnlyVector | None=None, lin_taylor_u: float | ReadOnlyVector | None=None,
                 quad_taylor_l: float | ReadOnlyVector | None=None, quad_taylor_u: float | ReadOnlyVector | None=None, data: ReadOnlyVector | None = None,
                 omit_normalization:bool = False,
                 **kwargs: Any) -> None:
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
    def w(self) -> np.ndarray:
        r"""w.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        if self.is_data_func:
            return self._w + self.data
        else:
            return self._w
    @w.setter
    def w(self, new_w: ReadOnlyVector) -> None:
        r"""w -- first (fixed) argument of KL.
        
        Parameters
        ----------
        new_w: ReadOnlyVector
            Input vector; its coefficients are only read.
        """
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
    def data(self) -> ReadOnlyVector:
        r"""Data.
        
        
        Returns
        -------
        ReadOnlyVector
            Read-only vector view.
        """
        return self.domain.read_only(self._data)
    
    @data.setter
    def data(self, new_data: ReadOnlyVector) -> None:
        r"""Data.
        
        Parameters
        ----------
        new_data: ReadOnlyVector
            Input vector; its coefficients are only read.
        """
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
    def data(self) -> None:
        r"""Data.
        """
        if self.is_data_func:
            del self._data
            self.is_data_func = False
            self._precompute_f_l(); self._precompute_f_u()

    def as_data_func(self,data: ReadOnlyVector) -> Functional:
        r"""As data func.
        
        Parameters
        ----------
        data: ReadOnlyVector
            Observed data.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        self.data=data
        return self

    @property
    def Lipschitz(self) -> float:
        r"""Lipschitz.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        if self.constr_u is not None:
            return np.inf
        elif self._taylor_l is not None:
            return  max([self._Lipschitz_exterior,np.max(self.w/self._taylor_l**2) if np.min(self._taylor_l)>0 else np.inf])
        else: 
            return np.inf

    @property
    def convexity_param(self) -> float:
        r"""Convexity param.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        if self.lin_taylor_l is not None or self.lin_taylor_u is not None:
            return 0. 
        elif self.constr_u is not None or self.quad_taylor_u is not None:
            u = self.constr_u if self.constr_u is not None else self.quad_taylor_u
            return np.min([self._convexity_param_exterior,np.min(self.w/u**2) if np.min(u)>0 else np.inf]) 
        else:
            return 0

    def _f(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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
   
    def _f_deriv(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out =  1 - w / u
        if 'w' in kwargs:
            raise ValueError(Errors.value_error(
                'first parameter w of KullbackLeibler must be fixed in constructor.'
            ))
        where = self._where(mask, u)
        wm = self.w
        zero = where & (u == 0)
        if np.any(zero & (wm != 0)):
            raise NotInEssentialDomainError(Errors.value_error(
                'argument cannot be 0 at positions where w is not 0'
            ))
        out = self._prepare_out(out, u)
        regular = where & ~zero
        np.divide(wm, u, out=out, where=regular)
        np.negative(out, out=out, where=regular)
        np.add(out, 1., out=out, where=regular)
        np.copyto(out, 1., where=zero)
        return out

    def _f_second_deriv(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = w / u**2
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.square(u, out=out, where=where)
        np.divide(self.w, out, out=out, where=where)
        return out

    def _f_conj(self, u_star: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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

    def _f_conj_deriv(self, u_star: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = w / (1-u_star)
        if 'w' in kwargs:
            raise ValueError(Errors.value_error(
                'second parameter w of KullbackLeibler must be fixed in constructor.'
            ))
        where = self._where(mask, u_star)
        wm = self.w
        at_one = where & (u_star == 1)
        if np.any(at_one & (wm != 0)):
            raise NotInEssentialDomainError(Errors.value_error(
                'argument cannot be 1 at positions where w is not 0.'
            ))
        out = self._prepare_out(out, u_star)
        regular = where & ~at_one
        np.subtract(1., u_star, out=out, where=regular)
        np.divide(wm, out, out=out, where=regular)
        np.copyto(out, 0., where=at_one)
        return out
    
    def _f_conj_second_deriv(self, u_star: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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

    def _f_prox(self, v: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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

    def _f_conj_prox(self, vstar: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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


class _OffsetKullbackLeibler(HorizontalShiftDilation):
    """Keep data owned by KL even before the first observation is assigned."""

    @property
    def data(self) -> ReadOnlyVector:
        return self.func.data

    @data.setter
    def data(self, value: ReadOnlyVector | None) -> None:
        if value is None:
            del self.data
            return
        self.func.data = self.func._copy_vector(value)
        self._shifted_data_fid = True
        self.is_data_func = True

    @data.deleter
    def data(self) -> None:
        del self.func.data
        self._shifted_data_fid = True
        self.is_data_func = False

    def as_data_func(self, data: ReadOnlyVector | None = None) -> Functional:
        if data is not None:
            self.data = data
        elif not self.is_data_func:
            self.data = self.domain.zeros()
        return self


def KullbackLeiblerOffset(
    domain: MeasureSpaceFcts,
    offset: float | ReadOnlyVector | None = None,
    **kwargs: Any,
) -> Functional:
    r"""Construct the concrete implementation of ``KL(offset=...)``.

    With a positive offset sigma and observations g, the default result is

    .. code-block:: python

        KullbackLeibler(domain, w=sigma, data=g,
                       quad_taylor_l=sigma).shift(-sigma)

    Thus both arguments of KL are offset: ``KL(g + sigma, u + sigma)``.
    The lower quadratic extension starts at ``u=0`` in the final coordinates.
    It allows negative trial values without evaluating a logarithm there.

    Parameters
    ----------
    domain: MeasureSpaceFcts
        Space on which the abstract KL functional is instantiated.
    offset: float | ReadOnlyVector | None [default: None]
        Finite, nonnegative scalar or vector in domain. None or an
        identically zero offset preserves the ordinary KullbackLeibler
        construction. The default quadratic extension requires a strictly
        positive offset at every coordinate.
    **kwargs: Any
        Arguments for KullbackLeibler. An explicit ``w`` is added to the
        offset. ``data`` may be supplied now or later by Setting. Explicit
        lower constraints or linear/quadratic extension settings override
        the default lower quadratic extension. Bounds and extension
        thresholds are in the unshifted argument coordinates ``u + offset``.

    Returns
    -------
    Functional
        KullbackLeibler for an absent/zero offset, otherwise its horizontal
        shift by minus the offset. Later data assignment changes KL's first
        argument, not the horizontal shift. Inputs are copied.

    Examples
    --------
    >>> fidelity = KL(offset=sigma)
    >>> setting = Setting(op, penalty, fidelity, data=data, regpar=alpha)
    """
    if offset is None:
        return KullbackLeibler(domain, **kwargs)
    values = np.asarray(offset)
    if (not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values) or not np.all(np.isfinite(values))
            or np.any(values < 0)):
        raise ValueError('offset must contain finite, nonnegative real values.')
    if values.ndim == 0:
        sigma = float(values) * domain.ones()
    elif offset in domain:
        sigma = domain.empty()
        domain.copyto(sigma, offset)
    else:
        raise ValueError('offset must be a scalar or a vector in domain.')
    if np.all(sigma == 0):
        return KullbackLeibler(domain, **kwargs)

    if ('quad_taylor_l' not in kwargs
            and kwargs.get('lin_taylor_l') is None
            and kwargs.get('constr_l') is None):
        if np.any(sigma == 0):
            raise ValueError(
                'The default quadratic extension requires a strictly positive '
                'offset. Use quad_taylor_l=None to disable that extension.'
            )
        kwargs['quad_taylor_l'] = sigma.copy()
    w = kwargs.pop('w', None)
    weight = sigma if w is None else sigma + w
    data = kwargs.pop('data', None)
    if data is not None:
        if data not in domain:
            raise ValueError('data must belong to domain.')
        copied_data = domain.empty()
        domain.copyto(copied_data, data)
        data = copied_data
    func = KullbackLeibler(domain, w=weight, data=data, **kwargs)
    result = _OffsetKullbackLeibler(func, shift=-sigma)
    # The intrinsic shift never contains observations, even before data exist.
    result._shifted_data_fid = True
    return result


class RelativeEntropy(IntegralFunctionalBase):
    r"""Relative Entropy divergence define by
    
    .. math::
        F_w(u) = KL(u,w) = \int (u(x)\ln \frac{u(x)}{w(x)}) \mathrm{d}x
    
    Parameters
    ----------
    domain: MeasureSpaceFcts
        Domain on which to define the Relative Entropy divergence
    w: Scalar | ArrayLike | None [default: None]
        Fixed part of the second argument of the Relative Entropy divergence.
        If neither ``w`` nor ``data`` is supplied, ``w`` If
        ``data`` is supplied and ``w`` is None, the fixed part
    constr_l: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    constr_u: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    lin_taylor_l: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    lin_taylor_u: float | ReadOnlyVector | None [default: None]
        see IntegralFunctional
    quad_taylor_l: float | ReadOnlyVector | None [default: None]
        Lower threshold for a quadratic extension; None disables this extension.
    quad_taylor_u: float | ReadOnlyVector | None [default: None]
        Upper threshold for a quadratic extension; None disables this extension.
    data: ReadOnlyVector | None [default: None]
        Data-dependent part of the second argument. If supplied, the effective
        reference is ``w + data`` and the instance is marked as a data
        functional.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    """

    def  __init__(self, domain: MeasureSpaceFcts, w: Scalar | ArrayLike | None=None,
                  constr_l: float | ReadOnlyVector | None=None, constr_u: float | ReadOnlyVector | None=None, lin_taylor_l: float | ReadOnlyVector | None=None, lin_taylor_u: float | ReadOnlyVector | None=None,
                  quad_taylor_l: float | ReadOnlyVector | None=None, quad_taylor_u: float | ReadOnlyVector | None=None, data: ReadOnlyVector | None=None,
                  **kwargs: Any
                  ) -> None:
        self._w_is_default = w is None
        if w is None:
            w = 0. if data is not None else 1.
        if np.isscalar(w):
            w = np.broadcast_to(w,domain.shape)
        elif w not in domain:
            raise TypeError(Errors.not_in_vecsp(w,domain,vec_name="second argument",add_info=f" Error occured setting up the Relative Entropy functional."))
        if np.min(w) < 0:
            raise ValueError(Errors.value_error(
                "The fixed part w of RelativeEntropy must be non-negative!",
                self,
            ))
        self._w = w.copy()

        if data is not None:
            if data not in domain:
                raise ValueError(Errors.not_in_vecsp(
                    data, domain, vec_name="data", space_name="domain of functional"
                ))
            self._data = data
            effective_w = self._w + data
        else:
            effective_w = self._w
        if np.min(effective_w) <= 0:
            raise ValueError(Errors.value_error(
                "The second argument w + data of RelativeEntropy must be positive!",
                self,
            ))

        super().__init__(domain,nat_dom_l = 1e-14*effective_w,
                         constr_l=constr_l,constr_u=constr_u,lin_taylor_l=lin_taylor_l,lin_taylor_u=lin_taylor_u,
                         quad_taylor_l=quad_taylor_l,quad_taylor_u=quad_taylor_u,
                         is_data_func=(data is not None),
                         **kwargs
                         )

    @property
    def is_strictly_convex(self) -> bool:
        r"""Strict convexity without linear tails; constraints are allowed.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(
            True
            and self.lin_taylor_l is None
            and self.lin_taylor_u is None
        )

    @property
    def conj_is_strictly_convex(self) -> bool:
        r"""Strict convexity of the conjugate without primal hard constraints.
        
        Primal constraints can generate affine tails in the conjugate.
        Linear primal extensions instead restrict its essential domain.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(
            np.all(self.w > 0)
            and self.constr_l is None
            and self.constr_u is None
        )

    @property
    def w(self) -> np.ndarray:
        r"""W.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        if self.is_data_func:
            return self._w + self.data
        return self._w

    @w.setter
    def w(self, new_w: ReadOnlyVector) -> None:
        r"""w -- the reference value or second (fixed) argument of KL. 
        
        Parameters
        ----------
        new_w: ReadOnlyVector
            Input vector; its coefficients are only read.
        """
        if new_w in self.domain or (isinstance(new_w, (int,float,np.floating,np.integer))): 
            if new_w in self.domain:
                candidate = new_w.copy()
            else:
                candidate = np.broadcast_to(
                    float(new_w), self.domain.shape
                ).copy()
            if np.min(candidate) < 0:
                raise ValueError(Errors.value_error(
                    'The fixed part w must be non-negative.'
                ))
            effective_w = (
                candidate + self.data if self.is_data_func else candidate
            )
            if np.min(effective_w) <= 0:
                raise ValueError(Errors.value_error(
                    'The second argument w + data must be positive.'
                ))
            self._w = candidate
            self._w_is_default = False
            self._precompute_f_l(); self._precompute_f_u()
        else:
            raise ValueError(Errors.not_in_vecsp(new_w,self.domain,vec_name="new w vector",space_name="domain of functional"))

    @property
    def data(self) -> ReadOnlyVector:
        r"""Data.
        
        
        Returns
        -------
        ReadOnlyVector
            Read-only vector view.
        """
        return self.domain.read_only(self._data)

    @data.setter
    def data(self, new_data: ReadOnlyVector) -> None:
        r"""Data.
        
        Parameters
        ----------
        new_data: ReadOnlyVector
            Input vector; its coefficients are only read.
        """
        if new_data is None:
            del self.data
        elif new_data in self.domain:
            fixed_w = (
                np.zeros(self.domain.shape, dtype=self._w.dtype)
                if self._w_is_default and not self.is_data_func
                else self._w
            )
            effective_w = fixed_w + new_data
            if np.min(effective_w) <= 0:
                raise ValueError(Errors.value_error(
                    'The second argument w + data must be positive.'
                ))
            self._w = fixed_w
            self._data = new_data
            self.is_data_func = True
            self._precompute_f_l(); self._precompute_f_u()
        else:
            raise ValueError(Errors.not_in_vecsp(
                new_data, self.domain, vec_name="new data vector",
                space_name="domain of functional"
            ))

    @data.deleter
    def data(self) -> None:
        r"""Data.
        """
        if self.is_data_func:
            if self._w_is_default:
                self._w = np.ones(
                    self.domain.shape, dtype=self._w.dtype
                )
            elif np.min(self._w) <= 0:
                raise ValueError(Errors.value_error(
                    'Cannot remove data while the fixed part w is not positive.'
                ))
            del self._data
            self.is_data_func = False
            self._precompute_f_l(); self._precompute_f_u()

    def as_data_func(self, data: ReadOnlyVector) -> Self:
        r"""As data func.
        
        Parameters
        ----------
        data: ReadOnlyVector
            Observed data.
        
        
        Returns
        -------
        Self
            Result described above.
        """
        self.data = data
        return self

    @property
    def Lipschitz(self) -> float:
        r"""Lipschitz.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        if self.constr_u is not None:
            return np.inf
        elif self._taylor_l is not None:
            return np.max([self._Lipschitz_exterior,np.max(1/self._taylor_l) if np.min(self._taylor_l)>0 else np.inf])
        else: 
            return np.inf

    @property
    def convexity_param(self) -> float:
        r"""Convexity param.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        if self.lin_taylor_u is not None or self.lin_taylor_l is not None:
            return 0.
        elif self.constr_u is not None or self.quad_taylor_u is not None:
            u = self.constr_u if self.constr_u is not None else self.quad_taylor_u
            return np.min([self._convexity_param_exterior,np.min(1./u) if np.min(u)>0 else np.inf]) 
        else:
            return 0.

    def _f(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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
   
    def _f_deriv(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = 1 + log(u / w)
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.divide(u, self.w, out=out, where=where)
        # At zero use the limiting derivative, not a finite subgradient.
        zero = where & (out == 0)
        np.log(out, out=out, where=where & ~zero)
        np.copyto(out, -np.inf, where=zero)
        np.add(out, 1., out=out, where=where)
        return out

    def _f_second_deriv(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = 1 / u
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.divide(1., u, out=out, where=where)
        return out

    def _f_prox(self, v: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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

    def _f_conj_prox(self, vstar: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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

    def _f_conj(self, u_star: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = w * (exp(u_star - 1))
        where = self._where(mask, u_star)
        out = self._prepare_out(out, u_star)
        np.subtract(u_star, 1., out=out, where=where)
        np.exp(out, out=out, where=where)
        np.multiply(out, self.w, out=out, where=where)
        return out

    def _f_conj_deriv(self, u_star: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        return self._f_conj(u_star, out=out, mask=mask, **kwargs)
    
    def _f_conj_second_deriv(self, u_star: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        return self._f_conj(u_star, out=out, mask=mask, **kwargs)

def Huber(domain: MeasureSpaceFcts, as_primal: bool=True, sigma: float=1., L1_type: bool=False, **kwargs: Any) -> Quadratic:
    r"""Return the Huber functional as a Quadratic with linear tails.
    
    Its pointwise value is x**2/2 for abs(x)<=sigma and
    sigma*abs(x)-sigma**2/2 otherwise. The signature and normalization
    match Huber_old when L1_type=False. When L1_type=True, the
    integrand is divided pointwise by sigma, so its linear tails have
    slopes -1 and +1. sigma may be a positive scalar or domain element.
    as_primal=False uses the reciprocal measure, as in the legacy class.
    Data shifts preserve the Quadratic type and translate the tail joins.
    
    Parameters
    ----------
    domain: MeasureSpaceFcts
        Domain vector space.
    as_primal: bool [default: True]
        As primal.
    sigma: float [default: 1.0]
        Positive Huber transition threshold.
    L1_type: bool [default: False]
        If True, divide the pointwise Huber integrand by sigma so that
        the linear tails have slopes -1 and +1.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    
    
    Returns
    -------
    Quadratic
        Constructed functional.
    """
    if not isinstance(domain, MeasureSpaceFcts):
        raise TypeError("Huber requires a MeasureSpaceFcts domain.")
    if not np.isscalar(sigma) and sigma not in domain:
        raise TypeError("sigma must be a scalar or an element of the domain.")
    if np.iscomplexobj(sigma) or np.any(~np.isfinite(sigma)) or np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive, finite and real.")
    sigma = np.array(np.broadcast_to(sigma, domain.shape), dtype=float, copy=True)
    if not as_primal:
        domain = deepcopy(domain)
        domain.measure = 1. / domain.measure
    # Scaling the quadratic coefficient scales its linear Taylor tails too.
    # This also supports spatially varying sigma without changing the measure.
    if L1_type:
        if "a" in kwargs:
            raise ValueError("Huber(L1_type=True) does not accept an explicit a coefficient.")
        kwargs["a"] = 1. / sigma
    functional = Quadratic(domain, lin_taylor_l=-sigma, lin_taylor_u=sigma, **kwargs)
    functional.sigma = sigma
    return functional


class Huber_old(IntegralFunctionalBase):
    r"""Huber functional 
    
    .. math::
        F(x) = 1/2 |x|^2                &\quadif  |x|\leq \sigma \\
        F(x) = \sigma |x|-\sigma^2/2    &\quadif  |x|>\sigma
    
    
    Parameters
    ----------
    domain: MeasureSpaceFcts
        domain on which Huber functional is defined
    as_primal: bool [default: True]
        If False, then the functional is initiated as conjugate of QuadraticIntv. Then the dual metric is used, 
        and precautions against an infinite recursion of conjugations are taken.
    sigma: float [default: 1.0]
        parameter in the Huber functional.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    """
    # Keep the historical constructor and attributes; use the integral
    # functional's native constraints so SemiSmoothNewton can remove them
    # on its private copy and evaluate the smooth quadratic everywhere.

    def  __init__(self, domain: MeasureSpaceFcts, as_primal: bool=True, sigma: float = 1., **kwargs: Any) -> None:
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
            self.conjugate = QuadraticIntv(domain,as_primal=False,sigma=sigma)
        else:
            dual_domain = deepcopy(domain)
            dual_domain.measure = 1./domain.measure
            super().__init__(dual_domain, Lipschitz=1., **kwargs)
        # auxiliary vectors
        self._abs_u = self.domain.zeros() 
        self._small = np.zeros(self.domain.shape,dtype=bool)

    def _f(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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

    def _f_deriv(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = u                      if abs(u) <= sigma
        # out = sigma * sign(u)        else
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.maximum(u, -self.sigma, out=out, where=where)
        np.minimum(out, self.sigma, out=out, where=where)
        return out

    def _f_second_deriv(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = 1   if abs(u) <= sigma
        # out = 0   else
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        inside = np.abs(u) <= self.sigma
        np.copyto(out, inside, where=where, casting='unsafe')
        return out

    def _f_conj(self, ustar: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        return self.conjugate._f(ustar, out=out, mask=mask, **kwargs)
   
    def _f_conj_deriv(self, ustar: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        return self.conjugate._f_deriv(ustar, out=out, mask=mask, **kwargs)

    def _f_conj_second_deriv(self, ustar: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        return self.conjugate._f_second_deriv(ustar, out=out, mask=mask, **kwargs)

    def _f_conj_prox(self, ustar: ReadOnlyVector, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        return self.conjugate._f_prox(ustar, tau, out=out, mask=mask, **kwargs)


class QuadraticIntv(IntegralFunctionalBase):
    r"""Functional 
    
    .. math::
        F(x) = 1/2 |x|^2    &\quadif |x|\leq \sigma(x) \\
        F(x) = \infty    &\quadif |x|>\sigma(x)
    
    
    Parameters
    ----------
    domain: MeasureSpaceFcts
        Domain vector space.
    as_primal: bool [default: True]
        If False, then the functional is initiated as conjugate of Huber. Then the dual metric is used, 
        and precautions against an infinite recursion are taken.
    sigma: float [default: 1.0]
        interval width.
    eps: float [default: 1e-10]
        Eps.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    """
    # Keep the historical constructor and attributes; use the integral
    # functional's native constraints so SemiSmoothNewton can remove them
    # on its private copy and evaluate the smooth quadratic everywhere.

    def  __init__(self, domain: MeasureSpaceFcts, as_primal: bool=True, sigma: float=1., eps: float=1e-10, **kwargs: Any) -> None:
        if not isinstance(domain,MeasureSpaceFcts): 
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,"QuadraticIntv domain needs to be a MeasureSpaceFcts instance."))
        if not isinstance(sigma, (float,int)) and sigma not in domain:
            raise TypeError(Errors.type_error(f"Sigma in the QuadraticIntv needs to be a scalar or element in the domain! Given:"+"\n\t "+f"sigma = {sigma}",self))
        if np.min(sigma)<=0:
            raise ValueError(Errors.value_error("The interval width in the QuadraticIntv needs to be positive!", self))
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
            self.conjugate = Huber_old(domain,as_primal=False,sigma=sigma)
        else:
            dual_domain = deepcopy(domain)
            dual_domain.measure = 1./domain.measure
            super().__init__(dual_domain, convexity_param=1,                                                     
                         **kwargs)   

    def _f(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        # out = 0.5 * abs(u)**2    if abs(u)<=sigmaeps
        # out = np.inf             else
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        outside = where & (np.abs(u) > self.sigmaeps)
        np.square(u, out=out, where=where)
        np.multiply(out, 0.5, out=out, where=where)
        np.copyto(out, np.inf, where=outside)
        return out
   
    def _f_deriv(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.copyto(out, u, where=where)
        return out

    def _f_prox(self, u: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
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

    def _f_second_deriv(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.copyto(out, 1., where=where)
        return out

    def _f_conj(self, ustar: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        return self.conjugate._f(ustar, out=out, mask=mask, **kwargs)
   
    def _f_conj_deriv(self, ustar: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        return self.conjugate._f_deriv(ustar, out=out, mask=mask, **kwargs)

    def _f_conj_second_deriv(self, ustar: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        return self.conjugate._f_second_deriv(ustar, out=out, mask=mask, **kwargs)

    def _f_conj_prox(self, ustar: ReadOnlyVector, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        return self.conjugate._f_prox(ustar, tau, out=out, mask=mask, **kwargs)
    
    def _ptw_dist_subdiff(self, vstar: ReadOnlyVector, x: ReadOnlyVector) -> np.ndarray:
        diff = vstar.copy()
        diff -= self.subgradient(x)
        ubind = (x==self.sigma)
        diff[ubind] = np.minimum(diff[ubind],0.)
        lbind = (x==-self.sigma)
        diff[lbind] = np.maximum(diff[lbind],0.)
        return diff


class QuadraticNonneg(IntegralFunctionalBase):
    r"""Functional 
    
    .. math::
        F(x) = 1/2 |x|^2    &\quadif x\geq 0 \\
        F(x) = \infty       &\quadif  x<0
    
    Parameters
    ----------
    domain: MeasureSpaceFcts
        domain on which functional is defined
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    """
    # Keep the historical constructor and attributes; use the integral
    # functional's native constraints so SemiSmoothNewton can remove them
    # on its private copy and evaluate the smooth quadratic everywhere.

    def  __init__(self, domain: MeasureSpaceFcts, **kwargs: Any) -> None:
        super().__init__(domain,convexity_param = 1., nat_dom_l=0.,
                         methods =  {'eval', 'subgradient', 'hessian', 'proximal', 'dist_subdiff'},
                         conj_methods =  {'eval', 'subgradient', 'hessian', 'proximal', 'dist_subdiff'},
                         **kwargs)

    def _f(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        negative = where & (u < 0)
        np.square(u, out=out, where=where)
        np.multiply(out, 0.5, out=out, where=where)
        np.copyto(out, np.inf, where=negative)
        return out

    def _f_deriv(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.copyto(out, u, where=where)
        return out

    def _f_prox(self, u: np.ndarray, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.divide(u, 1. + tau, out=out, where=where)
        np.maximum(out, 0., out=out, where=where)
        return out

    def _f_second_deriv(self, u: np.ndarray, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, u)
        out = self._prepare_out(out, u)
        np.copyto(out, 1., where=where)
        return out

    def _f_conj(self, ustar: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, ustar)
        out = self._prepare_out(out, ustar)
        negative = where & (ustar < 0)
        np.square(ustar, out=out, where=where)
        np.multiply(out, 0.5, out=out, where=where)
        np.copyto(out, 0., where=negative)
        return out

    def _f_conj_deriv(self, ustar: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, ustar)
        out = self._prepare_out(out, ustar)
        np.copyto(out, ustar, where=where)
        np.copyto(out, 0., where=where & (ustar < 0))
        return out

    def _f_conj_second_deriv(self, ustar: ReadOnlyVector, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, ustar)
        out = self._prepare_out(out, ustar)
        nonnegative = ustar >= 0
        np.copyto(out, nonnegative, where=where, casting='unsafe')
        return out

    def _f_conj_prox(self, ustar: ReadOnlyVector, tau: float, out: np.ndarray | None=None, mask: Mask | None=None, **kwargs: Any) -> np.ndarray:
        where = self._where(mask, ustar)
        out = self._prepare_out(out, ustar)
        np.copyto(out, ustar, where=where)
        positive = where & (ustar > 0)
        np.divide(ustar, 1. + tau, out=out, where=positive)
        return out
    
    def _ptw_dist_subdiff(self, vstar: ReadOnlyVector, x: ReadOnlyVector) -> np.ndarray:
        r"""Dual residual to the subdifferential; integrated by dist_subdiff.
        
        At x>0 the subgradient is measure*x. At x=0 the
        subdifferential is the negative half-line. Negative x is infeasible.
        
        Parameters
        ----------
        vstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        if np.any(x < 0):
            raise NotInEssentialDomainError(
                "QuadraticNonneg requires nonnegative x."
            )
        residual = vstar - self.measure * x
        return np.where(x == 0, np.maximum(residual, 0.), residual)

    def _conj_ptw_dist_subdiff(self, v: ReadOnlyVector, xstar: ReadOnlyVector, **kwargs: Any) -> np.ndarray:
        r"""The conjugate is differentiable, with gradient max(xstar/w, 0).
        
        Parameters
        ----------
        v: ReadOnlyVector
            Input vector; its coefficients are only read.
        xstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        return v - np.maximum(xstar / self.measure, 0.)


def QuadraticBilateralConstraints(
    domain: MeasureSpaceFcts, lb: float | ReadOnlyVector | None=None, ub: float | ReadOnlyVector | None=None, x0: ReadOnlyVector | None=None, alpha: float = 1.,
    **kwargs: Any,
) -> Quadratic:
    r"""Returns `Functional` defined by 
    
    .. math::
        F(x) = \frac{\alpha}{2}\|x-x_0\|^2  &\quadif lb\leq x\leq ub \\
        F(x) = np.inf &\quadelse
    
    
    Parameters
    ----------
    domain: MeasureSpaceFcts
        domain on which functional is defined
    lb: float | ReadOnlyVector | None [default: None]
        lower bound, default:0
    ub: float | ReadOnlyVector | None [default: None]
        upper bound, default:1
    x0: ReadOnlyVector | None [default: None]
        reference value.5*(lb+ub)
    alpha: float [default: 1.0]
        regularization parameter
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    
    Returns
    -------
    Quadratic
        Quadratic integral functional with bilateral constraints.
    
    Notes
    -----
    Returns a Quadratic instance with coefficients a=alpha, b=-alpha*x0,
    and c=alpha*x0**2/2. The bounds are represented by constr_l and constr_u.
    Thus it is also an IntegralFunctionalBase, suitable for SemiSmoothNewton,
    which handles the bounds separately from the smooth quadratic.
    """

    if not isinstance(domain,MeasureSpaceFcts): raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,"QuadraticBilateralConstraints domain needs to be a MeasureSpaceFcts instance."))
    if isinstance(lb,(float,int)):
        lb = lb*domain.ones(read_only=True)
    elif lb is None:
        lb = domain.zeros(read_only=True)
    elif lb not in domain:
        raise TypeError(Errors.type_error(f"Lower bound lb in QuadraticBilateralConstraints needs to be a scalar or element in the domain! Given:"+"\n\t "+f"lb = {lb}"))
    if isinstance(ub,(float,int)):
        ub = ub*domain.ones(read_only=True)
    elif ub is None:
        ub = domain.ones(read_only=True)
    elif ub not in domain:
        raise TypeError(Errors.type_error(f"Upper bound ub in QuadraticBilateralConstraints needs to be a scalar or element in the domain! Given:"+"\n\t "+f"ub = {ub}"))
    if np.any(lb>=ub):
        raise TypeError(Errors.type_error(f"The lower bound needs to be below the upper bound in QuadraticBilateralCOnstraint:"+"\n\t "+f"lb = {lb},"+"\n\t "+f"ub = {ub}"))
    if x0 is None:
        x0 =0.5*(lb+ub)
    elif isinstance(x0,(float,int)):
        x0 = x0*domain.ones(read_only=True)
    elif x0 not in domain:
        raise TypeError(Errors.type_error(f"Reference value x0 in QuadraticBilateralConstraints needs to be a scalar or element in the domain! Given:"+"\n\t "+f"x0 = {x0}"))

    if not isinstance(alpha,(float,int)):
        raise TypeError(Errors.not_instance(alpha,float,add_info="Regularization parameter alpha in QuadraticBilateralConstraints needs to be a scalar!"))

    # Install coefficients before constraints: boundary precomputation
    # evaluates the quadratic and its derivative.
    functional = Quadratic(
        domain, a=alpha, b=-alpha*x0, c=0.5*alpha*x0**2, **kwargs
    )
    functional.constr_l = lb
    functional.constr_u = ub

    # Preserve the attributes exposed by the previous implementation.
    functional.lb, functional.ub = lb, ub
    functional.x0, functional.alpha = x0, alpha
    return functional

def QuadraticLowerBound(domain: MeasureSpaceFcts, lb: float | ReadOnlyVector | None=None, x0: ReadOnlyVector | None=None, a: float | int=1.) -> Quadratic:
    r"""Returns `Functional` defined by 
    
    .. math::
        F(x) = \frac{a}{2}\|x-x_0\|^2 &\quad\text{if } lb\leq x \\
        F(x) =  \inf  &\quad\text{else }
    
    
    Parameters
    ----------
    domain: MeasureSpaceFcts
        domain on which the functional is defined
    lb: float | ReadOnlyVector | None [default: None]
        lower bound (zero in the default case)
    x0: ReadOnlyVector | None [default: None]
        reference value (zero in the default case)
    a: float | int [default: 1.0]
        scaling factor
    
    Returns
    -------
    Quadratic
        Quadratic integral functional with coefficients a, -a*x0,
        a*x0**2/2 and native lower constraint constr_l=lb.
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

    # Coefficients must exist before constraint boundary precomputation.
    functional = Quadratic(domain, a=a, b=-a*x0, c=0.5*a*x0**2)
    functional.constr_l = lb
    return functional


class L1Generic(Functional):
    r"""Generic :math:`L ^1` Functional. Proximal implemented for default :math:`L^2` as `h_domain`.
    
    Parameters
    ----------
    domain: MeasureSpaceFcts
        Domain on which to define the generic L1.
    """
    def __init__(self, domain: MeasureSpaceFcts) -> None:
        if not isinstance(domain,NumPyVectorSpace):
            raise TypeError(Errors.not_instance(domain,NumPyVectorSpace,"To construct a L1Generic functional you need a NumPyVectorSpace"))
        super().__init__(domain,
                         methods = {'eval','subgradient','hessian','proximal'})

    def _eval(self, x: ReadOnlyVector) -> float:
        return np.sum(np.abs(x))

    def _subgradient(self, x: ReadOnlyVector, out: Vector | None=None, **subgradient_par: Any) -> Vector:
        if out is None:
            out = np.empty_like(x)
        np.sign(x, out=out)
        return out

    def _hessian(self, x: ReadOnlyVector) -> Operator:
        # Even approximate Hessians don't work here.
        raise NotImplementedError

    def _proximal(self, x: ReadOnlyVector, tau: float, out: Vector | None=None, **proximal_par: Any) -> Vector:
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
