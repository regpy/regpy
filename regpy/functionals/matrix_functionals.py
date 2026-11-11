from __future__ import annotations

from math import inf
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from regpy.vecsps.base import ReadOnlyVector, Vector, VectorSpaceBase
    from regpy.operators.base import Operator
    from regpy._typing import Mask
import numpy as np
from scipy.linalg import ishermitian

from regpy.util import Errors, memoized_property
from regpy.vecsps.numpy import *

from .base import Functional, NotInEssentialDomainError

__all__ = ["QuadraticPositiveSemidef", "SpectralFunctional"]


class SpectralFunctional(Functional):

    def __init__(self, domain,spec_func, h_domain = None):
        if not isinstance(domain,UniformGridFcts):
            raise TypeError(Errors.not_instance(domain,UniformGridFcts,add_info="SpectralFunctionals are only defined for matrices."))
        if not domain.ndim==2 or not domain.shape[0]==domain.shape[1]:
            raise RuntimeError("SpectralFunctionals are only defined for square matrices.")
        if not isinstance(spec_func.domain,UniformGridFcts) or spec_func.domain.ndim!=1:
            raise RuntimeError("Spectral functional has to be defined on one dimensional uniform domain.")
        if spec_func.domain.shape[0]!=domain.shape[0]:
            raise RuntimeError(f"Size {domain.shape[0]} of matrices is incompatible with size {spec_func.domain.shape} of domain of functional.")
        self.spec_func=spec_func
        all_methods = {'eval', 'subgradient', 'proximal'}
        methods=all_methods.intersection(spec_func.methods)
        conj_methods=all_methods.intersection(spec_func._conj_methods)
        super().__init__(domain,h_domain,is_linear=spec_func.is_linear,is_quadratic=spec_func.is_quadratic,is_convex=spec_func.is_convex,
                         convexity_param=spec_func.convexity_param,Lipschitz=spec_func.Lipschitz,methods=methods,conj_methods=conj_methods)


    def _eval(self, x):
        evs= np.linalg.eigvalsh(x)
        return self.spec_func(evs)

    def _subgradient(self, x, out = None):
        evs, U = np.linalg.eigh(x)
        grad=(U * self.spec_func.subgradient(evs)) @ np.conj(U).T
        if(out is None):
            return grad
        else:
            np.copyto(out,grad)
            return out

    def _proximal(self, x, tau, out = None):
        evs, U = np.linalg.eigh(x)
        prox=(U * self.spec_func.proximal(evs,tau)) @ np.conj(U).T
        if(out is None):
            return prox
        else:
            np.copyto(out,prox)
            return out

    def _conj(self, xstar):
        evs = np.linalg.eigvalsh(xstar)
        return self.spec_func.conj(evs)

    def _conj_subgradient(self, xstar, out = None):
        evs, U = np.linalg.eigh(xstar)
        grad=(U * self.spec_func.conj.subgradient(evs)) @ np.conj(U).T
        if(out is None):
            return grad
        else:
            np.copyto(out,grad)
            return out

    def _conj_proximal(self, xstar, tau, out = None):
        evs, U = np.linalg.eigh(xstar)
        prox=(U * self.spec_func.conj.proximal(evs,tau)) @ np.conj(U).T
        if(out is None):
            return prox
        else:
            np.copyto(out,prox)
        return out




class QuadraticPositiveSemidef(Functional):
    r"""Functional 
    
    .. math::
        F(x) = 1/2 ||x||_{HS}^2    &\quad\text{if } x\geq 0 \text{ and (optional) } tr(x)=c \\
        F(x) = \infty       &\quad\text{else}
    
    Here x is a quadratic matrix and HS is the Hilbert-Schmidt norm. Conjugate functional
    and prox are only correct for hermitian inputs.
    
    Parameters
    ----------
    domain: VectorSpaceBase
        two dimensional domain on which functional is defined, volume_elements have to be one
    trace_val: float | None [default: None]
        desired value of trace or None for no trace constraint.
    tol: float [default: 1e-15]
        tolerance for comparisons determining positive semidefiniteness and correctness of trace
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    """

    def  __init__(self, domain: VectorSpaceBase,trace_val: float | None=None,tol: float=1e-15, **kwargs: Any) -> None:
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

    def is_in_essential_domain(self, rho: ReadOnlyVector) -> bool:
        r"""Is in essential domain.
        
        Parameters
        ----------
        rho: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        if(not ishermitian(rho,atol=self.tol)):
            return False
        if(self.has_trace_constraint):
            if(np.abs(np.trace(rho)-self.trace_val)>self.tol):
                return False
        evs=np.linalg.eigvalsh(rho)
        return evs[0]>-self.tol
    
    @staticmethod
    def closest_point_simplex(p: np.ndarray, a: float) -> np.ndarray:
        r"""Algorithm from Held, Wolfe and Crowder (1974) to project onto simplex :math:`\{q:q_{i}\qeq 0,\sum q_{i}=a\}`.
        It uses that p is already sorted in increasing order.
        
        Parameters
        ----------
        p: np.ndarray
            Input point sorted in increasing order
        a: float
            positive value that is the sum of the elements in the result
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        p_flipped=np.flip(p)
        comp_vals=(np.cumsum(p_flipped)-a)/np.arange(1,p.shape[0]+1)
        k=np.where(comp_vals<p_flipped)[0][-1]
        t=comp_vals[k]
        return np.maximum(p-t,0)
        

    def _eval(self, x: ReadOnlyVector) -> float:
        if(self.is_in_essential_domain(x)):
            return np.sum(np.abs(x)**2)/2
        else:
            return np.inf

    def _proximal(self, x: ReadOnlyVector, tau: float, out: Vector | None=None, mask: Mask | None=None, **proximal_par: Any) -> Vector:
        evs, U = np.linalg.eigh(x)
        evs /= 1 + tau
        if self.has_trace_constraint:
            proj_evs = QuadraticPositiveSemidef.closest_point_simplex(evs, self.trace_val)
        else:
            proj_evs = np.maximum(0, evs)
        result = (U * proj_evs) @ np.conj(U).T
        if out is None:
            out = np.empty_like(result)
        where = True if mask is None else np.asarray(mask, dtype=bool)
        np.copyto(out, result, where=where)
        return out
        
    def _subgradient(self, x: ReadOnlyVector, out: Vector | None=None, **subgradient_par: Any) -> Vector:
        if self.is_in_essential_domain(x):
            if out is None:
                return np.copy(x)
            np.copyto(out, x)
            return out
        raise NotInEssentialDomainError
        
    def _hessian(self, x: ReadOnlyVector) -> Operator:
        if(self.is_in_essential_domain(x)):
            return self.domain.identity
        else:
            return NotInEssentialDomainError
        
    def _conj(self, xstar: ReadOnlyVector) -> float:
        evs=np.linalg.eigvalsh(xstar)
        if(self.has_trace_constraint):
            cps=QuadraticPositiveSemidef.closest_point_simplex(evs,self.trace_val)
            return (np.sum(evs**2)+np.sum((cps-evs)**2))/2
        else:
            return (np.sum(evs**2)+np.sum(evs**2,where=evs<0))/2
        