'''Special NGSolve functionals defined on the `regpy.vecsps.ngsolve.NgsVectorSpace`. 
'''
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from regpy.vecsps.base import ReadOnlyVector, Vector, VectorSpaceBase
from math import inf, sqrt

import ngsolve as ngs

from regpy.util import Errors
from regpy.operators.ngsolve import NgsGradOP
from regpy.vecsps.ngsolve import NgsVectorSpace

from .base import Functional

__all__ = ["SignumFilter", "NgsL1", "NgsTV"]

class SignumFilter(ngs.la.BaseMatrix):
    r"""SignumFilter interface.
    
    Parameters
    ----------
    space: VectorSpaceBase
        Space.
    vec: ReadOnlyVector
        Input vector; its coefficients are only read.
    """
    def __init__ (self, space: VectorSpaceBase, vec: ReadOnlyVector) -> None:
        super().__init__()
        self.gf = ngs.GridFunction(space)
        self.gf.vec.data = vec
        self.gf_out = ngs.GridFunction(space)
        self.gf_help = ngs.GridFunction(space)
    
    def Update(self, new_vec: ngs.BaseVector) -> None:
        r"""Update.
        
        Parameters
        ----------
        new_vec: ngs.BaseVector
            New vec.
        """
        self.gf.vec.data = new_vec
    
    def Mult (self, x: ReadOnlyVector, y: Vector) -> None:
        r"""Mult.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        y: Vector
            Writable vector used by this operation.
        """
        self.gf_help.vec.data = x
        self.gf_out.Interpolate(ngs.IfPos(self.gf,1,-1)*self.gf_help)
        y.data = self.gf_out.vec
    
    def Height (self) -> int:
        r"""Height.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        return self.space.ndof
    
    def Width (self) -> int:
        r"""Width.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        return self.space.ndof


class NgsL1(Functional):
    r"""Implementation of the :math:`L^1`-norm on a given `NgsVectorSpace`. It is registered under the
    Abstract functional `L1` and should not be called directly but rather used by defining the 
    abstract `L1` functional as the `penalty` or `data_fid` when initializing the regularization
    setting by calling `regpy.solvers.Setting`.
    
    Parameters
    ----------
    domain: VectorSpaceBase
        The underlying `ngsolve` space.
    """
    def __init__(self, domain: VectorSpaceBase) -> None:
        if not isinstance(domain, NgsVectorSpace):
            raise TypeError(Errors.not_instance(domain, NgsVectorSpace,"AN NgsL1 functional is only defined on an NgsVectorSpace."))
        if domain.codim != 1:
            raise ValueError(Errors.value_error("NgsL1 is only implemented for scalar valued spaces.",self))
        self._gfu = ngs.GridFunction(domain.fes)
        self._w_help = domain.empty()
        self.sign = SignumFilter(domain.fes,self._gfu.vec)
        super().__init__(domain,methods={'eval','subgradient','proximal'})

    def _eval(self, x: ReadOnlyVector) -> float:
        return ngs.Integrate( ngs.Norm(self.domain.to_gf(x)), self.domain.fes.mesh )

    def _subgradient(self, x: ReadOnlyVector, out: Vector | None=None, **subgradient_par: Any) -> Vector:
        self.sign.Update(x.vec)
        target = self._w_help if out is None else out
        target.vec.data = self.sign * x.vec
        return target.copy() if out is None else target

    def _proximal(self, x: ReadOnlyVector, tau: float, out: Vector | None=None, **proximal_par: Any) -> Vector:
        self.sign.Update(x.vec)
        self._gfu.vec.data = self.sign * x.vec
        self._gfu.Interpolate(ngs.IfPos(self._gfu - tau, 1, 0) * (self._gfu - tau))
        target = self._w_help if out is None else out
        target.vec.data = self.sign * self._gfu.vec
        return target.copy() if out is None else target

class NgsTV(Functional):
    r"""Implementation of the total variation functional :math:`TV` on a given `NgsVectorSpace`. It is 
    registered under the Abstract functional `TV` and should not be called directly but rather 
    used by defining the abstract `TV` functional as the `penalty` or `data_fid` when initializing 
    the regularization setting by calling `regpy.solvers.Setting`.
    
    Parameters
    ----------
    domain: VectorSpaceBase
        The underlying `ngsolve` space.
    """

    def __init__(self, domain: VectorSpaceBase) -> None:
        if not isinstance(domain, NgsVectorSpace):
            raise TypeError(Errors.not_instance(domain, NgsVectorSpace,"AN NgsL1 functional is only defined on an NgsVectorSpace."))
        if domain.codim != 1:
            raise ValueError(Errors.value_error("NgsL1 is only implemented for scalar valued spaces.",self))
        if isinstance(domain.fes,ngs.H1):
            self.vec_fes = ngs.VectorH1(domain.fes.mesh, dirichlet=domain.bdr if domain.bdr is not None else "", order=domain.fes.globalorder)
        elif isinstance(domain.fes,ngs.L2):
            self.vec_fes = ngs.VectorL2(domain.fes.mesh, order=domain.fes.globalorder, dirichlet=domain.bdr if domain.bdr is not None else "")
        else:
            raise ValueError(Errors.value_error("NgsTV is only implemented for H1 or L2 finite element spaces.",self))
        super().__init__(domain,methods={"eval","proximal"})
        self._gfu = ngs.GridFunction(self.domain.fes)

        self._grad_op = NgsGradOP(self.domain)

    def _eval(self, x: ReadOnlyVector) -> float:
        self._gfu.vec.data = x.vec
        gradu = ngs.grad(self._gfu)
        tvnorm = 0
        for i in range(gradu.dim):
            tvnorm += ngs.InnerProduct(gradu[i],gradu[i])
        return sqrt(ngs.Integrate(tvnorm, self.domain.fes.mesh))

    def _proximal(self, x: ReadOnlyVector, tau: float, out: Vector | None=None, stepsize: float=0.0002, maxiter: int=1000, tol: float=0.001, **proximal_par: Any) -> Vector:
        r"""Prox computation after the method suggested by A. Chambolle (J. Math. Imaging and Vision 20: 89-97, 2004)
        
        Parameters
        ----------
        x: ReadOnlyVector
            First argument of prox
        tau: float
            Second (scaling) argument of prox
        out: Vector | None [default: None]
            Writable output buffer. If None, allocate a new result.
        stepsize: float [default: 0.0002]
            The stepsize. Convergence is guaranteed for values <=0.125.
        maxiter: int [default: 1000]
            Maximum number of iterations
        tol: float [default: 0.001]
            Tolerance parameter for stopping criterion. Iteration is stopped if two consecutive 
            iteratives differ by less than tol in the maximum norm.
        **proximal_par: Any
            Proximal par.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        grad_u = self._grad_op.codomain.empty()
        grad_u_last = self._grad_op.codomain.empty()
        diff_last = inf
        for i in range(maxiter):
            _gfu_help = -self._grad_op.adjoint(grad_u) - x / tau
            update = self._grad_op.codomain.to_gf(stepsize * self._grad_op(_gfu_help))
            vec_norm = ngs.sqrt(sum(ngs.InnerProduct(g, g) for g in update))
            grad_u = self._grad_op.codomain.from_ngs(
                (self._grad_op.codomain.to_gf(grad_u) + update) / (1.0 + vec_norm)
            )
            diff = max(
                ngs.Integrate(ngs.Norm(g), self.domain.fes.mesh)
                for g in self._grad_op.codomain.to_gf(grad_u - grad_u_last)
            )
            if diff < tol and i > 0:
                self.log.info(f'TV proximal Chambolle terminated after {i} iterations, diff={diff}.')
                break
            if diff > diff_last:
                self.log.info(f'TV proximal Chambolle increasing residual, stopping after {i+1} iterations.')
                grad_u.vec.data = grad_u_last.vec
                break
            diff_last = diff
            grad_u_last.vec.data = grad_u.vec
        if out is None:
            return x + tau * self._grad_op.adjoint(grad_u)
        else:
            self.domain.copyto(out, x + tau * self._grad_op.adjoint(grad_u))
            return out

