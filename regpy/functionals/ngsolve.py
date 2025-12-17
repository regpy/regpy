'''Special NGSolve functionals defined on the `regpy.vecsps.ngsolve.NgsVectorSpace`. 
'''
from math import inf, sqrt

import ngsolve as ngs

from regpy.util import Errors
from regpy.operators.ngsolve import NgsGradOP
from regpy.vecsps.ngsolve import NgsVectorSpace

from .base import Functional

__all__ = ["SignumFilter", "NgsL1", "NgsTV"]

class SignumFilter(ngs.la.BaseMatrix):
    def __init__ (self, space, vec):
        super().__init__()
        self.gf = ngs.GridFunction(space)
        self.gf.vec.data = vec
        self.gf_out = ngs.GridFunction(space)
        self.gf_help = ngs.GridFunction(space)
    
    def Update(self, new_vec):
        self.gf.vec.data = new_vec
    
    def Mult (self, x, y):
        self.gf_help.vec.data = x
        self.gf_out.Interpolate(ngs.IfPos(self.gf,1,-1)*self.gf_help)
        y.data = self.gf_out.vec
    
    def Height (self):
        return self.space.ndof
    
    def Width (self):
        return self.space.ndof


class NgsL1(Functional):
    r"""Implementation of the :math:`L^1`-norm on a given `NgsVectorSpace`. It is registered under the
    Abstract functional `L1` and should not be called directly but rather used by defining the 
    abstract `L1` functional as the `penalty` or `data_fid` when initializing the regularization
    setting by calling `regpy.solvers.Setting`.

    Parameters
    ----------
    domain : NgsVectorSpace
        The underlying `ngsolve` space. 
    """
    def __init__(self, domain):
        if not isinstance(domain, NgsVectorSpace):
            raise TypeError(Errors.not_instance(domain, NgsVectorSpace,"AN NgsL1 functional is only defined on an NgsVectorSpace."))
        if domain.codim != 1:
            raise ValueError(Errors.value_error("NgsL1 is only implemented for scalar valued spaces.",self))
        self._gfu = ngs.GridFunction(domain.fes)
        self._w_help = domain.empty()
        self.sign = SignumFilter(domain.fes,self._gfu.vec)
        super().__init__(domain,methods={'eval','subgradient','proximal'})

    def _eval(self, x):
        return ngs.Integrate( ngs.Norm(self.domain.to_gf(x)), self.domain.fes.mesh )

    def _subgradient(self, x):
        self.sign.Update(x.vec)
        self._w_help.vec.data = self.sign * x.vec
        return self._w_help.copy()

    def _proximal(self, x, tau):
        self.sign.Update(x.vec)
        self._gfu.vec.data = self.sign * x.vec
        self._gfu.Interpolate(ngs.IfPos(self._gfu - tau,1,0)*(self._gfu - tau))
        self._w_help.vec.data = self.sign * self._gfu.vec
        return self._w_help.copy()

class NgsTV(Functional):
    r"""Implementation of the total variation functional :math:`TV` on a given `NgsVectorSpace`. It is 
    registered under the Abstract functional `TV` and should not be called directly but rather 
    used by defining the abstract `TV` functional as the `penalty` or `data_fid` when initializing 
    the regularization setting by calling `regpy.solvers.Setting`.

    Parameters
    ----------
    domain : NgsVectorSpace
        The underlying `ngsolve` space.
    h_domain : HilbertSpace
        The Hilbert space wrt which the proximal gets computed. 
    """

    def __init__(self, domain):
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

    def _eval(self, x):
        self._gfu.vec.data = x.vec
        gradu = ngs.grad(self._gfu)
        tvnorm = 0
        for i in range(gradu.dim):
            tvnorm += ngs.InnerProduct(gradu[i],gradu[i])
        return sqrt(ngs.Integrate(tvnorm, self.domain.fes.mesh))

    def _proximal(self, x, tau, stepsize=0.0002, maxiter=1000,tol=0.001):
        r"""Prox computation after the method suggested by A. Chambolle (J. Math. Imaging and Vision 20: 89-97, 2004)

        Parameters
        ----------
        x: np.array 
            First argument of prox
        tau: float >=0
            Second (scaling) argument of prox
        stepsize: float [optional, default: 0.0002]
            The stepsize. Convergence is guaranteed for values <=0.125.
        maxiter: int [optional: default: 1000]
            Maximum number of iterations
        tol: float>=0 [optional, default: 0.01]
            Tolerance parameter for stopping criterion. Iteration is stopped if two consecutive 
            iteratives differ by less than tol in the maximum norm. 
        """
        grad_u = self._grad_op.codomain.empty()
        grad_u_last = self._grad_op.codomain.empty()
        diff_last = inf
        for i in range(maxiter):
            _gfu_help = -self._grad_op.adjoint(grad_u)-x/tau
            update = self._grad_op.codomain.to_gf(stepsize*self._grad_op(_gfu_help))
            vec_norm = ngs.sqrt(sum(ngs.InnerProduct(g,g) for g in update))
            grad_u = self._grad_op.codomain.from_ngs((self._grad_op.codomain.to_gf(grad_u) + update)/(1.0 + vec_norm))
            diff = max(ngs.Integrate(ngs.Norm(g),self.domain.fes.mesh) for g in self._grad_op.codomain.to_gf(grad_u-grad_u_last))
            if diff<tol and i>0:
                self.log.info(f'TV proximal Chambolle terminated after {i} iterations, diff={diff}.')
                break
            elif diff>diff_last:
                self.log.info(f'TV proximal Chambolle increasing residual, stopping after {i+1} iterations.')
                grad_u.vec.data = grad_u_last.vec
                break
            else:
                diff_last = diff
                grad_u_last.vec.data = grad_u.vec
        return x+tau*self._grad_op.adjoint(grad_u)