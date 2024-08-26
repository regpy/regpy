'''Special NGSolve functionals defined on the `regpy.vecsps.ngsolve.NgsSpace`. 
'''
import ngsolve as ngs
import numpy as np

from regpy.hilbert import L2
from regpy.functionals import Functional
from regpy.vecsps.ngsolve import NgsBaseVector

class SignumFilter(ngs.la.BaseMatrix):
    def __init__ (self, space, vec):
        self.super(ngs.la.SymmetricGS, self).__init__()
        self.gf = ngs.GridFunction(space)
        self.gf.vec.data = vec
        self.gf_out = ngs.GridFunction(space)
        self.gf_help = ngs.GridFunction(space)
    
    def Update(self, new_vec):
        self.gf.vec.data = new_vec
    
    def Mult (self, x, y):
        self.gf2.vec.data = x
        self.gf_out.Interpolate(ngs.IfPos(self.gf,1,-1)*self.gf2)
        y.data = self.gf.vec
    
    def Height (self):
        return self.space.ndof
    
    def Width (self):
        return self.space.ndof


class NgsL1(Functional):
    r"""Implementation of the \(L^1\)-norm on a given `NgsSpace`. It is registered under the
    Abstract functional `L1` and should not be called directly but rather used by defining the 
    abstract `L1` functional as the `penalty` or `data_fid` when initializing the regularization
    setting by calling `regpy.solvers.RegularizationSetting`.

    Parameters
    ----------
    domain : NgsSpace
        The underlying `ngsolve` space. 
    """
    def __init__(self, domain):
        #imported here to prevent circular import
        from regpy.vecsps.ngsolve import NgsVectorSpace
        assert isinstance(domain, NgsVectorSpace)
        self._gfu = ngs.GridFunction(domain.fes)
        self._gfu_help = ngs.GridFunction(domain.fes)
        if domain.codim > 1:
            self._fes_util = ngs.VectorL2(domain.fes.mesh, order=0)
        else:
            self._fes_util = ngs.L2(domain.fes.mesh, order=0)
        self._gfu_util = ngs.GridFunction(self._fes_util)
        super().__init__(domain)

    def _eval(self, x):
        self._gfu.vec.data = x
        coeff = ngs.CoefficientFunction(self._gfu)
        return ngs.Integrate( ngs.Norm(coeff), self.domain.fes.mesh )

    def _subgradient(self, x):
        self._gfu.vec.data = x.vec
        self._gfu_help.Interpolate(ngs.IfPos(self._gfu,1,-1)*self._gfu)
        return NgsBaseVector(self._gfu_help.vec,make_copy=True)

    def _hessian(self, x):
        raise NotImplementedError

    def _proximal(self, x, tau): 
        self._gfu.vec.data = x.vec
        sign_x = ngs.IfPos(self._gfu)
        t = sign_x*self._gfu-0.25
        self._gfu_help = ngs.IfPos(t,1,0)*t*sign_x
        return NgsBaseVector(self._gfu_help.vec,make_copy=True)


class NgsTV(Functional):
    r"""Implementation of the total variation functional \(TV\) on a given `NgsSpace`. It is 
    registered under the Abstract functional `TV` and should not be called directly but rather 
    used by defining the abstract `TV` functional as the `penalty` or `data_fid` when initializing 
    the regularization setting by calling `regpy.solvers.RegularizationSetting`.

    Parameters
    ----------
    domain : NgsSpace
        The underlying `ngsolve` space.
    h_domain : HilbertSpace
        The Hilbert space wrt which the proximal gets computed. 
    """

    def __init__(self, domain, h_domain=L2):
        #imported here to prevent circular import
        from regpy.vecsps.ngsolve import NgsVectorSpace
        assert isinstance(domain, NgsVectorSpace)
        assert domain.codim == 1, "TV is not implemented for vector valued spaces." 
        super().__init__(domain,h_domain=h_domain)
        self._gfu = ngs.GridFunction(self.domain.fes)
        self._gfu.Set(0)
        self._p = list(ngs.grad(self._gfu))
        self._q = list(ngs.grad(self._gfu))
        self._gfu_update = ngs.GridFunction(self.domain.fes)
        self._gfu_out = ngs.GridFunction(self.domain.fes)
        self._gfu_div = ngs.GridFunction(domain.fes)
        self._gfu_div.vec.FV().NumPy()[:] = ngsdivergence(self._p, self.domain.fes)
        self._fes_util = ngs.L2(self.domain.fes.mesh, order=0)
        self._gfu_util = ngs.GridFunction(self._fes_util)

    def _eval(self, x):
        self._gfu.vec.data = x.vec
        gradu = ngs.grad(self._gfu)
        tvnorm = 0
        for i in range(gradu.dim):
            tvnorm += ngs.Integrate( ngs.Norm(gradu[i]), self.domain.fes.mesh )
        return tvnorm

    def _subgradient(self, x):
        raise NotImplementedError

    def _hessian(self, x):
        raise NotImplementedError

    def _proximal(self, x, tau, stepsize=0.1, maxiter=10):
        self._gfu.Set(0)
        self._p = list(ngs.grad(self._gfu))

        self._gfu.vec.data = x.vec
        for i in range(maxiter):
            self._gfu_update.Set( self._gfu_div - self._gfu/tau )
            update= stepsize * ngs.grad( self._gfu_update )
            #Calculate |update|
            for i in range(len(self._p)):
                self._q[i] = 1+ngs.Norm(update[i])
                self._p[i] = (self._p[i] + update[i]) / self._q[i]
            self._gfu_div.vec.FV().NumPy()[:] = ngsdivergence(self._p, self.domain.fes)
        self._gfu_out.Set(self._gfu - tau*self._gfu_div)
        return NgsBaseVector(self._gfu_out.vec,make_copy=True)  

def ngsdivergence(p, fes):
    """Computes the divergence of a vector field 'p' on a FES 'fes'. gradp is a list of ngsolve CoefficientFunctions
    p=(p_x, p_y, p_z, ...). The return value is the coefficient array of the GridFunction holding the divergence.
    
    Parameters
    ----------
    p : vector field
        Vector field on a FES 'fes' for which to compute the divergence.
    fes : ngsolve fes
        Underlying FES.

    Returns
    -------
    array
        Values of the divergence of the given vector `p`
    """
    toret = np.zeros(fes.ndof)
    gfu_in = ngs.GridFunction(fes)
    gfu_out = ngs.GridFunction(fes)
    for i in range(len(p)):
        gfu_in.Set(p[i])
        coeff = ngs.grad(gfu_in)[i]
        gfu_out.Set(coeff)
        toret += gfu_out.vec.FV().NumPy().copy()
    return toret