'''Special NGSolve functionals defined on the `regpy.vecsps.ngsolve.NgsVectorSpace`. 
'''
import ngsolve as ngs

from regpy.hilbert import L2
from regpy.functionals import Functional

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
    r"""Implementation of the :math:`L^1`-norm on a given `NgsVectorSpace`. It is registered under the
    Abstract functional `L1` and should not be called directly but rather used by defining the 
    abstract `L1` functional as the `penalty` or `data_fid` when initializing the regularization
    setting by calling `regpy.solvers.RegularizationSetting`.

    Parameters
    ----------
    domain : NgsVectorSpace
        The underlying `ngsolve` space. 
    """
    def __init__(self, domain):
        #imported here to prevent circular import
        from regpy.vecsps.ngsolve import NgsVectorSpace
        assert isinstance(domain, NgsVectorSpace)
        self._gfu = ngs.GridFunction(domain.fes)
        self._x_help = domain.zeros()
        self._gfu_help = domain.to_gf(self._x_help)
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
        return self._x_help

    def _hessian(self, x):
        raise NotImplementedError

    def _proximal(self, x, tau): 
        self._gfu.vec.data = x.vec
        sign_x = ngs.IfPos(self._gfu)
        t = sign_x*self._gfu-0.25
        self._gfu_help = ngs.IfPos(t,1,0)*t*sign_x
        return self._x_help


class NgsTV(Functional):
    r"""Implementation of the total variation functional :math:`TV` on a given `NgsVectorSpace`. It is 
    registered under the Abstract functional `TV` and should not be called directly but rather 
    used by defining the abstract `TV` functional as the `penalty` or `data_fid` when initializing 
    the regularization setting by calling `regpy.solvers.RegularizationSetting`.

    Parameters
    ----------
    domain : NgsVectorSpace
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
        self._p = ngs.grad(self._gfu)
        self._gfu_update = ngs.GridFunction(self.domain.fes)
        self._x_out = domain.zeros()
        self._gfu_out = domain.to_gf(self._x_out)
        self._gfu_div = ngs.GridFunction(self.domain.fes)
        self._gfu_div.vec.data = self.ngsdivergence(self._p, self.domain.fes)

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
        self._p = ngs.grad(self._gfu)
        self._gfu_div.vec.data = self.ngsdivergence(self._p, self.domain.fes)
        
        self._gfu.vec.data = x.vec
        for i in range(maxiter):
            self._gfu_update.Set( self._gfu_div - self._gfu/tau )
            update= stepsize * ngs.grad( self._gfu_update )
            #Calculate |update|
            self._p = (self._p + update) / tuple([1+ngs.Norm(update[i]) for i in range(update.dim)])
            self._gfu_div.vec.data = self.ngsdivergence(self._p, self.domain.fes)
        self._gfu_out.Set(self._gfu - tau*self._gfu_div)
        return self._x_out 

def ngsdivergence(p, fes):
    r"""Computes the divergence of a vector field 'p' on a FES 'fes'. gradp is a list of ngsolve CoefficientFunctions
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
        gfu_in = ngs.GridFunction(fes)
        gfu_out = ngs.GridFunction(fes)
        vec_out = gfu_in.vec.CreateVector()
        for i in range(p.dim):
            gfu_in.Set(p[i])
            coeff = ngs.grad(gfu_in)[i]
            gfu_out.Set(coeff)
            vec_out.data += gfu_out.vec
        return vec_out