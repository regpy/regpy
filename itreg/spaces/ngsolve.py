import numpy as np
import ngsolve as ngs

from . import Grid
from .hilbert import HilbertSpace, L2, Sobolev, SobolevBoundary, L2Boundary


class NGSolveDiscretization(Grid):
    def __init__(self, fes, *args, **kwargs):
        self.fes=fes
#        gfu=ngs.GridFunction(self.fes)
#        self.u=gfu.vec.CreateVector()
#        self.v=gfu.vec.CreateVector()
#        self.toret=np.empty(fes.ndof)

        #u, v=self.fes.TnT()
        self.a=ngs.BilinearForm(self.fes, symmetric=True)
        #self.a+=SymbolicBFI(u*v)
        #self.a.Assemble()

        #self.b=self.a.mat.Inverse(freedofs=self.fes.FreeDofs())
        #self.b=self.a.mat

        self.gfu_in=ngs.GridFunction(self.fes)
        self.gfu_toret=ngs.GridFunction(self.fes)
        super().__init__(np.empty(fes.ndof), *args, **kwargs)

#    def inner(self, x):
#        self.v.FV().NumPy()[:]=x
#        toret=np.zeros(self.fes.ndof)
#        for i in range(self.fes.ndof):
#            self.u.FV().NumPy()[:]=np.eye(1, self.fes.ndof, i)[0]
#            toret[i]=InnerProduct(self.u, self.v)
#        return toret

    def apply_gram(self, x):
        self.gfu_in.vec.FV().NumPy()[:]=x
        self.gfu_toret.vec.data = self.a.mat*self.gfu_in.vec
        return self.gfu_toret.vec.FV().NumPy().copy()

    def apply_gram_inverse(self, x):
        self.gfu_in.vec.FV().NumPy()[:]=x
        self.gfu_toret.vec.data = self.b*self.gfu_in.vec
        return self.gfu_toret.vec.FV().NumPy().copy()


class NGSolveBoundaryDiscretization(Grid):
    def __init__(self, fes, fes_bdr, ind, *args, **kwargs):
        self.fes=fes

        #u, v=self.fes.TnT()
        self.a=ngs.BilinearForm(self.fes, symmetric=True)

        self.gfu_in=ngs.GridFunction(self.fes)
        self.gfu_toret=ngs.GridFunction(self.fes)

        super().__init__(np.empty(fes.ndof), *args, **kwargs)

    def apply_gram(self, x):
        self.gfu_in.vec.FV().NumPy()[:]=x
        self.gfu_toret.vec.data = self.a.mat*self.gfu_in.vec
        return self.gfu_toret.vec.FV().NumPy().copy()

    def apply_gram_inverse(self, x):
        self.gfu_in.vec.FV().NumPy()[:]=x
        self.gfu_toret.vec.data = self.b*self.gfu_in.vec
        return self.gfu_toret.vec.FV().NumPy().copy()


@L2.register(NGSolveDiscretization)
class NGSolveFESSpace_L2(HilbertSpace):
    def __init__(self, discr):
        self.discr = discr

        u, v=self.discr.fes.TnT()
        self.discr.a+=ngs.SymbolicBFI(u*v)
        self.discr.a.Assemble()

        self.discr.b=self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())

    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse


@Sobolev.register(NGSolveDiscretization)
class NGSolveFESSpace_H1(HilbertSpace):
    def __init__(self, discr):
        self.discr = discr

        u, v=self.discr.fes.TnT()
        self.discr.a+=ngs.SymbolicBFI(u*v+ngs.grad(u)*ngs.grad(v))
        self.discr.a.Assemble()

        self.discr.b=self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())


    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse


@SobolevBoundary.register(NGSolveDiscretization)
class NGSolveFESSpace_H1_bdr(HilbertSpace):
    def __init__(self, discr):
        self.discr = discr

        u, v=self.discr.fes.TnT()
        self.discr.a+=ngs.SymbolicBFI(u.Trace()*v.Trace()+u.Trace().Deriv()*v.Trace().Deriv(), definedon=self.discr.fes.mesh.Boundaries("cyc"))
        self.discr.a.Assemble()

        self.discr.b=self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())

    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse


@L2Boundary.register(NGSolveDiscretization)
class NGSolveFESSpace_L2_bdr(HilbertSpace):
    def __init__(self, discr):
        self.discr = discr

        u, v=self.discr.fes.TnT()
        self.discr.a+=ngs.SymbolicBFI(u.Trace()*v.Trace(), definedon=self.discr.fes.mesh.Boundaries("cyc"))
        self.discr.a.Assemble()

        self.discr.b=self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())

    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse
