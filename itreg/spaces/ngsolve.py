import ngsolve as ngs
import numpy as np

from . import Discretization
from .hilbert import HilbertSpace, L2, L2Boundary, Sobolev, SobolevBoundary


class NGSolveDiscretization(Discretization):
    def __init__(self, fes):
        super().__init__(fes.ndof)
        self.fes = fes
        self.a = ngs.BilinearForm(self.fes, symmetric=True)
        self.gfu_in = ngs.GridFunction(self.fes)
        self.gfu_toret = ngs.GridFunction(self.fes)

    def apply_gram(self, x):
        self.gfu_in.vec.FV().NumPy()[:] = x
        self.gfu_toret.vec.data = self.a.mat * self.gfu_in.vec
        return self.gfu_toret.vec.FV().NumPy().copy()

    def apply_gram_inverse(self, x):
        self.gfu_in.vec.FV().NumPy()[:] = x
        self.gfu_toret.vec.data = self.b * self.gfu_in.vec
        return self.gfu_toret.vec.FV().NumPy().copy()


class NGSolveBoundaryDiscretization(Discretization):
    def __init__(self, fes):
        super().__init__(fes.ndof)
        self.fes = fes
        self.a = ngs.BilinearForm(self.fes, symmetric=True)
        self.gfu_in = ngs.GridFunction(self.fes)
        self.gfu_toret = ngs.GridFunction(self.fes)

    def apply_gram(self, x):
        self.gfu_in.vec.FV().NumPy()[:] = x
        self.gfu_toret.vec.data = self.a.mat * self.gfu_in.vec
        return self.gfu_toret.vec.FV().NumPy().copy()

    def apply_gram_inverse(self, x):
        self.gfu_in.vec.FV().NumPy()[:] = x
        self.gfu_toret.vec.data = self.b * self.gfu_in.vec
        return self.gfu_toret.vec.FV().NumPy().copy()


@L2.register(NGSolveDiscretization)
class L2NGSolve(HilbertSpace):
    def __init__(self, discr):
        super().__init__(discr)
        u, v = self.discr.fes.TnT()
        self.discr.a += ngs.SymbolicBFI(u * v)
        self.discr.a.Assemble()
        self.discr.b = self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())

    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse


@Sobolev.register(NGSolveDiscretization)
class SobolevNGSolve(HilbertSpace):
    def __init__(self, discr):
        super().__init__(discr)
        u, v = self.discr.fes.TnT()
        self.discr.a += ngs.SymbolicBFI(u * v + ngs.grad(u) * ngs.grad(v))
        self.discr.a.Assemble()
        self.discr.b = self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())

    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse


@SobolevBoundary.register(NGSolveDiscretization)
class SobolevBoundaryNGSolve(HilbertSpace):
    def __init__(self, discr):
        super().__init__(discr)
        u, v = self.discr.fes.TnT()
        self.discr.a += ngs.SymbolicBFI(
            u.Trace() * v.Trace() + u.Trace().Deriv() * v.Trace().Deriv(),
            definedon=self.discr.fes.mesh.Boundaries("cyc")
        )
        self.discr.a.Assemble()
        self.discr.b = self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())

    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse


@L2Boundary.register(NGSolveDiscretization)
class L2BoundaryNGSolve(HilbertSpace):
    def __init__(self, discr):
        super().__init__(discr)
        u, v = self.discr.fes.TnT()
        self.discr.a += ngs.SymbolicBFI(u.Trace() * v.Trace(), definedon=self.discr.fes.mesh.Boundaries("cyc"))
        self.discr.a.Assemble()
        self.discr.b = self.discr.a.mat.Inverse(freedofs=self.discr.fes.FreeDofs())

    @property
    def gram(self):
        return self.discr.apply_gram

    @property
    def gram_inv(self):
        return self.discr.apply_gram_inverse
