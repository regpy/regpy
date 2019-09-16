import ngsolve as ngs

from . import Discretization
from .hilbert import HilbertSpace, L2, L2Boundary, Sobolev, SobolevBoundary
from ..operators import Operator
from ..util import memoized_property


class FESpace(Discretization):
    def __init__(self, fespace):
        assert isinstance(fespace, ngs.FESpace)
        super().__init__(fespace.ndof)
        self.fespace = fespace

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.fespace == other.fespace


class Matrix(Operator):
    def __init__(self, domain, form):
        assert isinstance(domain, FESpace)
        if isinstance(form, ngs.BilinearForm):
            assert domain.fespace == form.space
            form.Assemble()
            self.mat = form.mat
        elif isinstance(form, ngs.BaseMatrix):
            self.mat = form
        else:
            raise TypeError('Invalid type: {}'.format(type(form)))
        super().__init__(domain, domain, linear=True)
        self._gfu_in = ngs.GridFunction(domain.fespace)
        self._gfu_out = ngs.GridFunction(domain.fespace)
        self._inverse = None

    def _eval(self, x):
        self._gfu_in.vec.FV().NumPy()[:] = x
        self._gfu_out.vec.data = self.mat * self._gfu_in.vec
        return self._gfu_out.vec.FV().NumPy().copy()

    def _adjoint(self, y):
        self._gfu_in.vec.FV().NumPy()[:] = y
        self._gfu_out.vec.data = self.mat.T * self._gfu_in.vec
        return self._gfu_out.vec.FV().NumPy().copy()

    @property
    def inverse(self):
        if self._inverse is not None:
            return self._inverse
        else:
            self._inverse = Matrix(
                self.domain,
                self.mat.Inverse(freedofs=self.domain.fespace.FreeDofs())
            )
            self._inverse._inverse = self
            return self._inverse


@L2.register(FESpace)
class L2FESpace(HilbertSpace):
    @memoized_property
    def gram(self):
        u, v = self.discr.fespace.TnT()
        form = ngs.BilinearForm(self.discr.fespace, symmetric=True)
        form += ngs.SymbolicBFI(u * v)
        return Matrix(self.discr, form)


@Sobolev.register(FESpace)
class SobolevFESpace(HilbertSpace):
    @memoized_property
    def gram(self):
        u, v = self.discr.fespace.TnT()
        form = ngs.BilinearForm(self.discr.fespace, symmetric=True)
        form += ngs.SymbolicBFI(u * v + ngs.grad(u) * ngs.grad(v))
        return Matrix(self.discr, form)


@L2Boundary.register(FESpace)
class L2BoundaryFESpace(HilbertSpace):
    @memoized_property
    def gram(self):
        u, v = self.discr.fespace.TnT()
        form = ngs.BilinearForm(self.discr.fespace, symmetric=True)
        form += ngs.SymbolicBFI(
            u.Trace() * v.Trace(),
            definedon=self.discr.fespace.mesh.Boundaries("cyc")
        )
        return Matrix(self.discr, form)


@SobolevBoundary.register(FESpace)
class SobolevBoundaryFESpace(HilbertSpace):
    @memoized_property
    def gram(self):
        u, v = self.discr.fespace.TnT()
        form = ngs.BilinearForm(self.discr.fespace, symmetric=True)
        form += ngs.SymbolicBFI(
            u.Trace() * v.Trace() + u.Trace().Deriv() * v.Trace().Deriv(),
            definedon=self.discr.fespace.mesh.Boundaries("cyc")
        )
        return Matrix(self.discr, form)
