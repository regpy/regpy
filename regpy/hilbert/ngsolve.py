import ngsolve as ngs
import numpy as np

from regpy.hilbert import HilbertSpace
from regpy.operators import Operator
from regpy.util import memoized_property

class Matrix(Operator):
    r"""An operator defined by an NGSolve bilinear form. This is a helper to define 
    Gram matrices by a bilinear form.  

    Parameters
    ----------
    domain : NgsSpace
        The vector space.
    form : ngsolve.BilinearForm or ngsolve.BaseMatrix
        The bilinear form or matrix. A bilinear form will be assembled.
    """

    def __init__(self, domain, form):
        #imported here to prevent circular import
        from regpy.vecsps.ngsolve import NgsSpace
        assert isinstance(domain, NgsSpace)
        if isinstance(form, ngs.BilinearForm):
            assert domain.fes == form.space
            form.Assemble()
            mat = form.mat
        elif isinstance(form, ngs.BaseMatrix):
            mat = form
        else:
            raise TypeError('Invalid type: {}'.format(type(form)))
        self.mat = mat
        """The assembled matrix."""
        super().__init__(domain, domain, linear=True)
        self._gfu_in = ngs.GridFunction(domain.fes)
        self._gfu_out = ngs.GridFunction(domain.fes)
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
        """The inverse as a `Matrix` instance."""
        if self._inverse is not None:
            return self._inverse
        else:
            self._inverse = Matrix(
                self.domain,
                self.mat.Inverse(freedofs=self.domain.fes.FreeDofs())
            )
            self._inverse._inverse = self
            return self._inverse


class L2FESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.L2` on an `NgsSpace`."""
    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(u * v)
        return Matrix(self.vecsp, form)


class SobolevFESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.Sobolev` on an `NgsSpace`."""
    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(u * v + ngs.InnerProduct(ngs.Grad(u),ngs.Grad(v)))
        return Matrix(self.vecsp, form)


class H10FESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.Hm0` on an `NgsSpace`."""

    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(ngs.InnerProduct(ngs.grad(u), ngs.grad(v)))
        return Matrix(self.vecsp, form)


class L2BoundaryFESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.L2Boundary` on an `NgsSpace`."""
    def __init__(self, vecsp):
        assert vecsp.bdr is not None
        super().__init__(vecsp)

    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(
            u.Trace() * v.Trace(),
            definedon=self.vecsp.fes.mesh.Boundaries(self.vecsp.bdr)
        )
        return Matrix(self.vecsp, form)


class SobolevBoundaryFESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.SobolevBoundary` on an `NgsSpace`."""
    def __init__(self, vecsp):
        assert vecsp.bdr is not None
        super().__init__(vecsp)


    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(
            ngs.InnerProduct(u.Trace(),v.Trace()) + ngs.InnerProduct(u.Trace().Deriv(), v.Trace().Deriv()),
            definedon=self.vecsp.fes.mesh.Boundaries(self.vecsp.bdr)
        )
        return Matrix(self.vecsp, form)


