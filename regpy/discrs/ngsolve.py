"""Finite element discretizations using NGSolve

This module implements a `regpy.discrs.Discretization` instance for NGSolve spaces and corresponding
Hilbert space structures. Operators are in the `regpy.operators.ngsolve` module.
"""

import ngsolve as ngs

from regpy.discrs import Discretization
from regpy.hilbert import HilbertSpace, L2, L2Boundary, Sobolev, SobolevBoundary
from regpy.operators import Operator
from regpy.util import memoized_property


class NgsSpace(Discretization):
    """A discretization wrapping an `ngsolve.FESpace`.

    Parameters
    ----------
    fes : ngsolve.FESpace
       The wrapped NGSolve discretization.
    """

    def __init__(self, fes):
        assert isinstance(fes, ngs.FESpace)
        super().__init__(fes.ndof)
        self.fes = fes

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.fes == other.fes


class Matrix(Operator):
    """An operator defined by an NGSolve bilinear form.

    Parameters
    ----------
    domain : NgsSpace
        The discretization.
    form : ngsolve.BilinearForm or ngsolve.BaseMatrix
        The bilinear form or matrix. A bilinear form will be assembled.
    """

    def __init__(self, domain, form):
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


@L2.register(NgsSpace)
class L2FESpace(HilbertSpace):
    """The implementation of `regpy.hilbert.L2` on an `NgsSpace`."""
    @memoized_property
    def gram(self):
        u, v = self.discr.fes.TnT()
        form = ngs.BilinearForm(self.discr.fes, symmetric=True)
        form += ngs.SymbolicBFI(u * v)
        return Matrix(self.discr, form)


@Sobolev.register(NgsSpace)
class SobolevFESpace(HilbertSpace):
    """The implementation of `regpy.hilbert.Sobolev` on an `NgsSpace`."""
    @memoized_property
    def gram(self):
        u, v = self.discr.fes.TnT()
        form = ngs.BilinearForm(self.discr.fes, symmetric=True)
        form += ngs.SymbolicBFI(u * v + ngs.grad(u) * ngs.grad(v))
        return Matrix(self.discr, form)


@L2Boundary.register(NgsSpace)
class L2BoundaryFESpace(HilbertSpace):
    """The implementation of `regpy.hilbert.L2Boundary` on an `NgsSpace`."""
    @memoized_property
    def gram(self):
        u, v = self.discr.fes.TnT()
        form = ngs.BilinearForm(self.discr.fes, symmetric=True)
        form += ngs.SymbolicBFI(
            u.Trace() * v.Trace(),
            definedon=self.discr.fes.mesh.Boundaries("cyc")
        )
        return Matrix(self.discr, form)


@SobolevBoundary.register(NgsSpace)
class SobolevBoundaryFESpace(HilbertSpace):
    """The implementation of `regpy.hilbert.SobolevBoundary` on an `NgsSpace`."""
    @memoized_property
    def gram(self):
        u, v = self.discr.fes.TnT()
        form = ngs.BilinearForm(self.discr.fes, symmetric=True)
        form += ngs.SymbolicBFI(
            u.Trace() * v.Trace() + u.Trace().Deriv() * v.Trace().Deriv(),
            definedon=self.discr.fes.mesh.Boundaries("cyc")
        )
        return Matrix(self.discr, form)
