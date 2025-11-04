import ngsolve as ngs
import numpy as np

from regpy.vecsps import NgsVectorSpace
from regpy.hilbert import HilbertSpace
from regpy.operators import NgsMatrixMultiplication
from regpy.util import memoized_property

class L2FESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.L2` on an `NgsVectorSpace`."""
    def __init__(self, vecsp):
        if not isinstance(vecsp, NgsVectorSpace):
            raise ValueError(f"The Implementation of a ngsolve L2 space requires an NgsVectorSpace was given {vecsp}")
        super().__init__(vecsp=vecsp)
        self._no_pickle = {*self._no_pickle,"__memoized_L2FESpace.gram","__memoized_HilbertSpace.norm_functional"}

    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(u * v)
        return NgsMatrixMultiplication(self.vecsp, form)


class SobolevFESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.Sobolev` on an `NgsVectorSpace`."""
    def __init__(self, vecsp):
        if not isinstance(vecsp, NgsVectorSpace):
            raise ValueError(f"The Implementation of a ngsolve L2 space requires an NgsVectorSpace was given {vecsp}")
        super().__init__(vecsp=vecsp)
        self._no_pickle = {*self._no_pickle,"__memoized_SobolevFESpace.gram","__memoized_HilbertSpace.norm_functional"}
    
    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(u * v + ngs.InnerProduct(ngs.Grad(u),ngs.Grad(v)))
        return NgsMatrixMultiplication(self.vecsp, form)


class H10FESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.Hm0` on an `NgsVectorSpace`."""
    def __init__(self, vecsp):
        if not isinstance(vecsp, NgsVectorSpace):
            raise ValueError(f"The Implementation of a ngsolve L2 space requires an NgsVectorSpace was given {vecsp}")
        super().__init__(vecsp=vecsp)
        self._no_pickle = {*self._no_pickle,"__memoized_H10FESpace.gram","__memoized_HilbertSpace.norm_functional"}
    
    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(ngs.InnerProduct(ngs.grad(u), ngs.grad(v)))
        return NgsMatrixMultiplication(self.vecsp, form)


class L2BoundaryFESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.L2Boundary` on an `NgsVectorSpace`."""
    def __init__(self, vecsp):
        if not isinstance(vecsp, NgsVectorSpace):
            raise ValueError(f"The Implementation of a ngsolve L2 space requires an NgsVectorSpace was given {vecsp}")
        assert vecsp.bdr is not None
        super().__init__(vecsp)
        self._no_pickle = {*self._no_pickle,"__memoized_L2BoundaryFESpace.gram","__memoized_HilbertSpace.norm_functional"}

    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(
            u.Trace() * v.Trace(),
            definedon=self.vecsp.fes.mesh.Boundaries(self.vecsp.bdr)
        )
        return NgsMatrixMultiplication(self.vecsp, form)


class SobolevBoundaryFESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.SobolevBoundary` on an `NgsVectorSpace`."""
    def __init__(self, vecsp):
        if not isinstance(vecsp, NgsVectorSpace):
            raise ValueError(f"The Implementation of a ngsolve L2 space requires an NgsVectorSpace was given {vecsp}")
        assert vecsp.bdr is not None
        super().__init__(vecsp)
        self._no_pickle = {*self._no_pickle,"__memoized_SobolevBoundaryFESpace.gram","__memoized_HilbertSpace.norm_functional"}

    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(
            ngs.InnerProduct(u.Trace(),v.Trace()) + ngs.InnerProduct(u.Trace().Deriv(), v.Trace().Deriv()),
            definedon=self.vecsp.fes.mesh.Boundaries(self.vecsp.bdr)
        )
        return NgsMatrixMultiplication(self.vecsp, form)


