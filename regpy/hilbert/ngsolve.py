import ngsolve as ngs
import numpy as np

from regpy.vecsps.ngsolve import NgsVectorSpace, NgsVectorSpaceWithInnerProduct
from regpy.hilbert import HilbertSpace
from regpy.operators.ngsolve import NgsMatrixMultiplication
from regpy.util import memoized_property, Errors

class L2FESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.L2` on an `NgsVectorSpace`."""
    def __init__(self, vecsp):
        if not isinstance(vecsp, NgsVectorSpace):
            raise TypeError(Errors.not_instance(vecsp,NgsVectorSpace, f"The Implementation of a ngsolve L2 space requires an NgsVectorSpace was given {vecsp}"))
        super().__init__(vecsp=vecsp)
        self._no_pickle = {*self._no_pickle,"__memoized_L2FESpace.gram","__memoized_HilbertSpace.norm_functional"}

    @memoized_property
    def gram(self):
        if isinstance(self.vecsp, NgsVectorSpaceWithInnerProduct):
            return self.vecsp.identity
        elif isinstance(self.vecsp, NgsVectorSpace):
            u, v = self.vecsp.fes.TnT()
            form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
            form += ngs.SymbolicBFI(u * v)
            return NgsMatrixMultiplication(self.vecsp, form)
        else:
            raise NotImplementedError(Errors.generic_message(f"L2FESSpace not implemented for vector spaces of {type(self.vecsp)}"))

class SobolevFESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.Sobolev` on an `NgsVectorSpace`."""
    def __init__(self, vecsp):
        if isinstance(vecsp, NgsVectorSpaceWithInnerProduct):
            raise TypeError(Errors.type_error(f"The default implementation of a ngsolve Sobolev boundary space needs an NgsVectorSpace without a specified InnerProduct was given a NgsVectorSpaceWithInnerProduct"))
        elif not isinstance(vecsp, NgsVectorSpace):
            raise TypeError(Errors.not_instance(vecsp,NgsVectorSpace, f"The Implementation of a ngsolve L2 space requires an NgsVectorSpace was given {vecsp}"))
        super().__init__(vecsp=vecsp)
        self._no_pickle = {*self._no_pickle,"__memoized_SobolevFESpace.gram","__memoized_HilbertSpace.norm_functional"}
    
    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(u * v + ngs.InnerProduct(ngs.Grad(u),ngs.Grad(v)))
        if isinstance(self.vecsp, NgsVectorSpace):
            return NgsMatrixMultiplication(self.vecsp, form)
        else:
            raise NotImplementedError(Errors.generic_message(f"L2FESSpace not implemented for vector spaces of {type(self.vecsp)}"))


class H10FESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.Hm0` on an `NgsVectorSpace`."""
    def __init__(self, vecsp):
        if isinstance(vecsp, NgsVectorSpaceWithInnerProduct):
            raise TypeError(Errors.type_error(f"The default implementation of a ngsolve Sobolev boundary space needs an NgsVectorSpace without a specified InnerProduct was given a NgsVectorSpaceWithInnerProduct"))
        elif not isinstance(vecsp, NgsVectorSpace):
            raise TypeError(Errors.not_instance(vecsp,NgsVectorSpace, f"The Implementation of a ngsolve L2 space requires an NgsVectorSpace was given {vecsp}"))
        super().__init__(vecsp=vecsp)
        self._no_pickle = {*self._no_pickle,"__memoized_H10FESpace.gram","__memoized_HilbertSpace.norm_functional"}
    
    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(ngs.InnerProduct(ngs.grad(u), ngs.grad(v)))
        if isinstance(self.vecsp, NgsVectorSpace):
            return NgsMatrixMultiplication(self.vecsp, form)
        else:
            raise NotImplementedError(Errors.generic_message(f"L2FESSpace not implemented for vector spaces of {type(self.vecsp)}"))


class L2BoundaryFESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.L2Boundary` on an `NgsVectorSpace`."""
    def __init__(self, vecsp, bdr = None):
        if isinstance(vecsp, NgsVectorSpaceWithInnerProduct):
            raise TypeError(Errors.type_error(f"The default implementation of a ngsolve Sobolev boundary space needs an NgsVectorSpace without a specified InnerProduct was given a NgsVectorSpaceWithInnerProduct"))
        elif not isinstance(vecsp, NgsVectorSpace):
            raise TypeError(Errors.not_instance(vecsp,NgsVectorSpace, f"The Implementation of a ngsolve L2 space requires an NgsVectorSpace was given {vecsp}"))
        if bdr is None:
            if vecsp.bdr is None:
                raise ValueError(Errors.value_error("To use L2BoundaryFESpace on an NgsVectorSpace the vector space needs to define a boundary or you use a specified boundary as argument and it cannot be None."))
            self.bdr = vecsp.fes.mesh.Boundaries(vecsp.bdr)
        elif isinstance(bdr,ngs.comp.Region):
            self.bdr = bdr
        elif isinstance(bdr,str):
            self.bdr = vecsp.fes.mesh.Boundaries(bdr)
        else:
            raise TypeError(Errors.type_error(f"The given bdr can be either None, a ngsolve Region, a string regular expression. You gave bdr = {bdr}."))
        super().__init__(vecsp)
        self._no_pickle = {*self._no_pickle,"__memoized_L2BoundaryFESpace.gram","__memoized_HilbertSpace.norm_functional"}

    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(
            u.Trace() * v.Trace(),
            definedon=self.bdr
        )
        if isinstance(self.vecsp, NgsVectorSpace):
            return NgsMatrixMultiplication(self.vecsp, form)
        else:
            raise NotImplementedError(Errors.generic_message(f"L2FESSpace not implemented for vector spaces of {type(self.vecsp)}"))


class SobolevBoundaryFESpace(HilbertSpace):
    r"""The implementation of `regpy.hilbert.SobolevBoundary` on an `NgsVectorSpace`."""
    def __init__(self, vecsp, bdr = None):
        if isinstance(vecsp, NgsVectorSpaceWithInnerProduct):
            raise TypeError(Errors.type_error(f"The default implementation of a ngsolve Sobolev boundary space needs an NgsVectorSpace without a specified InnerProduct was given a NgsVectorSpaceWithInnerProduct"))
        elif not isinstance(vecsp, NgsVectorSpace):
            raise TypeError(Errors.not_instance(vecsp,NgsVectorSpace, f"The Implementation of a ngsolve L2 space requires an NgsVectorSpace was given {vecsp}"))
        if bdr is None:
            if vecsp.bdr is None:
                raise ValueError(Errors.value_error("To use SobolevBoundaryFESpace on an NgsVectorSpace the vector space needs to define a boundary or you use a specified boundary as argument and it cannot be None."))
            self.bdr = vecsp.fes.mesh.Boundaries(vecsp.bdr)
        elif isinstance(bdr,ngs.comp.Region):
            self.bdr = bdr
        elif isinstance(bdr,str):
            self.bdr = vecsp.fes.mesh.Boundaries(bdr)
        else:
            raise TypeError(Errors.type_error(f"The given bdr can be either None, a ngsolve Region, a string regular expression. You gave bdr = {bdr}."))
        super().__init__(vecsp)
        self._no_pickle = {*self._no_pickle,"__memoized_SobolevBoundaryFESpace.gram","__memoized_HilbertSpace.norm_functional"}

    @memoized_property
    def gram(self):
        u, v = self.vecsp.fes.TnT()
        form = ngs.BilinearForm(self.vecsp.fes, symmetric=True)
        form += ngs.SymbolicBFI(
            ngs.InnerProduct(u.Trace(),v.Trace()) + ngs.InnerProduct(u.Trace().Deriv(), v.Trace().Deriv()),
            definedon=self.bdr
        )
        if isinstance(self.vecsp, NgsVectorSpace):
            return NgsMatrixMultiplication(self.vecsp, form)
        else:
            raise NotImplementedError(Errors.generic_message(f"L2FESSpace not implemented for vector spaces of {type(self.vecsp)}"))
