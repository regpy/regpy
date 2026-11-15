r"""Concrete and abstract Hilbert spaces on vector spaces.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable
    from regpy.vecsps.base import ReadOnlyVector, VectorSpaceBase
    from regpy.hilbert.base import AbstractSpace, HilbertSpace, MaskedHilbertSpace

import logging

from regpy.util import Errors
from regpy.vecsps import *
from regpy.vecsps import DirectSum as DirectSumVS
from regpy.vecsps.maskable import MaskedVectorSpace
from regpy.operators import Operator

from .base import *
from .numpy import *
from .surface import *

__all__ = ["L2", "Sobolev","Hm","Hm0","L2Boundary","SobolevBoundary","SobolevSphereSHBasis"]

def as_hilbert_space(h: HilbertSpace | Operator| Callabel|None, vecsp: VectorSpaceBase) -> HilbertSpace | None:
    r"""Convert h to HilbertSpace instance on vecsp.
    
    - If h is an Operator, it's wrapped in a GramHilbertSpace.
    - If h is callable, e.g. an AbstractSpace, it is called on vecsp to
      construct the concrete space.
    
    Parameters
    ----------
    h: HilbertSpace | Operator| Callabel| None
        The object definiing the Hilbert space. 
    vecsp: VectorSpaceBase
        Underlying vector space.
    
    
    Returns
    -------
    HilbertSpace
        Result described above.
    """
    if h is None:
        return None
    if not isinstance(h, HilbertSpace):
        if isinstance(h, Operator):
            h = GramHilbertSpace(h)
        elif callable(h):
            h = h(vecsp)
    if not isinstance(h, HilbertSpace):
        raise RuntimeError(Errors.not_instance(h,HilbertSpace,add_info="The construction with the method as_hilbert_space failed to construct a Hilbert Space!"))
    if h.vecsp != vecsp:
        raise RuntimeError(Errors.not_equal(vecsp,h.vecsp,add_info="as_hilbert_space returned a Hilbert space with a different underlying vector space."))
    return h


L2 = AbstractSpace('L2')
r""":math:`L^2` `AbstractSpace`."""

Sobolev = AbstractSpace('Sobolev')
r"""Sobolev `AbstractSpace`"""

Hm = AbstractSpace('Hm')
r""":math:`H^m` `AbstractSpace`"""

Hm0 = AbstractSpace('Hm0')
r""":math:`H^m_0` `AbstractSpace`"""

L2Boundary = AbstractSpace('L2Boundary')
r""":math:`L^2` `AbstractSpace` on a boundary. Mostly for use with NGSolve."""

SobolevBoundary = AbstractSpace('SobolevBoundary')
r"""Sobolev `AbstractSpace` on a boundary. Mostly for use with NGSolve."""


def componentwise(dispatcher: AbstractSpace, cls=DirectSum) -> Callable[[VectorSpaceBase], HilbertSpace]:
    r"""Return a callable that iterates over the components of some vector space, constructing a
    `HilbertSpace` on each component, and joining the result. Intended to be used like e.g.
    
        L2.register(vecsps.DirectSum, componentwise(L2))
    
    to register a generic component-wise implementation of `L2` on `regpy.vecsps.DirectSum`
    vector spaces. Any vector space that allows iterating over components using Python's
    iterator protocol can be used, but `regpy.vecsps.DirectSum` is the only example of that right
    now.
    
    Parameters
    ----------
    dispatcher: AbstractSpace
        The callable, most likely an `AbstractSpace`, to be applied in each component
        vector space to construct the `HilbertSpace` instances.
    
    Returns
    -------
    Callable[[VectorSpaceBase], HilbertSpace]
        A callable that can be used to register an `AbstractSpace` implementation on
        direct sums.
    """
    def factory(vecsp: VectorSpaceBase, **kwargs: Any) -> HilbertSpace:
        r"""Factory.
        
        Parameters
        ----------
        vecsp: VectorSpaceBase
            Underlying vector space.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        HilbertSpace
            Result described above.
        """
        return cls(*(dispatcher(s, **kwargs) for s in vecsp), vecsp=vecsp)
    return factory




def _masked_l2(vecsp: VectorSpaceBase, **kwargs: Any) -> MaskedHilbertSpace:
    r"""Restrict the full space's L2 geometry; kwargs describe that full geometry.
    
    In particular, weights have the shape of the full space, not the masked
    coefficient vector. For another chosen full-space geometry, construct
    MaskedHilbertSpace explicitly. Automatic registration requires a Gram
    operator recognized by the explicit diagonal-inverse implementation;
    no numerical or general matrix-diagonality test is performed.
    
    Parameters
    ----------
    vecsp: VectorSpaceBase
        Underlying vector space.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    
    
    Returns
    -------
    MaskedHilbertSpace
        Result described above.
    """
    full_hilbert = L2(vecsp.get_full_space(), **kwargs)
    if MaskedHilbertSpace._coordinate_diagonal(full_hilbert.gram) is None:
        raise ValueError(
            "L2 on a MaskedVectorSpace requires a full-space Gram operator "
            "recognized as coordinate-diagonal (Identity, PtwMultiplication, "
            "or supported direct sums, scalings, sums and compositions). "
            "Construct MaskedHilbertSpace explicitly for other Gram operators "
            "and provide an inverse if needed."
        )
    return MaskedHilbertSpace(full_hilbert, vecsp)


def _masked_hm0(vecsp: MaskedVectorSpace, **kwargs: Any) -> HmDomain:
    r"""Construct Hm0 on the reduced support, with zero exterior values.

    HmDomain provides the sparse finite-difference matrix and its inverse.
    Unlike full-grid Hm, Hm0 defaults to Dirichlet also at the edges of the
    enclosing computational grid, as required for a zero extension.
    """
    kwargs.setdefault('ext_bd_cond', 'Dir')
    return HmDomain(vecsp, **kwargs)


def _register_spaces() -> None:
    r"""Auxiliary method to register abstract spaces for various vector spaces. Using the decorator
    method described in `AbstractSpace` does not work due to circular depenencies when
    loading modules.
    
    This is called from the `regpy` top-level module once, and can be ignored otherwise.
    """

    L2.register(Prod, componentwise(L2,cls=TensorProd))
    L2.register(DirectSumVS, componentwise(L2))
    L2.register(NumPyVectorSpace, L2Generic)
    # Inherit the L2 geometry of the full space, including quadrature weights.
    L2.register(MaskedVectorSpace, _masked_l2)
    L2.register(MeasureSpaceFcts,L2MeasureSpaceFcts)
    L2.register(UniformGridFcts, L2UniformGridFcts)

    Sobolev.register(DirectSumVS, componentwise(Sobolev))
    Sobolev.register(UniformGridFcts, SobolevUniformGridFcts)
    from regpy.vecsps.surface import StarSHSurfaceSpc
    Sobolev.register(StarSHSurfaceSpc, SobolevSphereSHBasis)

    Hm.register(Prod, componentwise(HmDomain,cls=TensorProd))
    Hm.register(DirectSumVS, componentwise(HmDomain))
    Hm.register(UniformGridFcts,HmDomain)
    # Hm0 is defined on the reduced coordinates of a masked grid. HmDomain
    # assembles the Dirichlet finite-difference Gram matrix on that support.
    Hm0.register(MaskedVectorSpace, _masked_hm0)

    L2Boundary.register(DirectSumVS, componentwise(L2Boundary))

    SobolevBoundary.register(DirectSumVS, componentwise(SobolevBoundary))

    # Import of ngsolve hilbert spaces if possible to import 
    try:
        from .ngsolve import L2FESpace, SobolevFESpace, H10FESpace, L2BoundaryFESpace, SobolevBoundaryFESpace

        L2.register(NgsVectorSpace, L2FESpace)
        Sobolev.register(NgsVectorSpace,SobolevFESpace)
        Hm0.register(NgsVectorSpace,H10FESpace)
        L2Boundary.register(NgsVectorSpace, L2BoundaryFESpace)
        SobolevBoundary.register(NgsVectorSpace,SobolevBoundaryFESpace)
    except :
        logging.info("'Ngsolve' appears to be not installed not registering the respective functionals.")
