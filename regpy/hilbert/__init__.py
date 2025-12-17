r"""Concrete and abstract Hilbert spaces on vector spaces.
"""

import logging

from regpy.util import Errors
from regpy.vecsps import *
from regpy.vecsps import DirectSum as DirectSumVS
from regpy.operators import Operator

from .base import *
from .numpy import *

__all__ = ["L2", "Sobolev","Hm","Hm0","L2Boundary","SobolevBoundary"]

def as_hilbert_space(h, vecsp):
    r"""Convert h to HilbertSpace instance on vecsp.

    - If h is an Operator, it's wrapped in a GramHilbertSpace.
    - If h is callable, e.g. an AbstractSpace, it is called on vecsp to
      construct the concrete space.
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
        raise RuntimeError(Errors.not_equal(vecsp,h.vecsp,add_info="The constructed Hilbert space with as_hilbert_space constructed a Hilbert space with diverting vector space than the method was called with!"))
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


def componentwise(dispatcher, cls=DirectSum):
    r"""Return a callable that iterates over the components of some vector space, constructing a
    `HilbertSpace` on each component, and joining the result. Intended to be used like e.g.

        L2.register(vecsps.DirectSum, componentwise(L2))

    to register a generic component-wise implementation of `L2` on `regpy.vecsps.DirectSum`
    vector spaces. Any vector space that allows iterating over components using Python's
    iterator protocol can be used, but `regpy.vecsps.DirectSum` is the only example of that right
    now.

    Parameters
    ----------
    dispatcher : callable
        The callable, most likely an `AbstractSpace`, to be applied in each component
        vector space to construct the `HilbertSpace` instances.
    cls : callable, optional
        The callable, most likely a `HilbertSpace` subclass, to combine the individual
        `HilbertSpace` instances. Will be called with all spaces as arguments. Default: `DirectSum`.

    Returns
    -------
    callable
        A callable that can be used to register an `AbstractSpace` implementation on
        direct sums.
    """
    def factory(vecsp, **kwargs):
        return cls(*(dispatcher(s, **kwargs) for s in vecsp), vecsp=vecsp)
    return factory




def _register_spaces():
    r"""Auxiliary method to register abstract spaces for various vector spaces. Using the decorator
    method described in `AbstractSpace` does not work due to circular depenencies when
    loading modules.

    This is called from the `regpy` top-level module once, and can be ignored otherwise.
    """

    L2.register(Prod, componentwise(L2,cls=TensorProd))
    L2.register(DirectSumVS, componentwise(L2))
    L2.register(NumPyVectorSpace, L2Generic)
    L2.register(MeasureSpaceFcts,L2MeasureSpaceFcts)
    L2.register(UniformGridFcts, L2UniformGridFcts)

    Sobolev.register(DirectSumVS, componentwise(Sobolev))
    Sobolev.register(UniformGridFcts, SobolevUniformGridFcts)

    Hm.register(Prod, componentwise(HmDomain,cls=TensorProd))
    Hm.register(DirectSumVS, componentwise(HmDomain))
    Hm.register(UniformGridFcts,HmDomain)

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
