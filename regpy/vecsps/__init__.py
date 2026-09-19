r"""Vector spaces and vector-storage protocols used throughout RegPy.

RegPy deliberately uses duck typing for vectors. :class:`ReadOnlyVector`
describes non-mutating linear operations; :class:`Vector` additionally
provides the corresponding in-place operations. Generic operators,
functionals, solvers, and stopping rules must not assume NumPy indexing,
attributes such as ``dtype`` or ``shape``, a ``copy`` method, or componentwise
arithmetic.

Every vector representation is accompanied by a :class:`VectorSpaceBase`
subclass. The vector space owns operations that depend on representation or
storage, including membership tests, allocation, copying, inner products,
random sampling, flattening, and reconstruction. Generic code should therefore
use, for example, ``space.empty()`` and ``space.copyto(target, source)`` rather
than ``source.copy()``.

Specialized code may require additional vector operations. Such requirements
must be part of the specialized vector-space or algorithm contract; NumPy- and
NGSolve-specific implementations may naturally use their native APIs.

Vector-space classes also carry metadata such as grid coordinates and define
space equality for consistency checks. All spaces are treated as real vector
spaces, even when their vectors have complex coefficients. This convention
affects basis iteration, dimensions, inner products, and flattened real
representations.
"""

import logging

from .base import *
from .maskable import MaskableVectorSpace, MaskedVectorSpace, TupleMask, MaskableDirectSum
from .numpy import *
from .curve import *

__all__ = ["ReadOnlyVector", "Vector", "TupleVector", "VectorSpaceBase", "DirectSum",
           "MaskableVectorSpace", "MaskedVectorSpace", "TupleMask", "MaskableDirectSum",
           "NumPyVectorSpace", "MeasureSpaceFcts", "GridFcts", "UniformGridFcts", "Prod",
           "GenCurve","StarCurve","Kite","Peanut","Round_rect","Apple","Three_lobes","Pinched_ellipse",
           "Smoothed_rectangle","Nonsym_shape","Circle","ParameterizedCurve","ParameterizedCurveSpc",
           "GenTrigSpc","GenTrig","StarTrigRadialFcts","StarTrigCurve"]

try:
    from .ngsolve import *

    __all__ += ["NgsVectorSpace"]
except ImportError:
    logging.info("NGSolve is not installed; its vector spaces are not imported.")
