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
space equality for consistency checks. as_real_space selects real or complex
scalars. The default is real, even for complex storage; in that case coordinates
interleave real and imaginary parts and dimension doubles.
"""
from __future__ import annotations

import logging

from .base import *
from .maskable import MaskableVectorSpace, MaskedVectorSpace, TupleMask, MaskableDirectSum
from .numpy import *
from .curve import *

__all__ = ["ReadOnlyVector", "Vector", "TupleVector", "VectorSpaceBase", "VectorSpace", "DirectSum",
           "MaskableVectorSpace", "MaskedVectorSpace", "TupleMask", "MaskableDirectSum",
           "NumPyVectorSpace", "MeasureSpaceFcts", "MaskedMeasureSpaceFcts", "GridFcts", "UniformGridFcts", "Prod",
           "GenCurve","StarCurve","Kite","Peanut","Round_rect","Apple","Three_lobes","Pinched_ellipse",
           "Smoothed_rectangle","Nonsym_shape","Circle","ParameterizedCurve","ParameterizedCurveSpc",
           "GenTrigSpc","GenTrig","StarTrigRadialFcts","StarTrigCurve"]

try:
    from .ngsolve import *

    __all__ += ["NgsVectorSpace"]
except ImportError:
    logging.info("NGSolve is not installed; its vector spaces are not imported.")
