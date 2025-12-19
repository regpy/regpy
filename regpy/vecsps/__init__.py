r"""VectorSpaceBases on which operators are defined.

The classes in this module implement various vector spaces on which the
`regpy.operators.Operator` implementations are defined. The base class is `VectorSpaceBase`\,
which represents plain numpy arrays of some shape and dtype. So far it is assumed that 
vectors are always represented by numpy arrays. 

VectorSpaceBases serve the following main purposes:

 * Derived classes can contain additional data like grid coordinates, bundling metadata in one
   place instead of having every operator generate linspaces / basis functions / whatever on their
   own.
 * Providing methods for generating elements of the proper shape and dtype, like zero arrays,
   random arrays or iterators over a basis.
 * Checking whether a given array is an element of the vector space. This is used for
   consistency checks, e.g. when evaluating operators. The check is only based on shape and dtype,
   elements do not need to carry additional structure. Real arrays are considered as elements of
   complex vector spaces.
 * Checking whether two vector spaces are considered equal. This is used in consistency checks
   e.g. for operator compositions.

All vector spaces are considered as real vector spaces, even if the dtype is complex. This
affects iteration over a basis as well as functions returning the dimension or flattening arrays.
"""

import logging

from .base import *
from .numpy import *
from .curve import *

__all__ = ["TupleVector", "VectorSpaceBase", "DirectSum", "NumPyVectorSpace", "MeasureSpaceFcts", "GridFcts", "UniformGridFcts", "Prod","GenCurve","kite","StarCurve","peanut","round_rect","apple","three_lobes","pinched_ellipse","smoothed_rectangle","nonsym_shape","circle","GenTrigSpc","GenTrig","StarTrigRadialFcts","StarTrigCurve"]

try:
    from .ngsolve import *

    __all__ += ["NgsVectorSpace"]
except ImportError:
    logging.info("Ngsolve appears to be not installed not importing the respective vector spaces")