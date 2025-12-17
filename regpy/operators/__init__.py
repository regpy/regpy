r"""
Forward operators
=================

This module provides the basic forward operators, as well as some simple auxiliary operators. 
It contains submodules divided by purpose and dependency. The `base` module provides the base classes
for operators `Operators` as well as their derivative and adjoint. Moreover it implements general 
functionality for linear combination, composition, projections and more. The others are structured as follows

- `numpy` provides the basic operators for `NumPyVectorSpace`/s and its derivatives
- `convolution` provides a basic class `ConvolutionOperator` for convolution operators on `UniformGridFcts`/s
- `graph_operator` provides a class `OperatorGraph` that enables to define a new operator as a graph of
  existing operators. The classes `Edge` and `Node` respectively define what an edge and node in this 
  Graph is and are used to define an Operator.
- `bases_transform` provides a class `BasisTransform` that enables to define a transform from one vector 
  basis into another by specified coefficient matrix.
- `ngsolve` module provides the base class for operators defined on `NgsVectorSpace`/s instances given by
  `NgsOperator`. Moreover it defines some basic operators to define forward operators by their PDE in
  NGSolve.
- `parallel_operators` module provides a feature to evaluate vector of operators in parallel.

All classes and modules that are provided by each submodules can be imported by simply typing:

.. code-block::python

    from regpy.operators import *

Note that the operators of the `ngsolve` submodule are only imported given the fact that you have installed ngsolve.
"""

import logging

from .base import *
from .numpy import * 
from .convolution import *
from .graph_operator import *
from .bases_transform import *
from .parallel_operators import *

__all__ = []

# include every solvers
from .base import __all__ as mod_all
__all__ += mod_all
from .numpy import __all__ as mod_all
__all__ += mod_all
from .convolution import __all__ as mod_all
__all__ += mod_all
from .graph_operator import __all__ as mod_all
__all__ += mod_all
from .bases_transform import __all__ as mod_all
__all__ += mod_all
from .parallel_operators import __all__ as mod_all
__all__ += mod_all

try:
    from .ngsolve import *
    from .ngsolve import __all__ as mod_all
    __all__ += mod_all
except ImportError:
    logging.info("Ngsolve appears to be not installed not importing the respective vector spaces")