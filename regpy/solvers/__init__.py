r"""# Solvers for inverse problems.

This module provides the full variety of solvers for linear and nonlinear inverse problems and all the framework.

The structure is as follows

- ``setting`` contains :class:`Setting`, which binds together operator, data,
  and the data fidelity and penalty functionals.
- ``solver`` contains :class:`Solver` and :class:`RegSolver`, the base classes
  for concrete solvers.
- `linear` is a submodule that provides linear solvers 
- `nonlinear` is a submodule that provides nonlinear solvers.

"""

from .solver import RegSolver, Solver
from .setting import Setting, check_Hilbert_applicability, check_Hilbert_applicability_linear
from .special_settings import *
from .sequences import (
    GeometricSequence, powers_of_two, powers_of_b, natural_numbers, squares,
    effective_alpha,
)
from . import linear
from . import nonlinear

__all__ = ["Solver", "Setting", "RegSolver", "linear", "nonlinear", "HilbertSetting",
           "check_Hilbert_applicability", "check_Hilbert_applicability_linear", 
            "GeometricSequence", "powers_of_two", "powers_of_b",
            "natural_numbers", "squares", "effective_alpha"]
