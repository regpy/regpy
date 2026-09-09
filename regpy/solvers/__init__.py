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
from .setting import Setting
from . import linear
from . import nonlinear

__all__ = ["Solver", "Setting", "RegSolver", "linear", "nonlinear"]
