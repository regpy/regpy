r"""# Solvers for inverse problems.

This module provides the full variety of solvers for linear and nonlinear inverse problems and all the framework.

The structure is as follows

- The base module contains 
     - the class Setting which binds together operator, data, and the data fidelity and penalty functionals.
     - the class "RegSolver", which is the base class for almost all of the solvers. 
- `linear` is a submodule that provides linear solvers 
- `nonlinear` is a submodule that provides nonlinear solvers.

"""

from .setting import Setting, RegSolver
from .solver import Solver
from . import linear
from . import nonlinear

__all__ = ["Solver", "Setting", "RegSolver", "linear", "nonlinear"]
