r"""# Solvers for inverse problems.

This module provides the full variety of solvers for linear and non-linear inverse problems and all the framework.

The structure is as follows

- `general` module containing the base class for solvers and regularization settings
- `linear` is the module that provides linear solvers 
- `nonlinear` is the module that provides non-linear solvers.

Note that for convenience you may use 

```
from regpy.solvers import *
```

Which provides the regularization class `Setting`  which is
used to define a setting binds together operator, data fidelity and regularization functional. This setting is 
used by solvers to compute the regularized solution. Moreover, with this import you get access to all linear and
non-linear solvers since `linear` and `nonlinear` are also provided as submodules.
"""

from .general import Setting
from . import linear
from . import nonlinear

__all__ = ["Setting","linear","nonlinear"]
