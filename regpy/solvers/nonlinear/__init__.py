r"""This module provides solvers for ill-posed and inverse problems that are modelled 
by nonlinear forward operators.

For ease of use we import all solvers here and provide them to you. So you only need to
import by

```
from regpy.solvers.nonlinear import Landweber
```    

to import the nonlinear Landweber solver. Moreover, for simplicity you may also use

```
from regpy.solvers import *
```

where you have access to `nonlinear` and can use for example `nonlinear.FISTA` to define
a Tikhonov solver.
"""
from __future__ import annotations


from .fista import *
from .forward_backward_splitting import *
from .irgnm_semismooth import *
from .irgnm import *
from .landweber import *
from .newton import *
from .gen_tikhonov import *

__all__ = []

# include every solver
from .fista import __all__ as mod_all
__all__ += mod_all
from .forward_backward_splitting import __all__ as mod_all
__all__ += mod_all
from .irgnm_semismooth import __all__ as mod_all
__all__ += mod_all
from .irgnm import __all__ as mod_all
__all__ += mod_all
from .landweber import __all__ as mod_all
__all__ += mod_all
from .newton import __all__ as mod_all
__all__ += mod_all
