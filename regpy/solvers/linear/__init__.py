r"""This module provides solvers for ill-posed and inverse problems that are modelled 
by linear forward operators.

For ease of use we import all solvers here and provide them to you. So you only need to
import by

```
from regpy.solvers.linear import Landweber
```    

to import the linear Landweber solver. Moreover, for simplicity you may also use

```
from regpy.solvers import *
```

where you have access to `linear` and can use for example `linear.TikhonovCG` to define 
a Tikhonov solver.
"""

from .tikhonov import * 
from .admm import *
from .cgne import *
from .linear_krylov import *
from .primal_dual import *
from .proximal_gradient import *
from .richardson_lucy import *
from .bregman import *
from .ss_newton import *
from .spectral import *

__all__ = []

# include every solvers
from .tikhonov import __all__ as mod_all
__all__ += mod_all
from .admm import __all__ as mod_all
__all__ += mod_all
from .cgne import __all__ as mod_all
__all__ += mod_all
from .linear_krylov import __all__ as mod_all
__all__ += mod_all
from .spectral import __all__ as mod_all
__all__ += mod_all
from .primal_dual import __all__ as mod_all
__all__ += mod_all
from .proximal_gradient import __all__ as mod_all
__all__ += mod_all
from .richardson_lucy import __all__ as mod_all
__all__ += mod_all
from .bregman import __all__ as mod_all
__all__ += mod_all
from .ss_newton import __all__ as mod_all
__all__ += mod_all
from .semismoothNewton import __all__ as mod_all
__all__ += mod_all
