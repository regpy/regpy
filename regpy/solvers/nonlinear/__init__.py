r"""This module provides solvers for ill-posed and inverse problems that are modelled 
by non- forward operators.

For ease of use we import all solvers here and provide them to you. So you only need to
import by

```
from regpy.solvers.nonlinear import *
```    

to gain access to all linear operators. Moreover, for simplicity you may also use 

```
from regpy.solvers import *
```

where you have access to `nonlinear` and can use for example `linear.FISTA` to define 
a TIkhonov solver.
"""


from .fista import *
from .forward_backward_splitting import *
from .irgnm_semismooth import *
from .irgnm import *
from .landweber import *
from .newton import *

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