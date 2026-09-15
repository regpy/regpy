"""Stopping rules for iterative solvers and parameter-choice methods.

The public names are re-exported here to preserve imports such as
``from regpy.stoprules import CountIterations``.
"""

from .general import *
from .parameter_choice_rules import *
from .inner_stoprules import *

from .general import __all__ as _general_all
from .parameter_choice_rules import __all__ as _parameter_choice_all
from .inner_stoprules import __all__ as _inner_stoprules_all

__all__: list[str] = (
    _general_all + _parameter_choice_all + _inner_stoprules_all
)

from .default_factories import *
from .default_factories import __all__ as _factory_all
__all__ += _factory_all
