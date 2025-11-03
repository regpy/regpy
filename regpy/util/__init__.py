"""The utility module for `RegPy` providing some basic methods such as logging and managing remembered properties.

It is divided into submodules:

- `general` which provides general functionality
- `operator_tests` providing test methods for operators
- `functional_tests` providing tests for functionals

All methods from the `general` subpackage can be imported from the `util` module as

```
from regpy.util import *
```

Moreover the general testing methods `test_operator` and `test_functional` are available.
"""

from .general import *
from .operator_tests import test_operator
from .functional_tests import test_functional

__all__ = ["test_operator","test_functional","ClassLogger","memoized_property","set_defaults","complex2real","real2complex","is_complex_dtype","is_real_dtype","is_uniform","linspace_circle","make_repr"]