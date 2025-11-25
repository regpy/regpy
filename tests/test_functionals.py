from warnings import warn
import traceback

from functional_tests.base_functionals import *
from functional_tests.numpy_functional import *

try: 
    from functional_tests.ngsolve_functional import *
except ImportError as e:
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    warn(f"Registered an import error for the ngsolve functional tests. Could be caused by non existence of ngsolve. Please check again or ngsolve to properly test every functionality. Coming from {tb}")