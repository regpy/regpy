from warnings import warn
import traceback

from vecsps_tests.base_vecsps import *
from vecsps_tests.numpy_vecsps import *
from vecsps_tests.curve_vecsps import *

try: 
    from vecsps_tests.ngsolve_vecsps import *
except ImportError as e:
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    warn(f"Registered an import error for the ngsolve vector space tests. Could be caused by non existence of ngsolve. Please check again or ngsolve to properly test every functionality. Coming from {tb}")