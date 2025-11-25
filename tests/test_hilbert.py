from warnings import warn
import traceback

from hilbert_tests.base_hilbert import *
from hilbert_tests.numpy_hilbert import *

try: 
    from hilbert_tests.ngsolve_hilbert import *
except ImportError as e:
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    warn(f"Registered an import error for the ngsolve hilbert space tests. Could be caused by non existence of ngsolve. Please check again or ngsolve to properly test every functionality. Coming from {tb}")