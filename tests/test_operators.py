from warnings import warn
import traceback

from operator_tests.base_operator import *
from operator_tests.numpy_operators import *
from operator_tests.convolution_operator import *
from operator_tests.basetransform_operator import *

try: 
    from operator_tests.ngsolve_operator import *
except ImportError as e:
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    warn("Registered an import error for the ngsolve vector space tests. Could be caused by non existence of ngsolve. Please check again or ngsolve to properly test every functionality.Coming from \n {tb}")