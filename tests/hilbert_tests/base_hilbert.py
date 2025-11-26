from math import isclose
from copy import deepcopy
import traceback

import numpy as np

from regpy.vecsps import NumPyVectorSpace
from regpy.operators import MatrixMultiplication
from regpy.hilbert.base import *

def collect_errors(cls,errors):
    if errors:
        sep = "\n" +"-"*125 +"\n"
        title = f"During the testing of {cls} the following errors were collected"
        massage = sep+title+sep+ sep.join(errors)
        raise AssertionError(massage)
    else:
        return None

def call_safe(obj, method_name, error_log, *args, **kwargs):
    """
    Calls a method of an object safely.
    
    Returns a tuple:
        result_or_none
    
    result_or_none: the return value or none
    """
    method = getattr(obj, method_name)
    try:
        result = method(*args, **kwargs)
        return result
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        error_log.append(f"The method {method_name} of {obj} with arguments {args} and keyword arguments {kwargs} could not construct an object resulting in exception {e}."+"\n"+f"Resulting from {tb}")
        return None

def check_parallelogram_identity(h_space,tol=1e-10, u= None, v = None):
    u = h_space.vecsp.randn() if u is None else u
    v = h_space.vecsp.randn() if v is None else v
    diff = abs(h_space.norm(u+v)**2-h_space.norm(u-v)**2-4*(h_space.inner(u,v)))
    if diff<tol:
        return None
    else:
        return f"The parallelogram identity for {h_space} failed with diff = {diff}."


def hilbert_basics(sp,test_methods = False,**kwargs):
    if not isinstance(sp,HilbertSpace):
        raise ValueError(f"This test should test HilbertSpace instances not {sp}")
    
    errors = []

    if test_methods:
        x = sp.vecsp.randn()
        y = sp.vecsp.randn()
        _ = call_safe(sp,"inner",errors,x,y)
        _ = call_safe(sp,"norm",errors,x)
        try:
            _ = sp.gram_inv(x)
            _ = call_safe(sp,"norm_functional",errors,x)
            _ = call_safe(sp,"dual_space",errors)
        except NotImplementedError:
            pass

    sp_alt = deepcopy(sp)
    try:
        if not sp == sp_alt:
            raise AssertionError(f"Making a deepcopy of {sp} creates a different object {sp_alt}")
        _ = sp + sp_alt
        _ = 6 * sp 
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        errors.append(f"The addition and power implementations for {sp} do not properly work. Throwing and exception {e}. Resulting from {tb}")

    tol = 1e-10 if "tol" not in kwargs else kwargs["tol"]
    u = None if "u" not in kwargs else kwargs["u"]
    v = None if "v" not in kwargs else kwargs["v"]
    
    res = check_parallelogram_identity(sp,tol=tol,u=u,v=v)

    if res is not None:
        errors.append(res)

    return errors

def test_L2Generic():
    errors = []

    vs = NumPyVectorSpace((2,4),dtype=float)
    l2 = L2Generic(vs)
    errors += hilbert_basics(l2,test_methods=True)

    vs = NumPyVectorSpace((2,4),dtype=complex)
    l2 = L2Generic(vs)
    errors += hilbert_basics(l2,test_methods=True)

    collect_errors(L2Generic,errors)

def test_GramHilbertSpace():
    errors = []

    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),inverse='cholesky',domain= NumPyVectorSpace(4),codomain=NumPyVectorSpace(4))
    gram_sp = GramHilbertSpace(op_mat)
    errors += hilbert_basics(gram_sp,test_methods=True)

    collect_errors(GramHilbertSpace,errors)

def test_HilbertPullBack():
    errors = []

    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),inverse='cholesky',domain= NumPyVectorSpace(4),codomain=NumPyVectorSpace(4))
    l2 = L2Generic(op_mat.domain)
    pullback = HilbertPullBack(l2,op_mat)
    errors += hilbert_basics(pullback,test_methods=True)

    pullback = HilbertPullBack(l2,op_mat,inverse="cholesky")
    errors += hilbert_basics(pullback,test_methods=True)

    pullback = HilbertPullBack(l2,op_mat,inverse="conjugate")
    errors += hilbert_basics(pullback,test_methods=True)

    collect_errors(HilbertPullBack,errors)

def test_TensorProd():
    errors = []

    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),inverse='cholesky',domain= NumPyVectorSpace(4),codomain=NumPyVectorSpace(4))
    l2 = L2Generic(op_mat.domain)
    pullback = HilbertPullBack(l2,op_mat)
    prod = TensorProd((3.0,l2),(1,pullback))

    errors += hilbert_basics(prod,test_methods=True)

    collect_errors(TensorProd,errors)


def test_AbstractSpace():
    errors = []
    try:
        ab_sp = AbstractSpace("TestSpace")
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        errors.append(f"Trying to create a new Abstract space failed with exception {e} from: {tb}")
    vs = NumPyVectorSpace((4,2),dtype=complex)
    try:
        ab_sp.register(NumPyVectorSpace,L2Generic)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        errors.append(f"Trying to register space failed with exception {e} from: {tb}")
    try:
        sp = ab_sp(vs)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        errors.append(f"Trying to evaluate on the registered space failed with exception {e} from: {tb}")

    collect_errors(AbstractSpace,errors)
    
    