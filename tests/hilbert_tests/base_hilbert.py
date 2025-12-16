from copy import deepcopy
import traceback

import numpy as np
import pytest

from regpy.vecsps import NumPyVectorSpace
from regpy.operators import MatrixMultiplication
from regpy.hilbert.base import *
from regpy.util import set_rng_seed, Errors

set_rng_seed(15873098306879350073259142812684978477)


def call_safe(obj, method_name, *args, **kwargs):
    """
    Calls a method of an object safely.
    
    Returns
    -------
    result
    
    Raises
    ------
    Assertion Error if the method fails
    """
    method = getattr(obj, method_name)
    try:
        result = method(*args, **kwargs)
        return result
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"The method {method_name} of {obj} with arguments {args} and keyword arguments {kwargs} could not construct an object resulting in exception {e}."+"\n"+"Traceback from:\n"+f" {tb}",obj=obj,meth=method_name))

def check_parallelogram_identity(h_space, rel = 1e-6,tol=1e-10, u= None, v = None):
    u = h_space.vecsp.randn() if u is None else u
    v = h_space.vecsp.randn() if v is None else v
    diff = abs(h_space.norm(u+v)**2-h_space.norm(u-v)**2-4*(h_space.inner(u,v)))
    assert diff == pytest.approx(0, rel = rel, abs = tol), Errors.failed_test(f"The parallelogram identity for {h_space} failed with diff = {diff}.",h_space,meth="Parallelogram Identity")

def hilbert_basics(sp,test_methods = False,**kwargs):
    if not isinstance(sp,HilbertSpace):
        raise ValueError(f"This test should test HilbertSpace instances not {sp}")
    
    if test_methods:
        x = sp.vecsp.randn()
        y = sp.vecsp.randn()
        _ = call_safe(sp,"inner",x,y)
        _ = call_safe(sp,"norm",x)
        try:
            _ = sp.gram_inv(x)
            _ = call_safe(sp,"norm_functional",x)
            _ = call_safe(sp,"dual_space")
        except NotImplementedError:
            pass

    sp_alt = deepcopy(sp)
    try:
        assert sp == sp_alt, Errors.failed_test(f"Making a deepcopy of {sp} creates a different object {sp_alt}",sp,meth="copy")
        _ = sp + sp_alt
        _ = 6 * sp 
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"The addition and power implementations for {sp} do not properly work. Throwing and exception {e}. Resulting from {tb}",sp,meth="Linear Combinations"))

    tol = 1e-10 if "tol" not in kwargs else kwargs["tol"]
    u = None if "u" not in kwargs else kwargs["u"]
    v = None if "v" not in kwargs else kwargs["v"]
    
    check_parallelogram_identity(sp,tol=tol,u=u,v=v)

@pytest.mark.parametrize("vs",[ 
    NumPyVectorSpace((2,4),dtype=float),
    NumPyVectorSpace((2,4),dtype=complex)
])
def test_L2Generic(vs):
    l2 = L2Generic(vs)
    hilbert_basics(l2,test_methods=True)

def test_GramHilbertSpace():
    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),inverse='cholesky',domain= NumPyVectorSpace(4),codomain=NumPyVectorSpace(4))
    gram_sp = GramHilbertSpace(op_mat)
    hilbert_basics(gram_sp,test_methods=True)

@pytest.mark.parametrize("inverse", [ 
    None,
    "cholesky",
    "conjugate"
])
def test_HilbertPullBack(inverse):
    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),inverse='cholesky',domain= NumPyVectorSpace(4),codomain=NumPyVectorSpace(4))
    l2 = L2Generic(op_mat.domain)
    pullback = HilbertPullBack(l2,op_mat, inverse = inverse)
    hilbert_basics(pullback,test_methods=True)

def test_TensorProd():
    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),inverse='cholesky',domain= NumPyVectorSpace(4),codomain=NumPyVectorSpace(4))
    l2 = L2Generic(op_mat.domain)
    pullback = HilbertPullBack(l2,op_mat)
    prod = TensorProd((3.0,l2),(1,pullback))
    
    hilbert_basics(prod,test_methods=True)

def test_AbstractSpace():
    try:
        ab_sp = AbstractSpace("TestSpace")
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"Trying to create a new Abstract space failed with exception {e} from: {tb}",AbstractSpace,meth="Creation"))
    vs = NumPyVectorSpace((4,2),dtype=complex)
    try:
        ab_sp.register(NumPyVectorSpace,L2Generic)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"Trying to register space failed with exception {e} from: {tb}",AbstractSpace,meth="register"))
    try:
        sp = ab_sp(vs)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"Trying to evaluate on the registered space failed with exception {e} from: {tb}",AbstractSpace,meth="evaluation"))
