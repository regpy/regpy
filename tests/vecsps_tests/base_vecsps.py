from random import random
from math import isclose
import pytest

import numpy as np

import regpy.vecsps.base as vs_base

random_seed = 28

def call_safe(obj, method_name, *args, **kwargs):
    """
    Calls a method of an object safely.
    
    Returns a tuple:
        (success, result_or_exception)
    
    success: True if the call succeeded, False if an exception occurred
    result_or_exception: the return value or the caught exception
    """
    method = getattr(obj, method_name)
    try:
        result = method(*args, **kwargs)
        return True, result
    except Exception as e:
        return False, e
    
def call_safe_inclusion(obj,method_name,error_log, *args,**kwargs):
    """Executes a methods and tests if the result is in the obj. 

    Parameters
    ----------
    obj : any
        The object
    method_name : string
        Name of the method
    error_log : list
        list where to append the possible error.
    """
    suc, x = call_safe(obj,method_name,*args,**kwargs)
    if suc:
        if x not in obj and not np.isscalar(x): 
            error_log.append(f"The method {method_name} of {obj} constructed an object {type(x)} with arguments {args} and keyword arguments {kwargs} which is not in it.")
            return None
    else:
        error_log.append(f"The method {method_name} of {obj} with arguments {args} and keyword arguments {kwargs} could not be construct an object resulting in error message {x}.")
        return None
    return x

def vecsps_basics(vs,*args,test_methods = False,**kwargs):
    """Initializes an object of `vs` with `kwargs` and tests it basic functionality. If `test_methods` is true it test the standard methods that should be available. 

    Parameters
    ----------
    vs : object
        The object to initialize
    test_methods : bool, optional
        Flag if to test methods, by default False
    kwargs : dict
        The keywords to pass to the initialization of the `vs` instance.
        
    Raises
    ------
    AssertionError
        Should any of the tests fail.
    """
    errors = []
    VS = vs(*args,**kwargs)
    if test_methods:
        _ = VS.zeros()
        _ = VS.ones()
        _ = VS.empty()
        _ = VS.rand()
        if VS.is_complex:
            _ = VS.poisson(VS.ones().real)
        else:
            _ = VS.poisson(VS.ones())
        res = VS.vdot(VS.zeros(),VS.rand())
        assert res == pytest.approx(0), f"The vdot method tested with a zero and random vector resulted in an non-zero answer of {res}"
    VS_alt = vs(*args,**kwargs)
    assert VS == VS_alt, f"The equivalence method __eq__ for {vs} is not properly working."

    _ = VS + VS_alt
    VS += VS_alt
    _ = VS**4
    
def vector_basics(vs,*args, N = 5,**kwargs):
    tol = kwargs["tol"] if "tol" in kwargs else 1e-10
    VS = vs(*args,**kwargs)
    if VS.is_complex:
        for _ in range(N):
            v_1 = VS.rand()
            v_2 = VS.randn()
            v_3 = VS.randn()
            scalar = random() + 1j*random()
            comb = v_1 + scalar * v_2
            assert VS.vdot(comb,v_3).real == pytest.approx((VS.vdot(v_1,v_3)+scalar.conjugate()*VS.vdot(v_2,v_3)),rel_tol=tol), f"Trying to compute the vector dot product of a linear combination of random vectors of {vs} and another random vector failed with `rel_tol` = {tol}"

            assert (v_1.imag == -v_1.conj().imag).all(),f"Tying to compare v.imag with v.conj().imag failed for {v_1}"
    else:
        for _ in range(N):
            v_1 = VS.rand()
            v_2 = VS.randn()
            v_3 = VS.randn()
            scalar = random()
            v_1 *= scalar
            v_1 /= scalar
            comb = v_1 + scalar * v_2
            assert VS.vdot(comb,v_3) == pytest.approx(VS.vdot(v_1,v_3)+scalar*VS.vdot(v_2,v_3),rel_tol=tol), f"Trying to compute the vector dot product of a linear combination of random vectors of {vs} and another random vector failed with `rel_tol` = {tol}"
            _ = v_1 < v_2
            _ = v_1 <= v_2
            _ = v_1 >= v_2
            reg = v_1 > v_2
            _ = reg.all()
            _ = reg.any()

    assert VS.ones().sum() == VS.size, f"Comparing the sum {VS.ones().sum()} of the ones vector to the real size {VS.size} of the vector space {vs} failed."
    

def test_VecSpaceBase():
    vecsps_basics(vs_base.VectorSpaceBase,None,0,random_seed = random_seed)
    
def test_DirectSum():
    vecsps_basics(vs_base.DirectSum,vs_base.VectorSpaceBase(None,0,random_seed=random_seed),vs_base.VectorSpaceBase(None,0,random_seed=random_seed*2))
