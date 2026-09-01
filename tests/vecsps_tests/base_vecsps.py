from random import random
import traceback

import pytest
import numpy as np

import regpy.vecsps.base as vs_base
from regpy.util import set_rng_seed,Errors

set_rng_seed(15873098306879350073259142812684978477)
    
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
    print(kwargs)
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

    try:
        for x_i in VS:
            _ = x_i
    except NotImplementedError:
        pass
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"The iteration over {VS} does not properly work. Throwing and exception {e}. Resulting from {tb}",VS,meth="Iteration"))

    try:
        _ = VS + VS_alt
        VS += VS_alt
        _ = VS**4
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"The addition and power implementations for {VS} do not properly work. Throwing and exception {e}. Resulting from {tb}",VS,meth="Linear Combinations"))
    
def vector_basics(vs,*args, test_comparison = True, ones_is_onevec= True, N = 5,**kwargs):
    tol = kwargs["tol"] if "tol" in kwargs else 1e-10
    VS = vs(*args,**kwargs)
    if VS.is_complex:
        for _ in range(N):
            v_1 = VS.rand()
            v_2 = VS.randn()
            v_3 = VS.randn()
            scalar = random() + 1j*random()
            comb = v_1 + scalar * v_2
            vdot_1 = VS.vdot(comb,v_3)
            vdot_2 = VS.vdot(v_1,v_3)+scalar.conjugate()*VS.vdot(v_2,v_3)
            assert vdot_1 == pytest.approx(vdot_2,rel=tol), f"Trying to compute the vector dot product of a linear combination of random vectors of {vs} and another random vector failed with `rel_tol` = {tol}. got for the combination {vdot_1} and expressed as sum of vdots {vdot_2} "
    else:
        for _ in range(N):
            v_1 = VS.rand()
            v_2 = VS.randn()
            v_3 = VS.randn()
            scalar = random()
            v_1 *= scalar
            v_1 /= scalar
            comb = v_1 + scalar * v_2
            vdot_1 = VS.vdot(comb,v_3)
            vdot_2 = VS.vdot(v_1,v_3)+scalar*VS.vdot(v_2,v_3)
            assert vdot_1 == pytest.approx(vdot_2,rel=tol), f"Trying to compute the vector dot product of a linear combination of random vectors of {vs} and another random vector failed with `rel_tol` = {tol}. got for the combination {vdot_1} and expressed as sum of vdots {vdot_2} "
            if test_comparison:
                _ = v_1 < v_2
                _ = v_1 <= v_2
                _ = v_1 >= v_2
                reg = v_1 > v_2
                _ = reg.all()
                _ = reg.any()
    if ones_is_onevec:
        assert VS.ones().sum() == VS.size, f"Comparing the sum {VS.ones().sum()} of the ones vector to the real size {VS.size} of the vector space {vs} failed."
    

def test_VecSpaceBase():
    vecsps_basics(vs_base.VectorSpaceBase,None,0)
    
def test_DirectSum():
    vecsps_basics(vs_base.DirectSum,vs_base.VectorSpaceBase(None,0),vs_base.VectorSpaceBase(None,0))
