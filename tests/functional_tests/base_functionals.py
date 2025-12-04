import numpy as np

from regpy.vecsps import NumPyVectorSpace, MeasureSpaceFcts,UniformGridFcts
from regpy.operators import ImaginaryPart
from regpy.functionals import *
from regpy.functionals.base import * 
from regpy.functionals.base import Conj
from regpy.util import functional_tests as ft
from regpy.hilbert import L2

import pytest

def test_initialization():
    """ Tests if it can initialize the Abstract functional instances with their most general registered spaces and it evaluates properly.
    """
    np_VS = NumPyVectorSpace((2,3))
    x = np_VS.rand()
    func = L1(np_VS)
    _ = func(x)
    ms_VS = MeasureSpaceFcts(np.random.rand(3,2))
    x = ms_VS.rand()
    func = L1(ms_VS)
    _ = func(x)
    func = Lpp(ms_VS)
    _ = func(x)
    func = KL(ms_VS, w = ms_VS.rand())
    _ = func(x)
    func = RE(ms_VS, w = ms_VS.rand())
    _ = func(x)
    func = Hub(ms_VS)
    _ = func(x)
    func = QuadIntv(ms_VS,sigma=1,eps=1e-10)
    _ = func(x)
    func = QuadNonneg(ms_VS)
    _ = func(x)
    func = QuadBil(ms_VS,lb = 0, ub = 1)
    _ = func(x)
    func = QuadLow(ms_VS, lb = 0, x0 = 0)
    ugf = UniformGridFcts((-1,1,10))
    x = ugf.rand()
    func = TV(ugf)
    _ = func(x)
    quad_ugf = UniformGridFcts(5,5)
    x = quad_ugf.rand()
    func = QuadPosSemi(quad_ugf)
    _ = func(x)

def test_operation():
    np_VS = NumPyVectorSpace((2,3),dtype=complex)
    func = Functional(np_VS.real_space())
    op = ImaginaryPart(np_VS)
    assert isinstance(func*op,Composed)
    r = func.domain.randn()
    assert isinstance(func*r,Composed)
    assert func == func*1
    assert func == 1*func
    s = np.random.rand()
    assert isinstance(s*func,LinearCombination)
    func_alt = Functional(np_VS.real_space())
    assert isinstance(func+func_alt,LinearCombination)
    assert isinstance(func+5, VerticalShift)
    assert isinstance(5+func, VerticalShift)
    assert isinstance(func-5, VerticalShift)
    assert isinstance(func.conj, Conj)

class TestSquaredNorm():
    tol=1e-10
    vs=MeasureSpaceFcts(measure=np.arange(1,7).reshape(2,3),dtype=np.complex128)
    func=SquaredNorm(L2(vs),a=2.0,b=3.0*vs.ones(),c=4.0)
    func_shift=SquaredNorm(L2(vs),a=2.0,shift=5.0*vs.ones())
    lin_func=LinearFunctional(vs.ones()*3,domain=vs,h_domain=L2(vs))

    def test_evaluation_abc(self):
        res=self.func(self.vs.ones())
        assert res==pytest.approx(88.0), f"Evaluation of squared norm failed. Difference to expected result is {res-88.0}."
        
    def test_ft_abc(self):
        ft.test_functional(self.func)

    def test_evaluation_shift(self):
        res=self.func_shift(6*self.vs.ones())
        assert res==pytest.approx(21.0), f"Evaluation of squared norm failed. Difference to expected result is {res-21.0}."
        
    def test_ft_shift(self):
        ft.test_functional(self.func_shift)

    def test_sums(self):
        sum1=self.func_shift+self.func
        if not isinstance(sum1,SquaredNorm):
            raise TypeError(f"Addition of compatible squared norms should yield squared norm  but yields {type(sum1)}")
        sum2=self.func_shift+self.lin_func
        if not isinstance(sum2,SquaredNorm):
            raise TypeError(f"Sum of compatible squared norm and linear functional should yield {SquaredNorm}  but yields {type(sum2)}")
        
    def test_differences(self):
        dif1=self.func_shift-self.func
        if not isinstance(dif1,SquaredNorm):
            raise TypeError(f"Difference of compatible squared norms should yield squared norm  but yields {type(dif1)}")
        dif2=self.func_shift-self.lin_func
        if not isinstance(dif2,SquaredNorm):
            raise TypeError(f"Difference of compatible squared norm and linear functional should yield {SquaredNorm}  but yields {type(dif2)}")
        
    def test_eval_data(self):
        data_func=self.func.as_data_func(self.vs.ones())
        res=data_func(2*self.vs.ones())
        assert res==pytest.approx(88.0), f"Evaluation of squared norm failed. Difference to expected result is {res-88.0}."

