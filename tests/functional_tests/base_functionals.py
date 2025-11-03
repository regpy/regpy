import numpy as np

from regpy.vecsps import NumPyVectorSpace, MeasureSpaceFcts,UniformGridFcts
from regpy.operators import ImaginaryPart
from regpy.functionals import *
from regpy.functionals.base import * 
from regpy.functionals.base import Conj

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

