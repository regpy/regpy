import numpy as np
from regpy.functionals import *
from regpy.vecsps import UniformGridFcts,MeasureSpaceFcts
from regpy.util import functional_tests as ft


def test_huber():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.complex128)
    sigma=np.real(dom.ones())
    sigma[0,0]=4
    F=Huber(dom,sigma=sigma,eps=1e-10)
    #essential domain of conjugate functional is |u_i|<=sigma
    u_stars=[dom.rand() for _ in range(5)]
    for i in range(len(u_stars)):
        scales=np.random.uniform(0,1,dom.shape)
        u_stars[i]*=scales/np.abs(u_stars[i])
    assert F(2*dom.ones())==32.0
    ft.test_functional(F,u_stars=u_stars)

def test_quadratic_intv():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.complex128)
    sigma=np.real(dom.ones())
    sigma[0,0]=4
    F=QuadraticIntv(dom,sigma=sigma,eps=1e-10)
    #essential domain of functional is |u_i|<=sigma
    u_s=[dom.rand() for _ in range(5)]
    for i in range(len(u_s)):
        scales=np.random.uniform(0,1,dom.shape)
        u_s[i]*=scales/np.abs(u_s[i])
    assert F(2*dom.ones())==np.inf
    ft.test_functional(F,u_s=u_s)

