import numpy as np

from regpy.operators import *
from regpy.operators.convolution import *
import regpy.util.operator_tests as ot
from regpy import vecsps
from examples.volterra import volterra 


def test_volterra():
    #linear
    op=volterra.Volterra(domain=vecsps.UniformGridFcts(np.linspace(0, 2 * np.pi, 10)))
    ot.test_operator(op)
    #nonlinear
    op=volterra.Volterra(domain=vecsps.UniformGridFcts(np.linspace(0, 2 * np.pi, 10)),exponent=3)
    ot.test_operator(op)
    #extra: adjoint derivative of composition
    op=volterra.Volterra(domain=vecsps.UniformGridFcts(np.linspace(0, 2 * np.pi, 10)),
                         exponent=3) *volterra.Volterra(domain=vecsps.UniformGridFcts(np.linspace(0, 2 * np.pi, 10)),exponent=2)
    ot.test_adjoint_derivative(op)

def test_identity():
    #real
    dom=vecsps.VectorSpace((2,2))
    op=Identity(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-x)<1e-20)
    ot.test_operator(op)
    #complex
    dom=vecsps.VectorSpace((2,2),np.complex128)
    op=Identity(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-x)<1e-20)

def test_exponential():
    #real
    dom=vecsps.VectorSpace((2,2))
    op=Exponential(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.exp(x))<1e-20)
    ot.test_operator(op)
    #complex
    dom=vecsps.VectorSpace((2,2),np.complex128)
    op=Exponential(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.exp(x))<1e-20)

