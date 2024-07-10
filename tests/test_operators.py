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

