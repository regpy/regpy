import pytest
import numpy as np

from regpy import operators, util, vecsps
from regpy.util import tests
from regpy.operators import volterra 
from examples.medium_scattering import mediumscattering

def do_linear_test(op):
    for _ in range(10):
        tests.test_adjoint(op)


def do_nonlinear_test(op):
    for _ in range(10):
        large, small = tests.test_derivative(op, steps=[1e-1, 1e-8])
        assert small / large < 1e-5
    for _ in range(10):
        x = op.domain.rand()
        _, deriv = op.linearize(x)
        do_linear_test(deriv)


def test_linear_volterra():
    do_linear_test(
        volterra.Volterra(
            domain=vecsps.UniformGridFcts(np.linspace(0, 2 * np.pi, 200))))


def test_nonlinear_volterra():
    do_nonlinear_test(
        volterra.Volterra(
            domain=vecsps.UniformGridFcts(np.linspace(0, 2 * np.pi, 200)),
            exponent=3))



