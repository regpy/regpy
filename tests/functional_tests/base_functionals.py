import traceback

import numpy as np
import pytest

from regpy.vecsps import NumPyVectorSpace, MeasureSpaceFcts,UniformGridFcts
from regpy.operators import ImaginaryPart, PtwMultiplication
from regpy.functionals import *
from regpy.functionals.base import * 
from regpy.functionals.base import Conj
from regpy.functionals.integral_functionals import LppPower
from regpy.hilbert import L2
from regpy.util import functional_tests as ft
from regpy.util import set_rng_seed, Errors

set_rng_seed(15873098306879350073259142812684978477)


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

def test_AbstractSpace():
    try:
        ab_sp = AbstractFunctional("TestSpace")
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"Trying to create a new Abstract space failed with exception {e} from: {tb}",AbstractFunctional,meth="Creation"))
    vs = MeasureSpaceFcts(shape=(4,2),dtype=float)
    try:
        ab_sp.register(MeasureSpaceFcts,LppPower)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"Trying to register space failed with exception {e} from: {tb}",AbstractFunctional,meth="register"))
    try:
        sp = ab_sp(vs)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"Trying to evaluate on the registered space failed with exception {e} from: {tb}",AbstractFunctional,meth="evaluation"))
    try:
        ab_scalar_mul = 8.0 * ab_sp
        _ = ab_scalar_mul(vs)
        ab_scalar_rmul = ab_sp * 1.0
        _ = ab_scalar_rmul(vs)
        ab_scalar_truediv = ab_sp / 4.5
        _ = ab_scalar_truediv(vs)
        ab_sum = ab_sp + ab_sp
        _ = ab_sum(vs)
        ab_vertical_shift = ab_sp + 10.5
        _ = ab_vertical_shift(vs)
        ab_neg = -ab_sp
        _ = ab_neg(vs)
        ab_combination = ab_sp * PtwMultiplication(vs,vs.ones()*4.0)
        _ = ab_combination(vs)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"Trying to evaluate on the registered space failed with exception {e} from: {tb}",AbstractFunctional,meth="evaluation"))
        


@pytest.mark.parametrize("kind", ["squared_norm", "linear_combination", "direct_sum"])
@pytest.mark.parametrize("conjugate", [False, True], ids=["primal", "conjugate"])
def test_bregman_functional_consistency(kind, conjugate):
    """Check values, derivatives, types, and the general functional protocol."""
    set_rng_seed(2026092401)
    rng = np.random.default_rng(2026092401)
    domain = MeasureSpaceFcts(measure=np.array([0.5, 1., 2., 3.]))
    hilbert = L2(domain)
    quadratic = SquaredNorm(hilbert, a=2., b=np.array([.2, -.3, .4, -.1]), c=3.)
    x = rng.normal(size=domain.shape)
    y = rng.normal(size=domain.shape)
    if kind == "linear_combination":
        linear = LinearFunctional(np.array([.3, -.2, .1, .4]), domain=domain,
                                  h_domain=hilbert, gradient_in_dual_space=True)
        functional = LinearCombination((1.5, quadratic), linear, offset=2.)
    elif kind == "direct_sum":
        functional = FunctionalOnDirectSum([quadratic, SquaredNorm(hilbert, a=3.)])
        x = functional.domain.join(x, 0.5 * x)
        y = functional.domain.join(y, -0.5 * y)
    else:
        functional = quadratic
    if conjugate:
        # Images of primal points are in the effective conjugate domain.
        x, y = functional.subgradient(x), functional.subgradient(y)
        functional = functional.conj
    divergence = functional.bregman_func(y)
    selected = functional.subgradient(y)
    explicit = functional.bregman_func(y, selected)
    expected = functional.bregman(x, y)
    assert divergence(x) == pytest.approx(expected, rel=1e-11, abs=1e-11)
    assert explicit(x) == pytest.approx(expected, rel=1e-11, abs=1e-11)
    assert divergence(y) == pytest.approx(0., abs=1e-11)
    assert divergence(x) >= -1e-11
    assert functional.bregman(x, None, selected) == pytest.approx(
        expected, rel=1e-11, abs=1e-11)
    difference = divergence.subgradient(x) - (functional.subgradient(x) - selected)
    assert functional.h_domain.dual_space().norm(difference) < 1e-10
    if kind == "squared_norm":
        assert isinstance(divergence, SquaredNorm)
    elif kind == "direct_sum":
        assert isinstance(divergence, FunctionalOnDirectSum)
    else:
        assert isinstance(divergence, LinearCombination)
    ft.test_functional(divergence)


def test_bregman_functional_copies_reference_and_rejects_invalid_subgradient():
    domain = NumPyVectorSpace(3)
    functional = LinearCombination(SquaredNorm(L2(domain), a=2.), offset=3.)
    y = np.array([1., 2., 3.])
    ystar = functional.subgradient(y)
    x = np.array([-.2, .5, 1.])
    divergence = functional.bregman_func(y, ystar)
    value = divergence(x)
    y[:] = -7.
    ystar[:] = 9.
    assert divergence(x) == pytest.approx(value, abs=1e-12)
    with pytest.raises(ValueError, match="subdifferential"):
        functional.bregman_func(np.ones(3), -np.ones(3))


def test_linear_combination_flattens_linear_terms_and_offsets():
    domain = MeasureSpaceFcts(measure=np.array([.5, 1., 2.]))
    h = L2(domain)
    f = SquaredNorm(h, a=2., b=np.array([.1, -.2, .3]), c=4.)
    g = SquaredNorm(h, a=3.)
    b = np.array([.4, -.5, .6])
    linear = LinearFunctional(b, domain=domain, h_domain=h, gradient_in_dual_space=True)
    nested = LinearCombination((2., f), linear, offset=3.)
    combined = LinearCombination((3., VerticalShift(nested, 5.)),
                                 (-2., linear), (4., g), offset=-1.)
    assert combined.offset == 23.
    assert len(combined.funcs) == 2
    assert all(not isinstance(component, LinearCombination) for component in combined.funcs)
    assert all(not component.is_linear for component in combined.funcs)
    assert isinstance(combined.linear_part, LinearFunctional)
    np.testing.assert_allclose(combined.linear_part.gradient, b)
    x = np.array([.2, -.3, .7])
    direction = np.array([.8, -.4, .1])
    expected = 6 * f(x) + domain.vdot(b, x).real + 4 * g(x) + 23.
    assert combined(x) == pytest.approx(expected)
    value, derivative = combined.linearize(x)
    assert value == pytest.approx(expected)
    gradient = 6 * f.subgradient(x) + b + 4 * g.subgradient(x)
    np.testing.assert_allclose(derivative, gradient)
    np.testing.assert_allclose(combined.subgradient(x), gradient)
    np.testing.assert_allclose(combined.hessian(x)(direction),
                               6 * f.hessian(x)(direction) + 4 * g.hessian(x)(direction))
    epsilon = 1e-6
    numerical = (combined(x + epsilon * direction) - combined(x - epsilon * direction)) / (2 * epsilon)
    assert numerical == pytest.approx(domain.vdot(gradient, direction).real, rel=1e-7)
    assert "proximal" not in combined.methods
    assert not combined.conj.methods
    assert isinstance(combined.is_convex, bool)
    shifted = VerticalShift(VerticalShift(f, 2.), -5.)
    assert isinstance(shifted, LinearCombination)
    assert shifted.offset == -3.
    assert shifted.linearize(x)[0] == pytest.approx(f(x) - 3.)


def test_linear_combination_conjugate_proximal_and_distances():
    """Use a nonidentity Gram matrix to distinguish vectors from covectors."""
    set_rng_seed(2026092402)
    weights = np.array([.5, 1., 2.])
    domain = MeasureSpaceFcts(measure=weights)
    h = L2(domain)
    f = SquaredNorm(h, a=2., b=np.array([.1, -.2, .3]), c=4.)
    linear = np.array([.4, -.5, .6])
    F = LinearCombination((3., f), LinearFunctional(
        linear, domain=domain, h_domain=h, gradient_in_dual_space=True), offset=7.)
    # F(x) = 1/2 <Q*x,x> + <q,x> + c in coordinate pairing.
    Q = 6 * weights
    q = 3 * weights * f.b + linear
    c = 3 * f.c + 7.
    x = np.array([.2, -.3, .7])
    p = np.array([.8, -.4, .1])
    tau = .7
    assert F.conj(p) == pytest.approx(.5 * np.sum((p-q)**2 / Q) - c)
    np.testing.assert_allclose(F.conj.subgradient(p), (p-q) / Q)
    np.testing.assert_allclose(F.conj.hessian(p)(x), x / Q)
    np.testing.assert_allclose(F.proximal(x, tau), (weights*x-tau*q) / (weights+tau*Q))
    np.testing.assert_allclose(F.conj.proximal(p, tau), (Q*p+tau*weights*q) / (Q+tau*weights))
    assert F.dist_subdiff(p, x) == pytest.approx(np.sqrt(np.sum((p-Q*x-q)**2 / weights)))
    # Conjugate distance uses the nonlinear functional's distance convention.
    assert F.conj.dist_subdiff(x, p) == pytest.approx(
        f.conj.dist_subdiff(x, (p-linear)/3.))
    for operation, argument in [(F.proximal, x), (F.conj.proximal, p)]:
        expected = operation(argument, tau)
        out = argument.copy()
        assert operation(out, tau, out=out) is out
        np.testing.assert_allclose(out, expected)
    ft.test_functional(F)


def test_linear_combination_affine_and_cancelled_terms():
    domain = NumPyVectorSpace(3)
    b = np.array([1., -2., 3.])
    linear = LinearFunctional(b, domain=domain, gradient_in_dual_space=True)
    f = SquaredNorm(L2(domain))
    affine = LinearCombination((2., linear), (-1., linear), offset=4.)
    x = np.array([.5, -.3, .1])
    assert not affine.funcs
    assert affine(x) == pytest.approx(np.dot(b, x) + 4.)
    assert affine.conj(b) == -4.
    assert np.isinf(affine.conj(b + 1.))
    np.testing.assert_allclose(affine.proximal(x, .2), x - .2*b)
    np.testing.assert_array_equal(affine.conj.proximal(x, .2), b)
    constant = LinearCombination((2., VerticalShift(f, 3.)), (-2., f), offset=1.)
    assert not constant.funcs
    assert constant(x) == 7.
    np.testing.assert_array_equal(constant.subgradient(x), domain.zeros())
    np.testing.assert_array_equal(constant.hessian(x)(x), domain.zeros())
    assert constant.conj(domain.zeros()) == -7.
