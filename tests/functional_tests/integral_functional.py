import numpy as np

import pytest

from regpy.vecsps import NumPyVectorSpace, MeasureSpaceFcts,UniformGridFcts
from regpy.functionals import *
from regpy.functionals import SquaredNorm, KullbackLeibler, RelativeEntropy
from regpy.functionals.base import HorizontalShiftDilation, LinearFunctional, FunctionalOnDirectSum
from regpy.functionals.integral_functionals import Quadratic, Huber, Huber_old
from regpy.hilbert import L2
from regpy.util import functional_tests as ft
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)


@pytest.mark.parametrize("p,l,u", [(1.5, 0.1, 2.3), (1.5, -2.1, -1.2), (1.5, -1.2, 1.), (2., 0.1, 2.3), (2., -2.1, -1.2), (2., -1.2, 1.), (2.5, 0.1, 2.3), (2.5, -2.1, -1.2), (2.5, -1.2, 1.),])
class TestLpp:
    dom = UniformGridFcts((-3.,3.,100))

    def test_constr_l_quad_taylor_u(self, p, l, u):
        ft.test_functional(Lpp(self.dom,p=p,constr_l=l,quad_taylor_u=u),
                        test_second_deriv=(p>=2),test_second_deriv_conj=False
                        )
    
    def test_quad_taylor_l_lin_taylor_u(self, p ,l , u):
        ft.test_functional(Lpp(self.dom,p=p,quad_taylor_l=l,lin_taylor_u=u),
                        test_second_deriv=(p>=2),test_second_deriv_conj=(p<=2)
                        )
    
    def test_lin_taylor_l_constr_u(self, p ,l , u):
        ft.test_functional(Lpp(self.dom,p=p,lin_taylor_l=l,constr_u=u),
                        test_second_deriv=False,test_second_deriv_conj=False
                        )
    def test_quad_taylor_l_quad_taylor_u(self, p ,l , u):
        ft.test_functional(Lpp(self.dom,p=p,quad_taylor_l=l,quad_taylor_u=u),
                        test_second_deriv=(p>=2),test_second_deriv_conj=(p<=2)
                        )

class TestDataFunc():
    dom = MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    
    func = Lpp(dom,p=2.5,constr_l=-1.2,quad_taylor_u=1.)

    @pytest.mark.parametrize("func", [
        Lpp(dom,p=2.5,constr_l=-1.2,quad_taylor_u=1.), 
        KL(dom,w=dom.ones()),    
    ])
    def test_data_func(self,func):
        data = self.dom.rand()
        data_func = func.as_data_func(data)
        ft.test_functional(data_func, test_second_deriv=False, test_second_deriv_conj=False)

        data_func.data = self.dom.ones()
        ft.test_functional(data_func, test_second_deriv=False, test_second_deriv_conj=False)

        data_func_shifted = data_func.shift(self.dom.ones()*0.4)
        ft.test_functional(data_func, test_second_deriv=False, test_second_deriv_conj=False)
        x = ft.sample_essential_domain(self.func)
        assert data_func_shifted(x) == pytest.approx(data_func(x-0.4))

        del data_func.data
        ft.test_functional(data_func, test_second_deriv=False, test_second_deriv_conj=False)

        x = ft.sample_essential_domain(self.func)
        assert data_func(x) == pytest.approx(func(x))


@pytest.mark.parametrize("dom, x, val", [
    (NumPyVectorSpace((2,10)), np.linspace(-5,4.5,20).reshape(2,10), 50.0), 
    (MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64)), np.ones((2,3), dtype=np.float64), 21.0)])
class TestL1():
    def test_evaluate(self,dom, x, val):
        func = L1(dom)
        assert func(x) == pytest.approx(val)

    def test_proximal(self, dom,x,val):
        func = L1(dom)
        for tau in [0.1,1,2]:
            assert (func.proximal(x,tau) == np.maximum(0, np.abs(x)-tau)*np.sign(x)).all()

    def test_ft(self, dom,x,val):
        ft.test_functional(L1(dom),test_second_deriv=False)

class TestKullbackLeibler():
    dom = UniformGridFcts((-1,1,10),(-2,3,5))

    def test_w_ones(self):
        F=KL(self.dom,w=self.dom.ones(),data = 0.2*self.dom.ones())
        ft.test_functional(F)
        F.data=None
        ft.test_functional(F)

        ft.test_functional(HorizontalShiftDilation(F,dilation=3.,shift=F.domain.ones()))
        ft.test_functional(F-2.)

    def test_quad_taylor(self):
        w=1.+0.5*np.sin(self.dom.coords[0]*self.dom.coords[1])   
        F = KL(self.dom,w=w,quad_taylor_l=0.5,quad_taylor_u=5.,omit_normalization=True)
        ft.test_functional(F)
        F.data = 0.2*self.dom.ones()
        ft.test_functional(F)
        F.quad_taylor_l = None
        ft.test_functional(F)
        F.lin_taylor_l = 0.6
        ft.test_functional(F)
        F.quad_taylor_u = None
        ft.test_functional(F)
        lin_taylor_u= 10.*self.dom.ones(); lin_taylor_u[0,0]=11.
        F.quad_taylor_u = lin_taylor_u
        ft.test_functional(F)
        #ft.test_functional(4.*F+LinearFunctional(F.domain.ones(),domain=F.domain))
        #ft.test_functional(HorizontalShiftDilation(F,dilation=3.,shift=2*F.domain.ones()))
        
    def test_lin_taylor_l(self):
        F = KL(self.dom,w=4.*self.dom.ones(),lin_taylor_l=0.1,quad_taylor_u=2.5)
        ft.test_functional(F)

    def test_constr_l_lin_taylor_u(self):
        F = KL(self.dom,w=self.dom.ones(),constr_l=0.3,lin_taylor_u=2.5)
        ft.test_functional(F,test_second_deriv=False,test_second_deriv_conj=False)

    def test_data(self):
        data = self.dom.rand()
        func = KL(self.dom,w=self.dom.ones())

        data = self.dom.rand()
        data_func = func.as_data_func(data)
        ft.test_functional(data_func)

        data_func.data = self.dom.ones()
        ft.test_functional(data_func)

        del data_func.data
        ft.test_functional(data_func)

        x = ft.sample_essential_domain(func)
        assert data_func(x) == pytest.approx(func(x))


@pytest.mark.parametrize("kwargs", [{},{"lin_taylor_l" : 0.2}, {"constr_u":3.}])
class TestRelativeEntropy():
    dom=UniformGridFcts((-5,7,4),(100,200,3))

    def test_ft(self,kwargs):
        F=RE(self.dom,w=self.dom.ones(), **kwargs)
        ft.test_functional(F)
        F.w = 2*self.dom.ones()
        ft.test_functional(F)

    def test_shifted(self, kwargs):
        F = RE(self.dom,w=self.dom.ones(),lin_taylor_l=0.2)
        ft.test_functional(F+LinearFunctional(F.domain.ones(),domain=F.domain),test_second_deriv=False)
        ft.test_functional(HorizontalShiftDilation(F,dilation=3.,shift=F.domain.ones()),test_second_deriv=False)

class TestHuber():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.float64)
    sigma=np.real(dom.ones())
    sigma[0,0]=4
    F=Hub(dom,sigma=sigma,eps=1e-10)

    def test_ft(self):
        ft.test_functional(self.F)

    def test_shifted(self):
        ft.test_functional(HorizontalShiftDilation(self.F,dilation=3,shift=self.dom.ones()))
        ft.test_functional(self.F-2.)
    
    def test_LinearComb(self):
        ft.test_functional(self.F+LinearFunctional(0.5*self.dom.ones(),domain=self.dom),
                       test_second_deriv=False,test_second_deriv_conj=False
                       )    

def test_quadratic():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.float64)
    F=Quadratic(dom,a=2.,b0=1.,c=-2.)
    ft.test_functional(F)

def test_quadratic_intv():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.float64)
    sigma=np.real(dom.ones())
    sigma[0,0]=4
    F=QuadIntv(dom,sigma=sigma,eps=1e-10)
    assert F(2*dom.ones())==np.inf
    ft.test_functional(F)

def test_quadnonneg():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    func = QuadNonneg(dom)
    ft.test_functional(func)

def test_quadbil():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    func = QuadBil(dom,lb = dom.zeros(), ub = dom.ones())
    ft.test_functional(func)

def test_quadlow():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    func = QuadLow(dom)
    ft.test_functional(func)

def test_Composed():
    dom = UniformGridFcts((0,1,10))
    func = Lpp(dom,p=2.5) * np.arange(1,11)
    ft.test_functional(func,test_second_deriv_conj=False)
    func2 = Lpp(dom,p=1.5) * np.arange(1,11)
    ft.test_functional(func2,test_second_deriv=False)

def test_L1_dist_subdiff():
    # further tests of dist_subdiff in  test_subgradient_conj_subgradient_dist_subdiff in functional_tests
    grid = UniformGridFcts((-0.5,0.5,10),periodic=True)
    ran = grid.rand()
    G = L1(grid).shift(ran)
    if not np.isclose(G.dist_subdiff(2*grid.measure*grid.ones(),ran),1.):
        raise RuntimeError('Distance to subdifferential should be 1.')
    H = 3.+L1(grid).dilation(3.)
    x = grid.zeros()
    grad = 3*np.sign(np.linspace(-1,1,10))*grid.measure
    if not np.isclose(H.dist_subdiff(grad,x),0.):
        raise RuntimeError('3*sgn should be in the subdifferential of int |3t| dt.')

def test_FunctionalOnDirectSum():
    grid1 = UniformGridFcts((0,1,10))
    grid2 = UniformGridFcts((0,1,12))
    f1 = KL(grid1,w=grid1.ones())
    f2 = L1(grid2)

    f = FunctionalOnDirectSum((f1,f2))
    ft.test_functional(f)
    ft.test_functional(f+2.)
    ft.test_functional(f.shift(f.domain.ones()))

def test_hilbertnorm():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.complex128)
    l2 = L2(dom)
    func = HilbertNorm(l2)
    ft.test_functional(func)

def test_quadNonneg():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.float64)
    func = QuadNonneg(dom)
    ft.test_functional(func)
    assert np.isclose(func(-6*dom.ones()),np.inf)
    assert np.isclose(func(dom.ones()),10.5) 
    
def test_quadlow():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.float64)
    func = QuadLow(dom,lb=-1)
    ft.test_functional(func)
    func = QuadLow(dom,lb=10,x0=-5)
    assert np.isclose(func(dom.ones()*11),2688.0)
    assert np.isclose(func(-6*dom.ones()),np.inf)
    ft.test_functional(func)
    

def test_quadbil():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.float64)
    func = QuadBil(dom,lb=-5,ub=3)
    ft.test_functional(func)
    func = QuadBil(dom,lb=9,ub=10,x0=0)
    assert np.isclose(func(-6*dom.ones()),np.inf)
    assert np.isclose(func(9*dom.ones()),850.5)
    ft.test_functional(func)


@pytest.fixture(params=[False, True], ids=["unit-measure", "weighted-measure"])
def huber_comparison_case(request):
    measure = (np.array([[0.5, 2., 3.], [1.5, 4., 0.75]])
               if request.param else np.ones((2, 3)))
    domain = MeasureSpaceFcts(measure=measure, dtype=np.float64)
    return domain


@pytest.mark.parametrize("spatial_sigma", [False, True], ids=["scalar-sigma", "array-sigma"])
def test_huber_matches_legacy(huber_comparison_case, spatial_sigma):
    """Compare both implementations inside the quadratic core and in both tails."""
    domain = huber_comparison_case
    sigma = (np.array([[0.5, 1., 2.], [1.5, 0.7, 3.]])
             if spatial_sigma else 1.3)
    new = Huber(domain, sigma=sigma)
    old = Huber_old(domain, sigma=sigma)
    assert type(new) is Quadratic
    x = np.array([[-3., -0.4, 0.], [0.3, 2.5, -2.]]) * sigma
    direction = np.array([[1., -0.5, 0.3], [0.2, -1., 0.7]])
    np.testing.assert_allclose(new(x), old(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new.subgradient(x), old.subgradient(x), rtol=1e-12, atol=1e-12)
    # Avoid the joins, where the classical second derivative does not exist.
    np.testing.assert_allclose(new.hessian(x)(direction), old.hessian(x)(direction), rtol=1e-12, atol=1e-12)
    joins = np.ones(domain.shape) * sigma
    for boundary in (joins, -joins):
        np.testing.assert_allclose(new(boundary), old(boundary), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(new.subgradient(boundary), old.subgradient(boundary), rtol=1e-12, atol=1e-12)
    for tau in (0.2, 2.):
        np.testing.assert_allclose(new.proximal(x, tau), old.proximal(x, tau), rtol=1e-12, atol=1e-12)
    # Strictly interior dual points avoid legacy boundary tolerances.
    dual = 0.3 * sigma * domain.measure * np.ones(domain.shape)
    np.testing.assert_allclose(new.conj(dual), old.conj(dual), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new.conj.subgradient(dual), old.conj.subgradient(dual), rtol=1e-12, atol=1e-12)


def test_huber_data_functional_matches_shift(huber_comparison_case):
    domain = huber_comparison_case
    sigma = np.array([[0.5, 1., 2.], [1.5, 0.7, 3.]])
    data = np.array([[0.2, -0.3, 0.6], [1., -0.5, 0.4]])
    original_data = data.copy()
    residual = sigma * np.array([[-3., -0.4, 0.], [0.3, 2.5, -2.]])
    x = data + residual
    functional = Huber(domain, sigma=sigma)
    assert functional.as_data_func(data) is functional
    assert type(functional) is Quadratic
    assert functional.is_data_func
    np.testing.assert_array_equal(data, original_data)
    np.testing.assert_allclose(functional.b, -data)
    np.testing.assert_allclose(functional.c, 0.5 * data**2)
    np.testing.assert_allclose(functional.lin_taylor_l, data - sigma)
    np.testing.assert_allclose(functional.lin_taylor_u, data + sigma)
    reference = Huber_old(domain, sigma=sigma).as_data_func(data.copy())
    np.testing.assert_allclose(functional(x), reference(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(functional.subgradient(x), reference.subgradient(x), rtol=1e-12, atol=1e-12)
    for tau in (0.2, 2.):
        np.testing.assert_allclose(functional.proximal(x, tau), reference.proximal(x, tau), rtol=1e-12, atol=1e-12)
    dual = 0.3 * sigma * domain.measure
    np.testing.assert_allclose(functional.conj(dual), reference.conj(dual), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(functional.conj.subgradient(dual), reference.conj.subgradient(dual), rtol=1e-12, atol=1e-12)
    # The data shift changes the conjugate by a linear term, not its domain.
    np.testing.assert_allclose(functional.conj_dom_l, -sigma * domain.measure)
    np.testing.assert_allclose(functional.conj_dom_u, sigma * domain.measure)


@pytest.mark.parametrize("remove_with_none", [False, True], ids=["delete-data", "assign-none"])
def test_huber_data_replacement_and_removal(huber_comparison_case, remove_with_none):
    domain = huber_comparison_case
    data = np.array([[0.2, -0.3, 0.6], [1., -0.5, 0.4]])
    replacement = -2. * data
    x = np.array([[-3., -0.4, 0.], [0.3, 2.5, -2.]])
    plain = Huber(domain, sigma=0.8)
    functional = Huber(domain, sigma=0.8).as_data_func(data)
    first_value = functional(x)
    functional.as_data_func(data)
    assert functional(x) == pytest.approx(first_value)
    assert functional.as_data_func() is functional
    assert functional(x) == pytest.approx(first_value)
    functional.data = replacement
    assert functional(x) == pytest.approx(plain(x - replacement))
    np.testing.assert_allclose(functional.lin_taylor_l, replacement - 0.8)
    # Both the setter and as_data_func must replace, not accumulate, data.
    functional.as_data_func(data)
    assert functional(x) == pytest.approx(first_value)
    if remove_with_none:
        functional.data = None
    else:
        del functional.data
    assert not functional.is_data_func
    assert type(functional) is Quadratic
    assert functional(x) == pytest.approx(plain(x))
    np.testing.assert_allclose(functional.b, 0., atol=1e-14)
    np.testing.assert_allclose(functional.c, 0., atol=1e-14)
    np.testing.assert_allclose(functional.lin_taylor_l, -0.8)
    np.testing.assert_allclose(functional.lin_taylor_u, 0.8)



@pytest.fixture
def bregman_case():
    """Positive reproducible inputs on the notebook's periodic grid."""
    domain = UniformGridFcts((0, 1, 10), periodic=True)
    rng = np.random.default_rng(20260917)
    x = 5. + rng.uniform(-1., 1., size=domain.shape)
    y = 6. + rng.uniform(-1., 1., size=domain.shape)
    return domain, x, y


@pytest.mark.parametrize("kind", ["squared_norm", "kullback_leibler"])
def test_bregman_identities(bregman_case, kind):
    """Symmetrization, explicit subgradients, and reversed conjugate identity."""
    domain, x, y = bregman_case
    functional = (SquaredNorm(L2(domain)) if kind == "squared_norm"
                  else KullbackLeibler(domain, w=domain.ones()))
    forward = functional.bregman(x, y)
    assert forward + functional.bregman(y, x) == pytest.approx(
        functional.symm_bregman(x, y), rel=0, abs=1e-12
    )
    xstar = functional.subgradient(x)
    ystar = functional.subgradient(y)
    assert forward == pytest.approx(
        functional.bregman(x, y, ystar), rel=0, abs=1e-12
    )
    assert forward == pytest.approx(
        functional.conj.bregman(ystar, xstar), rel=0, abs=1e-12
    )


def test_squared_norm_bregman_is_squared_distance(bregman_case):
    domain, x, y = bregman_case
    functional = SquaredNorm(L2(domain))
    assert functional.bregman(x, y) == pytest.approx(
        functional(x - y), rel=0, abs=1e-12
    )


def test_relative_entropy_bregman_independent_of_reference(bregman_case):
    domain, x, y = bregman_case
    unit_reference = RelativeEntropy(domain, w=1.)
    variable_reference = RelativeEntropy(domain, w=y)
    assert unit_reference.bregman(x, y) == pytest.approx(
        variable_reference.bregman(x, y), rel=0, abs=1e-12
    )


def test_nonnegative_quadratic_bregman_with_boundary_subgradient():
    """Include the normal-cone contribution at zero coordinates of y."""
    domain = UniformGridFcts((0, 1, 10), periodic=True)
    functional = Quadratic(domain, constr_l=0.)
    quadratic = Quadratic(domain)
    # Fixed arrays ensure both active and inactive constraints are exercised.
    x = np.array([1., 0., 2., 0.5, 0., 3., 0., 1.5, 0.2, 0.])
    y = np.array([0., 1., 0., 0.5, 0., 2., 1., 0., 0.3, 0.])
    gradient = functional.subgradient(y)
    ystar = gradient - 5. * (y == 0)
    assert functional.dist_subdiff(ystar, y) == pytest.approx(
        0., rel=0, abs=1e-12
    )
    expected = quadratic(x - y) + domain.vdot(gradient - ystar, x - y)
    assert functional.bregman(x, y, ystar) == pytest.approx(
        expected, rel=0, abs=1e-12
    )


@pytest.mark.parametrize("kind", ["quadratic", "kullback_leibler", "relative_entropy"])
@pytest.mark.parametrize("conjugate", [False, True], ids=["primal", "conjugate"])
def test_integral_bregman_functional_consistency(kind, conjugate):
    """Exercise constructed Bregman functionals with the general test suite."""
    set_rng_seed(2026092403)
    rng = np.random.default_rng(2026092403)
    domain = MeasureSpaceFcts(measure=np.array([.5, 1., 2., 3.]))
    if kind == "quadratic":
        functional = Quadratic(domain, a=2., b=.3, c=4.)
    elif kind == "kullback_leibler":
        functional = KullbackLeibler(domain, w=np.array([1., 2., 1.5, .8]))
    else:
        functional = RelativeEntropy(domain, w=np.array([1., 2., 1.5, .8]))
    x, y = rng.uniform(.5, 2., size=(2, 4))
    if conjugate:
        x, y = functional.subgradient(x), functional.subgradient(y)
        functional = functional.conj
    divergence = functional.bregman_func(y)
    explicit = functional.bregman_func(y, functional.subgradient(y))
    expected = functional.bregman(x, y)
    assert divergence(x) == pytest.approx(expected, rel=1e-10, abs=1e-11)
    assert explicit(x) == pytest.approx(expected, rel=1e-10, abs=1e-11)
    assert divergence(y) == pytest.approx(0., abs=1e-11)
    assert divergence(x) >= -1e-11
    assert functional.bregman(x, None, functional.subgradient(y)) == pytest.approx(
        expected, rel=1e-10, abs=1e-11)
    if kind == "quadratic" and not conjugate:
        assert isinstance(divergence, Quadratic)
    ft.test_functional(divergence)


@pytest.mark.parametrize("extension", [False, True], ids=["constrained", "extended"])
def test_quadratic_bregman_preserves_data_constraints_and_extensions(extension):
    domain = MeasureSpaceFcts(measure=np.array([.5, 1., 2.]))
    options = ({"lin_taylor_l": -1., "lin_taylor_u": 3.} if extension
               else {"constr_l": -1., "constr_u": 3.})
    functional = Quadratic(domain, a=2., b=.3, c=4., **options)
    functional.as_data_func(np.array([.2, .4, .6]))
    original_coefficients = [getattr(functional, name).copy() for name in ("_a", "_b", "_c")]
    y = np.array([.7, 1., 1.3])
    x = np.array([-.9, 1.5, 4.]) if extension else np.array([.1, 1.5, 2.])
    divergence = functional.bregman_func(y)
    assert isinstance(divergence, Quadratic)
    assert divergence(x) == pytest.approx(functional.bregman(x, y), abs=1e-11)
    for name in options:
        np.testing.assert_array_equal(getattr(divergence, name), getattr(functional, name))
    np.testing.assert_array_equal(divergence.data, functional.data)
    for name, value in zip(("_a", "_b", "_c"), original_coefficients):
        np.testing.assert_array_equal(getattr(functional, name), value)
    if not extension:
        assert np.isinf(divergence(-2 * domain.ones()))


def test_bregman_functional_with_noncanonical_boundary_subgradient():
    """ystar differs from subgradient(y), but lies in the full subdifferential."""
    set_rng_seed(2026092404)
    weights = np.array([.5, 1., 2., 3.])
    domain = MeasureSpaceFcts(measure=weights)
    functional = Quadratic(domain, a=2., constr_l=0., constr_u=2.)
    y = np.array([0., 2., .5, 1.])
    x = np.array([1., 1., 1.5, .2])
    gradient = functional.subgradient(y)
    normal = np.array([-1., 2., 0., 0.])
    ystar = gradient + normal
    assert not np.array_equal(ystar, gradient)
    assert functional.dist_subdiff(ystar, y) == pytest.approx(0., abs=1e-12)
    divergence = functional.bregman_func(y, ystar)
    assert isinstance(divergence, Quadratic)
    expected = np.sum(weights*(x-y)**2) - np.dot(normal, x-y)
    assert divergence(x) == pytest.approx(expected, abs=1e-11)
    assert functional.bregman(x, y, ystar) == pytest.approx(expected, abs=1e-11)
    assert divergence(y) == pytest.approx(0., abs=1e-11)
    assert divergence(x) > functional.bregman_func(y)(x)
    assert np.isinf(divergence(np.array([-1., 1., 1., 1.])))
    # The opposite normal directions are not subgradients at these bounds.
    with pytest.raises(ValueError, match="subdifferential"):
        functional.bregman_func(y, gradient - normal)
    # Hessians are not defined at the constraint/dual transition points.
    ft.test_functional(divergence, test_second_deriv=False, test_second_deriv_conj=False)
    dual_divergence = functional.conj.bregman_func(ystar, y)
    assert dual_divergence(gradient) == pytest.approx(
        functional.conj.bregman(gradient, ystar, y), abs=1e-11)
    ft.test_functional(dual_divergence, test_second_deriv=False, test_second_deriv_conj=False)
