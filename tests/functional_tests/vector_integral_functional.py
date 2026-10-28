import pytest
import numpy as np
from regpy.functionals import Hub, TV, VFunc, HuberL2, L1L2, LppL2
from regpy.util import functional_tests as ft
from regpy.util import set_rng_seed
from regpy.vecsps import UniformGridFcts, MaskedMeasureSpaceFcts, MaskedVectorSpace
from regpy.operators import ForwardFDGradient

set_rng_seed(15873098306879350073259142812684978477)


def test_TV():
    ugf = UniformGridFcts((-1, 1, 10), (-1, 1, 10))
    func = TV(ugf)
    func(ugf.rand())
    func.proximal(ugf.rand(), 1.0)


class TestVectorIntegralFunctional:
    grid = UniformGridFcts((-1, 1, 10))
    N_v = 5
    vgrid = grid.vector_valued_space(N_v)

    @pytest.mark.parametrize("p", [1.5, 2, 4])
    def test_LppL2(self, p):
        ft.test_functional(LppL2(self.vgrid, p=p))

    def test_L1L2(self):
        func = L1L2(self.vgrid, conj_tol=1e-15)
        u_s = [ft.sample_vector_in_domain(func) for _ in range(5)]
        u_stars = [
            ft.sample_vector_in_domain(func.conj, dist=1e-6)
            for _ in range(5)
        ]
        ft.test_functional(
            func, u_s=u_s, u_stars=u_stars, test_second_deriv=False
        )

    @pytest.mark.parametrize("sigma", [1e-2, 1e-1, 1.0, 10.0])
    def HuberL2(self, sigma):
        func = VFunc(self.vgrid, scalar_func=Hub(sigma=sigma))
        u_s = [ft.sample_vector_in_domain(func) for _ in range(5)]
        u_stars = [ft.sample_vector_in_domain(func.conj) for _ in range(5)]
        ft.test_functional(
            func,
            u_s=u_s,
            u_stars=u_stars,
            test_second_deriv=False,
            test_second_deriv_conj=False,
        )

def rgb_grid():
    return UniformGridFcts((0, 1, 5), (0, 1, 6), shape_codomain=(3,))


def example_rgb(grid):
    rng = np.random.default_rng(42)
    return rng.normal(size=grid.shape)


@pytest.mark.parametrize('functional_type', [L1L2, HuberL2])
def test_frobenius_not_spectral_norm(functional_type):
    grid = rgb_grid()
    domain = ForwardFDGradient(grid).codomain
    kw = {'sigma': 0.8} if functional_type is HuberL2 else {}
    func = functional_type(domain, **kw)
    z = domain.zeros()
    # This 2x2 submatrix has spectral norm 1 and Frobenius norm sqrt(2).
    z[..., 0, 0] = 1.
    z[..., 1, 1] = 1.
    pointwise = np.sqrt(np.sum(np.abs(z)**2, axis=(-2, -1)))
    np.testing.assert_allclose(pointwise, np.sqrt(2))
    np.testing.assert_allclose(func.vector_norm(z, axis=func._vaxes), pointwise)
    np.testing.assert_allclose(func(z), func.scalar_func(pointwise))


@pytest.mark.parametrize('functional_type', [L1L2, HuberL2])
def test_conjugate_proximal_radial_frobenius(functional_type):
    grid = rgb_grid()
    domain = ForwardFDGradient(grid).codomain
    func = (functional_type(domain, sigma=0.5)
            if functional_type is HuberL2 else functional_type(domain))
    rng = np.random.default_rng(432)
    z = rng.normal(size=domain.shape)
    tau = 0.4
    norms = np.sqrt(np.sum(z**2, axis=(-2, -1)))
    scalar_prox = func.scalar_func.conj.proximal(norms, tau)
    expected = z * (scalar_prox / norms)[..., None, None]
    out = domain.empty()
    assert func.conj.proximal(z, tau, out=out) is out
    np.testing.assert_allclose(out, expected, rtol=2e-12, atol=2e-12)
    # True coupled projection/Huber prox acts on the whole 3x2 block,
    # not on three independent two-vectors.


def test_huber_subgradient_and_matrix_valued_hessian():
    grid = rgb_grid()
    domain = ForwardFDGradient(grid).codomain
    func = HuberL2(domain, sigma=0.7)
    rng = np.random.default_rng(77)
    z = rng.normal(size=domain.shape)
    z *= 1.5
    q = rng.normal(size=domain.shape)
    eps = 1e-6
    hess = func.hessian(z)
    derivative = (func.subgradient(z + eps*q)
                  - func.subgradient(z - eps*q)) / (2*eps)
    np.testing.assert_allclose(hess(q), derivative, rtol=2e-5, atol=2e-5)


def test_tv_exact_uses_single_frobenius_norm_per_pixel():
    grid = rgb_grid()
    grad = ForwardFDGradient(grid)
    func = L1L2(grad.codomain)
    gx = grad(example_rgb(grid))
    expected = func.scalar_func(
        np.sqrt(np.sum(np.abs(gx)**2, axis=(-2, -1))),
    )
    np.testing.assert_allclose(func(gx), expected)

def disk_mask(grid, radius=0.8):
    xx, yy = grid.coords
    return xx**2 + yy**2 <= radius**2


@pytest.mark.parametrize('beta', [0., 0.2])
@pytest.mark.parametrize('boundary_condition', ['Neum', 'Diri', 'per'])
def test_masked_tv_equals_full_grid_tv_of_zero_extension(beta, boundary_condition):
    grid = UniformGridFcts((-1., 1., 9), (-1., 1., 11))
    masked = grid.get_masked_space(disk_mask(grid))
    assert isinstance(masked, MaskedVectorSpace)
    tv = TV(masked, beta=beta, boundary_condition=boundary_condition)
    full_tv = TV(grid, beta=beta, boundary_condition=boundary_condition)

    x = np.linspace(-0.5, 1., masked.shape[0])
    np.testing.assert_allclose(tv(x), full_tv(masked.extend(x)), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(tv.grad(x), full_tv.grad(masked.extend(x)),
                               rtol=1e-13, atol=1e-13)
    assert tv.domain == masked
    assert tv.grad.domain == masked
    assert tv.grad.codomain == full_tv.grad.codomain


@pytest.mark.parametrize('beta', [0., 0.1])
def test_masked_tv_penalizes_jump_at_support_boundary(beta):
    grid = UniformGridFcts((-1., 1., 11), (-1., 1., 11))
    masked = grid.get_masked_space(disk_mask(grid, radius=0.55))
    tv = TV(masked, beta=beta)
    assert tv(masked.ones()) > 0.
    assert TV(grid, beta=beta)(grid.ones()) == pytest.approx(0.)


def test_masked_gradient_algebraic_adjoint():
    grid = UniformGridFcts((-1., 1., 8), (-1., 1., 9))
    masked = grid.get_masked_space(disk_mask(grid, radius=0.7))
    grad = TV(masked).grad
    rng = np.random.default_rng(52)
    x = rng.standard_normal(masked.shape)
    p = rng.standard_normal(grad.codomain.shape)
    np.testing.assert_allclose(np.vdot(grad(x), p),
                               np.vdot(x, grad.adjoint(p)),
                               rtol=2e-13, atol=2e-13)
