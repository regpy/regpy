import pytest

from regpy.functionals import Hub, TV, VFunc
from regpy.functionals.vector_integral_functionals import L1L2, LppL2
from regpy.util import functional_tests as ft
from regpy.util import set_rng_seed
from regpy.vecsps import UniformGridFcts


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
