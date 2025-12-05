import ngsolve as ngs
from netgen.geom2d import unit_square
import pytest

from regpy.util import functional_tests as ft
from regpy.vecsps import *
from regpy.functionals import *
from regpy.util import set_rng_seed, Errors

set_rng_seed(15873098306879350073259142812684978477)


class TestNgsL1():
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    vs = NgsVectorSpace(fes,bdr=bdr)
    func = L1(vs)

    def test_ft(self):
        u = self.vs.from_ngs(ngs.IfPos(ngs.sin(ngs.x**2)*ngs.cos(ngs.y),1,-1)*(ngs.x+ngs.y+2))

        ft.test_functional(self.func,u_s=[u],u_stars=[u])

    def test_evaluation(self):
        f = self.vs.from_ngs(ngs.x**2)
        assert self.func(f) == pytest.approx(1/3), Errors.failed_test(f"The evaluation of L1 failed for x**2 on unit square expected 1/3 got {self.func(f)}",self.func,"eval")
    



# def test_NgsTV():
#     bdr = "left|top|right|bottom"
#     mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
#     fes = ngs.H1(mesh, order=6, dirichlet = bdr)
#     vs = NgsVectorSpace(fes,bdr=bdr)

#     func = NgsTV(vs)

#     u = vs.from_ngs(ngs.IfPos(ngs.sin(ngs.x**2)*ngs.cos(ngs.y),1,-1)*(ngs.x+ngs.y+2))

#     ft.test_functional(func,u_s=[u],u_stars=[u])

