from math import sqrt
import traceback

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

    def test_proximal(self):
        try:
            _ = self.func.proximal(self.vs.from_ngs(ngs.x**2),tau = 0.5)
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            raise AssertionError(Errors.failed_test(f"The proximal computation of NgsL1 functional could not construct an object resulting in exception {e}."+"\n"+"Traceback from:\n"+f" {tb}",obj=self.func,meth="proximal"))
    


class TestNgsTV():
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    vs = NgsVectorSpace(fes,bdr=bdr)
    func = TV(vs)
    h = ngs.sin(ngs.x*2*ngs.pi)*ngs.sin(ngs.y*2*ngs.pi)
    h_p = h*ngs.IfPos(h,1,0)
    h_m = h*ngs.IfPos(-h,1,0)
    h_abs = h_p - h_m
    f = vs.from_ngs(ngs.IfPos(h_abs-0.2,1,0)*ngs.IfPos(h_abs-0.8,0,1)*h+ngs.IfPos(h_m+0.8,0,-1)+ngs.IfPos(h_p-0.8,1,0))

    def test_ft(self):
        ft.test_functional(self.func,u_s=[self.f])

    def test_evaluation(self):
        f = self.vs.from_ngs(ngs.x + ngs.y)
        assert self.func(f) == pytest.approx(sqrt(2),rel=1e-1), Errors.failed_test(f"The evaluation of TV failed for f(x,y) = x+y on unit square expected sqrt(2) got {self.func(f)}",self.func,"eval")

    def test_proximal(self):
        try:
            _ = self.func.proximal(self.vs.from_ngs(ngs.x**2),tau = 0.5)
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            raise AssertionError(Errors.failed_test(f"The proximal computation of NgsTV functional could not construct an object resulting in exception {e}."+"\n"+"Traceback from:\n"+f" {tb}",obj=self.func,meth="proximal"))
