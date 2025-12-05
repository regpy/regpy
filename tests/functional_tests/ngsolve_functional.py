import ngsolve as ngs
from netgen.geom2d import unit_square

from regpy.util import functional_tests as ft
from regpy.vecsps.ngsolve import *
from regpy.functionals.ngsolve import *
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)


# def test_NgsL1():
#     bdr = "left|top|right|bottom"
#     mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
#     fes = ngs.H1(mesh, order=6, dirichlet = bdr)
#     vs = NgsVectorSpace(fes,bdr=bdr)

#     func = NgsL1(vs)

#     u = vs.from_ngs(ngs.IfPos(ngs.sin(ngs.x**2)*ngs.cos(ngs.y),1,-1)*(ngs.x+ngs.y+2))

#     ft.test_functional(func,u_s=[u],u_stars=[u])

# def test_NgsTV():
#     bdr = "left|top|right|bottom"
#     mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
#     fes = ngs.H1(mesh, order=6, dirichlet = bdr)
#     vs = NgsVectorSpace(fes,bdr=bdr)

#     func = NgsTV(vs)

#     u = vs.from_ngs(ngs.IfPos(ngs.sin(ngs.x**2)*ngs.cos(ngs.y),1,-1)*(ngs.x+ngs.y+2))

#     ft.test_functional(func,u_s=[u],u_stars=[u])

