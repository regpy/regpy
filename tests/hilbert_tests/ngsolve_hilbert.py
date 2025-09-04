import ngsolve as ngs
from netgen.geom2d import unit_square

from regpy.vecsps.ngsolve import *
from regpy.hilbert.ngsolve import *

from .base_hilbert import hilbert_basics,collect_errors

def test_L2FESpace():
    errors = []
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    vs = NgsVectorSpace(fes,bdr=bdr)

    l2 = L2FESpace(vs)

    errors += hilbert_basics(l2,test_methods=True)

    collect_errors(L2FESpace,errors)

def test_SobolevFESpace():
    errors = []
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    vs = NgsVectorSpace(fes,bdr=bdr)

    l2 = SobolevFESpace(vs)
    print(l2._no_pickle)

    errors += hilbert_basics(l2,test_methods=True)

    collect_errors(SobolevFESpace,errors)

def test_H10FESpace():
    errors = []
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    vs = NgsVectorSpace(fes,bdr=bdr)

    l2 = H10FESpace(vs)
    print(l2._no_pickle)

    errors += hilbert_basics(l2,test_methods=True)

    collect_errors(H10FESpace,errors)

def test_L2BoundaryFESpace():
    errors = []
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    vs = NgsVectorSpace(fes,bdr=bdr)

    l2 = L2BoundaryFESpace(vs)
    print(l2._no_pickle)

    errors += hilbert_basics(l2,test_methods=True)

    collect_errors(L2BoundaryFESpace,errors)

def test_SobolevBoundaryFESpace():
    errors = []
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    vs = NgsVectorSpace(fes,bdr=bdr)

    l2 = SobolevBoundaryFESpace(vs)
    print(l2._no_pickle)

    errors += hilbert_basics(l2,test_methods=True)

    collect_errors(SobolevBoundaryFESpace,errors)