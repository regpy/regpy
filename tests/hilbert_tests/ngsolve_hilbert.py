import ngsolve as ngs
from netgen.geom2d import unit_square
import pytest

from regpy.vecsps.ngsolve import *
from regpy.hilbert.ngsolve import *
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)

from .base_hilbert import hilbert_basics,collect_errors

class TestL2FESpace():
    errors = []
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)

    @pytest.mark.parametrize("vs",[
        NgsVectorSpace(fes,bdr=bdr),
        NgsVectorSpaceWithInnerProduct(fes,bdr=bdr)
    ])
    def test_general(self,vs):
        l2 = L2FESpace(vs)

        self.errors += hilbert_basics(l2,test_methods=True)

        collect_errors(L2FESpace,self.errors)

class TestSobolevFESpace():
    errors = []
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)

    @pytest.mark.parametrize("vs,tol",[
        (NgsVectorSpace(fes,bdr=bdr),1e-10),
        (NgsVectorSpaceWithInnerProduct(fes,bdr=bdr),1e-2)
    ])
    def test_general(self,vs,tol):
        sob = SobolevFESpace(vs)
    
        self.errors += hilbert_basics(sob,test_methods=True,tol=tol)

        collect_errors(SobolevFESpace,self.errors)

class TestH10FESpace():
    errors = []
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    
    @pytest.mark.parametrize("vs,tol",[
        (NgsVectorSpace(fes,bdr=bdr),1e-10),
        (NgsVectorSpaceWithInnerProduct(fes,bdr=bdr),1e-2)
    ])
    def test_general(self,vs,tol):
        h10 = H10FESpace(vs)
        self.errors += hilbert_basics(h10,test_methods=True,tol=tol)
        collect_errors(H10FESpace,self.errors)

class TestL2BoundaryFESpace():
    errors = []
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    
    @pytest.mark.parametrize("vs",[
        NgsVectorSpace(fes,bdr=bdr),
        NgsVectorSpaceWithInnerProduct(fes,bdr=bdr)
    ])
    def test_general(self,vs):
        l2b = L2BoundaryFESpace(vs)
        self.errors += hilbert_basics(l2b,test_methods=True)
        collect_errors(L2BoundaryFESpace,self.errors)

class TestSobolevBoundaryFESpace():
    errors = []
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)

    @pytest.mark.parametrize("vs",[
        NgsVectorSpace(fes,bdr=bdr),
        NgsVectorSpaceWithInnerProduct(fes,bdr=bdr)
    ])
    def test_general(self,vs):
        sobb = SobolevBoundaryFESpace(vs)
    
        self.errors += hilbert_basics(sobb,test_methods=True)

        collect_errors(SobolevBoundaryFESpace,self.errors)