import ngsolve as ngs
from netgen.geom2d import unit_square

from regpy.vecsps.ngsolve import *
from regpy.operators.ngsolve import *

from .base_operator import op_basics_wrapper,op_evaluation_and_ot


def test_basic_NgsOperator():
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    vs = NgsVectorSpace(fes,bdr=bdr)

    op_basics_wrapper(NgsOperator,vs,vs)

def test_SecondOrderEllipticCoefficientPDE():
    #taken from diffusion example
    class diffusion(SecondOrderEllipticCoefficientPDE):
        def __init__(self, domain, sol_domain,bdr_val = None,a_bdr_val=None):
            super().__init__(domain, sol_domain, bdr_val=bdr_val,a_bdr_val=a_bdr_val)

        def _bf(self,a,u,v):
            return a*ngs.grad(u)*ngs.grad(v)*ngs.dx
        
        def _lf(self):
            p = ngs.GridFunction(self.codomain.fes)
            p.Set(-2*ngs.exp(ngs.x+ngs.y))
            lf = ngs.LinearForm(self.codomain.fes)
            lf += p * self.v * ngs.dx
            return lf.Assemble()

    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes_domain = ngs.H1(mesh, order=6, dirichlet = bdr)
    domain = NgsVectorSpace(fes_domain,bdr = bdr)

    bdr = "left|top|right|bottom"
    fes_codomain = ngs.H1(mesh, order=6, dirichlet=bdr)
    codomain = NgsVectorSpace(fes_codomain, bdr=bdr)

    bdr_coeff = ngs.sin(ngs.x*4)+2*ngs.y
    bdr_gf = ngs.GridFunction(codomain.fes)
    bdr_gf.Set(bdr_coeff,definedon=codomain.fes.mesh.Boundaries(codomain.bdr))
    bdr_val = codomain.from_ngs(bdr_gf)

    exact_solution_coeff = 0.5*ngs.exp(-4*(ngs.x-0.5)**2 +4*(ngs.y-0.5)**2)
    p = ngs.GridFunction(domain.fes)
    p.Set(exact_solution_coeff,definedon=domain.fes.mesh.Boundaries(domain.bdr))
    a_bdr_val = domain.from_ngs( p )

    op = diffusion(
        domain, codomain, bdr_val=bdr_val,a_bdr_val = a_bdr_val
    )

    op_basics_wrapper(diffusion,domain,codomain,test_methods=True,bdr_val=bdr_val,a_bdr_val = a_bdr_val)

    op_evaluation_and_ot(op,sample_N=5,tolerance=1e-10,steps=[10**k for k in range(-5, -8, -1)],adjoint_derivative=False)