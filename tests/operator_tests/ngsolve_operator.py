import ngsolve as ngs
from netgen.geom2d import unit_square
import pytest

from regpy.vecsps.ngsolve import *
from regpy.operators.ngsolve import *

from .base_operator import op_basics_wrapper,op_evaluation_and_ot, op_basics
from regpy.util import set_rng_seed, operator_tests

set_rng_seed(15873098306879350073259142812684978477)

def test_basic_NgsOperator():
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    vs = NgsVectorSpace(fes,bdr=bdr)

    op_basics_wrapper(NgsOperator,vs,vs)

class diffusion(SecondOrderEllipticCoefficientPDE):
#taken from diffusion example
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

class TestSecondOrderEllipticCoefficientPDE():
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes_domain = ngs.H1(mesh, order=6, dirichlet = bdr)
    domain = NgsVectorSpace(fes_domain,bdr = bdr)

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

    def test_basics(self):
        op_basics_wrapper(diffusion,self.domain,self.codomain,test_methods=True,bdr_val=self.bdr_val,a_bdr_val = self.a_bdr_val)

    op = diffusion(
        domain, codomain, bdr_val=bdr_val,a_bdr_val = a_bdr_val
    )

    def test_fixed_coefficient_boundary_is_recognized(self):
        """Regression: fixed coefficient values must survive construction."""
        import numpy as np

        free_dofs = self.domain.fes.FreeDofs()
        free = np.array([bool(free_dofs[i]) for i in
                         range(self.domain.fes.ndof)], dtype=bool)
        expected = np.asarray(self.domain.flatten(self.a_bdr_val))
        actual = np.asarray(self.op.a_bdr.FV().NumPy())
        assert np.any(~free)
        assert np.any(np.abs(expected[~free]) > 0.)
        np.testing.assert_allclose(actual[~free], expected[~free],
                                   rtol=0., atol=1e-12)
        np.testing.assert_allclose(actual[free], 0., rtol=0., atol=1e-12)

    def test_derivative_full_dof_euclidean_adjoint(self):
        """Fixed coefficient boundary values must not enter the derivative adjoint.

        The derivative ignores perturbations at constrained coefficient DOFs.
        Its Euclidean adjoint must therefore have zero constrained
        coefficients, even when its argument contains boundary values.
        """
        import numpy as np

        a = self.domain.from_ngs(1 + 0.2 * ngs.exp(
            -20 * ((ngs.x - 0.4)**2 + (ngs.y - 0.6)**2)))
        _, deriv = self.op.linearize(a)

        # A genuine linear adjoint sends the zero covector to zero.
        adj_zero = deriv.adjoint(self.codomain.zeros())
        np.testing.assert_allclose(
            self.domain.flatten(adj_zero), 0., rtol=0., atol=2e-10)

        free = np.asarray(list(self.domain.fes.FreeDofs()), dtype=bool)
        assert np.any(~free)
        h = self.domain.randn(
            seed=1234, use_space_sampling=True,
            mass_matrix_weighting=False)  # deliberately NOT projected
        g = self.codomain.randn(
            seed=5678, use_space_sampling=True,
            mass_matrix_weighting=False)
        h_coeff = self.domain.flatten(h)
        assert np.any(np.abs(h_coeff[~free]) > 0.)

        # Derivative must ignore h on the fixed Dirichlet DOFs.
        h_free = self.domain.fromflat(np.where(free, h_coeff, 0.))
        np.testing.assert_allclose(
            self.codomain.flatten(deriv(h)),
            self.codomain.flatten(deriv(h_free)), rtol=1e-8, atol=1e-9)

        # Its adjoint maps to the same free subspace and satisfies the
        # full-coordinate duality pairing, not just a projected-input test.
        adj = deriv.adjoint(g)
        np.testing.assert_allclose(
            self.domain.flatten(adj)[~free], 0., rtol=0., atol=2e-10)
        lhs = self.codomain.vdot(deriv(h), g)
        rhs = self.domain.vdot(h, adj)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-9)

    def test_ot_eval(self):
        coords = self.domain._draw_sample("uniform",10).reshape((5,2))
        x_s = [self.domain.from_ngs(1 + 0.5 * ngs.exp(-20*((ngs.x-c[0])**2 + (ngs.y-c[1])**2))) for c in coords
        ]
        op_evaluation_and_ot(self.op,sample_N=5,tolerance=1e-10,steps=[10**k for k in range(-1, -6, -1)],adjoint_derivative=False, x_s=x_s)

class TestProjectToBoundary():
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    bdr = "left|top|right|bottom"
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    codomain = NgsVectorSpace(fes,bdr = bdr)
    fes_d = ngs.H1(mesh, order=3)
    domain = NgsVectorSpace(fes_d)
    

    @pytest.mark.parametrize("bdr",[ 
        None,
        "left|top|right",
        mesh.Boundaries("left|right"),
        fes.GetDofs(mesh.Boundaries("top|bottom"))
    ])
    def test_with_same(self, bdr):
        op = ProjectToBoundary(self.codomain,bdr = bdr)
        op_basics(op,test_methods=True,test_norm=False)
        op_evaluation_and_ot(op)

    @pytest.mark.parametrize("bdr",[ 
        None,
        "left|top|right",
        mesh.Boundaries("left|right")
    ])
    def test_with_different(self, bdr):
        op = ProjectToBoundary(domain=self.domain,codomain=self.codomain,bdr = bdr)
        op_basics(op,test_methods=True,test_norm=False)
        op_evaluation_and_ot(op)

class TestNgsGradOP():
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
    bdr = "left|top|right|bottom"
    fes = ngs.H1(mesh, order=6, dirichlet = bdr)
    domain = NgsVectorSpace(fes,bdr = bdr)
    
    def test_basics(self):
        op_basics(NgsGradOP(self.domain),test_methods=True,test_norm = False)
    
    def test_evaluation_ot(self):
        op = NgsGradOP(self.domain)
        print(operator_tests.test_operator(op))
        op_evaluation_and_ot(op,)

class TestSolveSystem():
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    bdr = "left|top|right|bottom"

    # Scalar FE space with homogeneous Dirichlet BC
    fes = ngs.H1(mesh, order=3, dirichlet=bdr)
    domain = NgsVectorSpace(fes, bdr=bdr)

    # Bilinear form for Laplacian
    u, v = fes.TnT()
    bf = ngs.BilinearForm(fes, symmetric=True)
    bf += ngs.grad(u) * ngs.grad(v) * ngs.dx
    bf.Assemble()

    def test_basics(self):
        """Basic operator properties"""
        op = SolveSystem(self.domain, self.bf)
        op_basics(op, test_methods=True, test_norm=False)

    def test_evaluation_and_adjoint(self):
        """
        Solve -Δu = f with zero Dirichlet BC and test against
        an analytic solution.
        """
        op = SolveSystem(self.domain, self.bf, use_prec=False, inverse = "pardiso")

        # Analytic solution u = sin(πx) sin(πy)
        u_exact = ngs.sin(ngs.pi * ngs.x) * ngs.sin(ngs.pi * ngs.y)

        # Corresponding RHS: f = -Δu = 2π² sin(πx) sin(πy)
        f_rhs = 2 * ngs.pi**2 * u_exact

        # regpy vectors
        x = self.domain.from_ngs(f_rhs)
        res = self.domain.from_ngs(u_exact)

        # Test evaluation and adjoint consistency
        op_evaluation_and_ot(op, x=x, res=res,tol=3e-2)

class TestLinearForm:
    # Create a simple mesh and FE space
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    bdr = "left|top|right|bottom"
    fes = ngs.H1(mesh, order=3, dirichlet=bdr)
    domain = NgsVectorSpace(fes, bdr=bdr)

    def test_basics(self):
        """Check basic operator properties."""
        op = LinearForm(self.domain)
        op_basics(op, test_methods=True, test_norm=False)

    def test_evaluation_and_adjoint(self):
        """Test evaluation and Euclidean adjoint consistency."""
        op = LinearForm(self.domain)

        x = self.domain.randn()
        u, v = self.domain.fes.TnT()
        bf_mass = ngs.BilinearForm(self.domain.fes)
        bf_mass += u * v * ngs.dx
        bf_mass.Assemble()  # assemble into bf_mass.mat
        res = NgsBaseVector(bf_mass.mat * x.vec)

        # Test evaluation and adjoint consistency
        op_evaluation_and_ot(op, x=x, res=res)

class TestLinearFormGrad:
    # Create mesh and FE space
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    bdr = "left|top|right|bottom"
    fes = ngs.H1(mesh, order=3, dirichlet=bdr)
    domain = NgsVectorSpace(fes, bdr=bdr)

    # Create a GridFunction to evaluate gradients against
    gfu_eval = ngs.GridFunction(fes)
    gfu_eval.Set(ngs.sin(ngs.pi * ngs.x) * ngs.sin(ngs.pi * ngs.y))

    def test_basics(self):
        op = LinearFormGrad(self.domain, self.gfu_eval)
        op_basics(op, test_methods=True, test_norm=False)

    def test_evaluation_and_adjoint(self):
        op = LinearFormGrad(self.domain, self.gfu_eval)
        op_evaluation_and_ot(op)