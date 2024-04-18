import ngsolve as ngs
from regpy.operators.ngsolve import NGSolveOperator


class TFM(NGSolveOperator):
    '''Traction Force Microscopy Problem

        PDE: - div (sigma(u)) = 0    in Omega
             sigma(u) n = t       on dOmega

             sigma(u) = 2 \mu epsilon + \lambda tr( epsilon) I
             epsilon = 0.5 (grad u + (grad u)^T )

        Evaluate: F: t \mapsto u

        Adjoint: (weak formulation with test function w)

        int_{Omega} sigma(w) : grad(phi) dx = <v,w>_{H¹_0(Omega)}

        Adj: F[s]^*: v \mapsto tr(phi)'''
    def __init__(self, domain, codomain, mu, lam):
        codomain = codomain
        # Need to know the boundary to calculate Neumann bdr condition
        assert domain.bdr is not None
        super().__init__(domain, codomain, linear=True)
        # From NgSolveOperator
        #self.gfu_read_in = ngs.GridFunction(self.domain.fes)

        # Lamé Parameters for substrate
        self.mu = mu
        self.lam = lam

        self.fes_domain = domain.fes
        self.fes_codomain = codomain.fes


        # grid functions for later use
        self.gfu_eval = ngs.GridFunction(self.fes_codomain)  # solution, return value of _eval
        self.gfu_adjoint = ngs.GridFunction(self.fes_domain) # grid function return value of adjoint (trace of gfu_adjoint_sol)

        self.gfu_bf = ngs.GridFunction(self.fes_codomain)  # grid function for defining integrator (bilinearform)
        self.gfu_lf = ngs.GridFunction(self.fes_codomain) # grid function for defining right hand side (linearform), f



        u, v = self.fes_codomain.TnT()

        # Define Bilinearform, will be assembled later
        self.a = ngs.BilinearForm(self.fes_codomain, symmetric=True)
        self.a += (2 * mu * ngs.InnerProduct(self._strain(u), self._strain(v)) + lam * ngs.div(u) * ngs.div(v)) * ngs.dx


        # Define Linearform for evaluation, will be assembled later
        self.b = ngs.LinearForm(self.fes_domain)
        self.b += self.gfu_lf * v * ngs.ds(domain.bdr)


        # Define linearform to trick ngsolve for computation of discrete adjoint
        self.b_help = ngs.LinearForm(self.fes_domain)
        self.b_help.Assemble()


        # Initialize preconditioner for solving the Dirichlet problems by ngs.BVP
        self.prec = ngs.Preconditioner(self.a, 'direct')
        self.a.Assemble()


    # Left term: Bilinearform self.a
    # Right term: Linearform self.b
    def _eval(self, traction, differentiate=False):

        # Assemble Linearform, boundary term
        self._read_in(traction, self.gfu_lf)
        self.b.Assemble()

        self._solve_dirichlet_problem(bf=self.a, lf=self.b, gf=self.gfu_eval, prec=self.prec)

        return self.gfu_eval.vec.FV().NumPy()[:].copy()


    def _adjoint(self, displacement):
        # Bilinearform already assembled in init -> initialization with 0, s.t. object exists
        # Diskrete Adjoint w.r.t. standard inner product

        self.b_help.vec.FV().NumPy()[:] = displacement.copy()
        self._solve_dirichlet_problem(bf=self.a, lf=self.b_help, gf=self.gfu_adjoint, prec=self.prec)
        self._read_in(self.gfu_adjoint.vec.FV().NumPy()[:].copy(), self.gfu_lf)
        self.b.Assemble()

        return self.b.vec.FV().NumPy()[:].copy()

    def _strain(self,u):
        return 0.5 * (ngs.Grad(u) + ngs.Grad(u).trans)
