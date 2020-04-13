"""PDE forward operators using NGSolve
"""

import ngsolve as ngs
import numpy as np

from regpy.operators import Operator


class Coefficient(Operator):
    # TODO: Use netgen for visualization instead of own functions
    # TODO: Further optimization, introduction of solve function, preconditioner, fewer gridfunctions ....
    # TODO: Maybe use gridfunctions and not coefficient-vectors as input and output

    def __init__(
        self, domain, rhs, bc_left=None, bc_right=None, bc_top=None, bc_bottom=None, codomain=None,
        diffusion=True, reaction=False, dim=1
    ):
        assert dim in (1, 2)
        assert diffusion or reaction

        codomain = codomain or domain
        self.rhs = rhs

        self.diffusion = diffusion
        self.reaction = reaction
        self.dim = domain.fes.mesh.dim

        bc_left = bc_left or 0
        bc_right = bc_right or 0
        bc_top = bc_top or 0
        bc_bottom = bc_bottom or 0

        # Define mesh and finite element space
        self.fes_domain = domain.fes
        self.fes_codomain = codomain.fes

        # grid functions for later use
        self.gfu_eval = ngs.GridFunction(self.fes_codomain)  # solution, return value of _eval
        self.gfu_deriv = ngs.GridFunction(self.fes_codomain)  # return value of derivative
        self.gfu_adjoint = ngs.GridFunction(self.fes_domain)  # grid function for returning values in adjoint

        self.gfu_bdr = ngs.GridFunction(self.fes_codomain)  # grid function holding boundary values

        self.gfu_integrator_domain = ngs.GridFunction(self.fes_domain)  # grid function for defining integrator (bilinearform)
        self.gfu_integrator_codomain = ngs.GridFunction(self.fes_codomain)
        self.gfu_rhs = ngs.GridFunction(
            self.fes_codomain)  # grid function for defining right hand side (Linearform)

        self.gfu_inner_domain = ngs.GridFunction(self.fes_domain)  # grid function for reading in values in derivative
        self.gfu_inner_codomain = ngs.GridFunction(
            self.fes_codomain)  # grid function for inner computation in derivative and adjoint

        u = self.fes_codomain.TrialFunction()  # symbolic object
        v = self.fes_codomain.TestFunction()  # symbolic object

        # Define Bilinearform, will be assembled later
        self.a = ngs.BilinearForm(self.fes_codomain, symmetric=True)
        if self.diffusion:
            self.a += ngs.SymbolicBFI(ngs.grad(u) * ngs.grad(v) * self.gfu_integrator_codomain)
        elif self.reaction:
            self.a += ngs.SymbolicBFI(
                ngs.grad(u) * ngs.grad(v) + u * v * self.gfu_integrator_codomain)

        # Define Linearform, will be assembled later
        self.f = ngs.LinearForm(self.fes_codomain)
        self.f += ngs.SymbolicLFI(self.gfu_rhs * v)

        if diffusion:
            self.f_deriv = ngs.LinearForm(self.fes_codomain)
            self.f_deriv += ngs.SymbolicLFI(-self.gfu_rhs * ngs.grad(self.gfu_eval) * ngs.grad(v))

        # Precompute Boundary values and boundary valued corrected rhs
        if self.dim == 1:
            self.gfu_bdr.Set([bc_left, bc_right],
                             definedon=self.fes_codomain.mesh.Boundaries("left|right"))
        elif self.dim == 2:
            self.gfu_bdr.Set([bc_left, bc_top, bc_right, bc_bottom],
                             definedon=self.fes_codomain.mesh.Boundaries("left|top|right|bottom"))
        self.r = self.f.vec.CreateVector()

        super().__init__(domain, codomain)

    def _eval(self, diff, differentiate=False):
        # Assemble Bilinearform
        self.gfu_integrator_domain.vec.FV().NumPy()[:] = diff
        self.gfu_integrator_codomain.Set(self.gfu_integrator)
        self.a.Assemble()

        # Assemble Linearform
        self.gfu_rhs.Set(self.rhs)
        self.f.Assemble()

        # Update rhs by boundary values
        self.r.data = self.f.vec - self.a.mat * self.gfu_bdr.vec

        # Solve system
        self.gfu_eval.vec.data = self.gfu_bdr.vec.data + self._solve(self.a, self.r)

        return self.gfu_eval.vec.FV().NumPy().copy()

    def _derivative(self, argument):
        # Bilinearform already defined from _eval

        # Translate arguments in Coefficient Function
        self.gfu_inner_domain.vec.FV().NumPy()[:] = argument
        # Interpolate to codomain
        self.gfu_inner_codomain.Set(self.gfu_inner_domain)

        # Define rhs
        if self.diffusion:
            rhs = self.gfu_inner_codomain
            self.gfu_rhs.Set(rhs)
            self.f_deriv.Assemble()

            self.gfu_deriv.vec.data = self._solve(self.a, self.f_deriv.vec)

        elif self.reaction:
            rhs = self.gfu_inner_codomain * self.gfu_eval
            self.gfu_rhs.Set(rhs)
            self.f.Assemble()

            self.gfu_deriv.vec.data = self._solve(self.a, self.f.vec)

        return self.gfu_deriv.vec.FV().NumPy().copy()

    def _adjoint(self, argument):
        # Bilinearform already defined from _eval

        # Definition of Linearform
        self.gfu_rhs.vec.FV().NumPy()[:] = argument
        #       self.gfu_rhs.Set(rhs)
        self.f.Assemble()

        # Solve system
        self.gfu_inner_codomain.vec.data = self._solve(self.a, self.f.vec)

        if self.diffusion:
            res = -ngs.grad(self.gfu_eval) * ngs.grad(self.gfu_inner_codomain)
        elif self.reaction:
            res = -self.gfu_eval * self.gfu_inner_codomain

        self.gfu_adjoint.Set(res)

        return self.gfu_adjoint.vec.FV().NumPy().copy()

    def _solve(self, bilinear, rhs, boundary=False):
        return bilinear.mat.Inverse(freedofs=self.fes_codomain.FreeDofs()) * rhs


class EIT(Operator):
    """Electrical Impedance Tomography Problem

    PDE: -div(s grad u)+alpha*u=0       in Omega
         s du/dn = g            on dOmega

    Evaluate: F: s \mapsto trace(u)
    Derivative:
        -div (s grad v)+alpha*v=div (h grad u) (=:f)
        s dv/dn = 0+(-h du/dn) [second term often omitted]

    Der: F'[s]: h \mapsto trace(v)

    Adjoint:
        -div (s grad w)+alpha*w=0
        s dw/dn=q

    Adj: F'[s]^*: q \mapsto -grad(u) grad(w)

    proof:
    (F'h, q)=int_dOmega [trace(v) q] = int_dOmega [trace(v) s dw/dn] = int_Omega [div(v s grad w )]
    Note div(s grad w) = alpha*w, thus above equation shows:
    (F'h, q) = (s grad v, grad w)+alpha (v, w) = int_Omega [div( s grad v w)] +(-div (s grad v)), w)+alpha (v, w)
    = int_dOmega [s dv/dn trace(w)]+(f, w) = (f, w)-int_dOmega [trace(w) h du/dn]
    = (h, -grad u grad w) + int_Omega [div(h grad u w)]-int_dOmega [trace(w) h du/dn]
    The last two terms are the same! It follows: (F'h, q) = (h, -grad u grad w). Hence:
    Adjoint: q \mapsto -grad u grad w
    """

    def __init__(self, domain, g, codomain=None, alpha=0.01):
        codomain = codomain or domain
        self.g = g

        self.fes_domain = domain.fes
        self.fes_codomain = codomain.fes

        # Variables for setting of boundary values later
        self.pts = [v.point for v in self.fes_codomain.mesh.vertices]
        self.ind = [np.linalg.norm(np.array(p)) > 0.95 for p in self.pts]
        self.pts_bdr = np.array(self.pts)[self.ind]

        #FES and Grid Function for reading in values
        self.fes_in = ngs.H1(self.fes_codomain.mesh, order=1)
        self.gfu_in = ngs.GridFunction(self.fes_in)

        # grid functions for later use
        self.gfu_eval = ngs.GridFunction(self.fes_codomain)  # solution, return value of _eval
        self.gfu_deriv = ngs.GridFunction(self.fes_codomain)  # grid function return value of derivative
        self.gfu_adjoint = ngs.GridFunction(self.fes_domain) #grid function return value of adjoint

        self.gfu_bdr = ngs.GridFunction(self.fes_codomain)  # grid function holding boundary values, g/sigma=du/dn

        self.gfu_integrator_domain = ngs.GridFunction(self.fes_domain)  # grid function for defining integrator (bilinearform)
        self.gfu_integrator_codomain = ngs.GridFunction(self.fes_codomain)
        self.gfu_rhs = ngs.GridFunction(self.fes_codomain)  # grid function for defining right hand side (linearform), f

        self.gfu_inner_domain = ngs.GridFunction(self.fes_domain)  # grid function for reading in values in derivative
        self.gfu_inner_codomain = ngs.GridFunction(self.fes_codomain)  # grid function for inner computation in derivative
        self.gfu_inner_adjoint = ngs.GridFunction(self.fes_domain)  # grid function for inner computations in adjoint

        self.Number = ngs.NumberSpace(self.fes_codomain.mesh)
        r, s = self.Number.TnT()

        u = self.fes_codomain.TrialFunction()  # symbolic object
        v = self.fes_codomain.TestFunction()  # symbolic object

        # Define Bilinearform, will be assembled later
        self.a = ngs.BilinearForm(self.fes_codomain, symmetric=True)
        self.a += ngs.SymbolicBFI(ngs.grad(u) * ngs.grad(v) * self.gfu_integrator_codomain+alpha*u*v)

        #Additional condition: The integral along the boundary vanishes
        #self.a += ngs.SymbolicBFI(u * s + v * r, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        self.fes1 = ngs.H1(self.fes_codomain.mesh, order=4, definedon=self.fes_codomain.mesh.Boundaries("cyc"))

        #Grid Functions for projecting solutions to Boundary values or reading them in from boundary
        self.gfu_getbdr = ngs.GridFunction(self.fes1)
        self.gfu_setbdr = ngs.GridFunction(self.fes_codomain)

        # Define Linearform for evaluation, will be assembled later
        
            # Define Linearform, will be assembled later
            #self.f = ngs.LinearForm(self.fes_codomain)
            #self.f += ngs.SymbolicLFI(self.gfu_rhs * v)
            #self.r = self.f.vec.CreateVector()
            #self.gfu_dir = ngs.GridFunction(self.fes_domain)  # grid function for solving the dirichlet problem in adjoint

        self.b = ngs.LinearForm(self.fes_codomain)
        self.gfu_b = ngs.GridFunction(self.fes_codomain)
        #self.b += ngs.SymbolicLFI(self.gfu_b * v.Trace(), definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        self.b += self.gfu_b*v*ngs.ds("cyc")

        # Define Linearform for derivative, will be assembled later
        self.f_deriv = ngs.LinearForm(self.fes_codomain)
        self.f_deriv += ngs.SymbolicLFI(-self.gfu_rhs * ngs.grad(self.gfu_eval) * ngs.grad(v))

        super().__init__(domain, codomain)

#Weak formulation:
#0=int_Omega [-div(s grad u) v + alpha u v]=-int_dOmega [s du/dn trace(v)]+int_Omega [s grad u grad v + alpha u v]
#Hence: int_Omega [s grad u grad v + alpha u v] = int_dOmega [g trace(v)]
#Left term: Bilinearform self.a
#Righ term: Linearform self.b
    def _eval(self, diff, differentiate=False):
        # Assemble Bilinearform
        self.gfu_integrator_domain.vec.FV().NumPy()[:] = diff
        self.gfu_integrator_codomain.Set(self.gfu_integrator_domain)
        self.a.Assemble()

        # Assemble Linearform, boundary term
        self.gfu_b.Set(0)
        self.gfu_b.Set(self.g, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        self.b.Assemble()

        # Solve system
        self.gfu_eval.vec.data = self._solve(self.a, self.b.vec)

        #If we differentiate, then remember du/dn
        if differentiate:
            sigma = ngs.CoefficientFunction(self.gfu_integrator_domain)
            self.gfu_bdr.Set(self.g / sigma)

        return self.gfu_eval.vec.FV().NumPy().copy()
        #return self._get_boundary_values(self.gfu_eval)

#Weak Formulation:
#0 = int_Omega [-div(s grad v) w + alpha v w]-int_Omega [div (h grad u) w]
#=-int_dOmega [s dv/dn trace(w)] + int_Omega [s grad v grad w + alpha v w]-int_dOmega [h du/dn trace(w)]+int_Omega [h grad u grad w]
#=int_Omega [s grad v grad w + alpha v w]+int_Omega [h grad u grad w]
#Hence: int_Omega [s grad v grad w + alpha v w] = int_Omega [-h grad u grad w]
#Left Term: Bilinearform self.a, already defined in _eval
#Right Term: Linearform f_deriv
    def _derivative(self, h, **kwargs):
        # Bilinearform already defined from _eval

        # Translate arguments in Coefficient Function
        self.gfu_inner_domain.vec.FV().NumPy()[:] = h
        self.gfu_inner_codomain.Set(self.gfu_inner_domain)

        # Define rhs (f)
        rhs = self.gfu_inner_codomain
        self.gfu_rhs.Set(rhs)
        self.f_deriv.Assemble()

        # Define boundary term
        #self.gfu_b.Set(0)
        #self.gfu_b.Set(-self.gfu_inner_codomain*self.gfu_bdr, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        #self.b.Assemble()

        self.gfu_deriv.vec.data = self._solve(self.a, self.f_deriv.vec)#+self._solve(self.a, self.b.vec)

        return self._get_boundary_values(self.gfu_deriv)

#Same problem as in _eval
    def _adjoint(self, argument):
        # Bilinearform already defined from _eval

        # Definition of Linearform
        # But it only needs to be defined on boundary
        self._set_boundary_values(argument)

            #Dirichlet Problem
            #self.gfu_dir.Set(self.gfu_in)

            # Note: Here the linearform f for the dirichlet problem is just zero
            # Update for boundary values
            #self.r.data=-self.a.mat * self.gfu_dir.vec

            # Solve system
            #self.gfu_inner_adjoint.vec.data=self.gfu_dir.vec.data+self._solve(self.a, self.r)

            #self.gfu_adjoint.Set(-ngs.grad(self.gfu_inner_adjoint)*ngs.grad(self.gfu_eval))
            #return self.gfu_adjoint.vec.FV().NumPy().copy()


        self.gfu_b.Set(self.gfu_in)
        self.b.Assemble()

        self.gfu_inner_adjoint.vec.data = self._solve(self.a, self.b.vec)

        self.gfu_adjoint.Set(-ngs.grad(self.gfu_inner_adjoint) * ngs.grad(self.gfu_eval))

        return self.gfu_adjoint.vec.FV().NumPy().copy()

    def _solve(self, bilinear, rhs, boundary=False):
        return bilinear.mat.Inverse(freedofs=self.fes_codomain.FreeDofs()) * rhs

    def _get_boundary_values(self, gfu):
        self.gfu_getbdr.Set(0)
        self.gfu_getbdr.Set(gfu, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        return self.gfu_getbdr.vec.FV().NumPy().copy()

    def _set_boundary_values(self, vals):
        self.gfu_setbdr.vec.FV().NumPy()[:] = vals
        self.gfu_in.Set(0)
        self.gfu_in.Set(self.gfu_setbdr, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        return


class ReactionBoundary(Operator):
    def __init__(self, domain, g, codomain=None):
        codomain = codomain or domain
        self.g = g

        self.fes_domain = domain.fes
        self.fes_codomain = codomain.fes

        self.fes_in = ngs.H1(self.fes_codomain.mesh, order=1)
        self.gfu_in = ngs.GridFunction(self.fes_in)

        # grid functions for later use
        self.gfu_eval = ngs.GridFunction(self.fes_codomain)  # solution, return value of _eval
        self.gfu_deriv = ngs.GridFunction(self.fes_codomain)  # grid function: return value of derivative
        self.gfu_adjoint = ngs.GridFunction(self.fes_domain)  # grid function: return value of adjoint

        self.gfu_bilinearform_domain = ngs.GridFunction(self.fes_domain)  # grid function for defining integrator (bilinearform)
        self.gfu_bilinearform_codomain = ngs.GridFunction(self.fes_codomain)  # grid function for defining integrator of bilinearform

        self.gfu_linearform_domain = ngs.GridFunction(self.fes_codomain)  # grid function for defining linearform
        self.gfu_linearform_codomain = ngs.GridFunction(self.fes_domain)

        self.gfu_b = ngs.GridFunction(self.fes_codomain)  # grid function for defining the boundary term

        self.gfu_inner_adjoint = ngs.GridFunction(self.fes_domain)  # grid function for inner computation in adjoint

        u = self.fes_codomain.TrialFunction()  # symbolic object
        v = self.fes_codomain.TestFunction()  # symbolic object

        # Define Bilinearform, will be assembled later
        self.a = ngs.BilinearForm(self.fes_codomain, symmetric=True)
        self.a += ngs.SymbolicBFI(-ngs.grad(u) * ngs.grad(v) + u * v * self.gfu_bilinearform_codomain)

        # Interaction with Trace
        self.fes_bdr = ngs.H1(self.fes_codomain.mesh, order=self.fes_codomain.globalorder, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        self.gfu_getbdr = ngs.GridFunction(self.fes_bdr)
        self.gfu_setbdr = ngs.GridFunction(self.fes_codomain)

        # Boundary term
        self.b = ngs.LinearForm(self.fes_codomain)
        self.b += ngs.SymbolicLFI(-self.gfu_b * v.Trace(),
                                  definedon=self.fes_codomain.mesh.Boundaries("cyc"))

        # Linearform (only appears in derivative)
        self.f_deriv = ngs.LinearForm(self.fes_codomain)
        self.f_deriv += ngs.SymbolicLFI(-self.gfu_linearform_codomain * self.gfu_eval * v)

        super().__init__(domain, codomain)

    def _eval(self, diff, differentiate=False):
        # Assemble Bilinearform
        self.gfu_bilinearform_domain.vec.FV().NumPy()[:] = diff
        self.gfu_bilinearform_codomain.Set(self.gfu_bilinearform_domain)
        self.a.Assemble()

        # Assemble Linearform of boundary term
        self.gfu_b.Set(self.g)
        self.b.Assemble()

        # Solve system
        self.gfu_eval.vec.data = self._solve(self.a, self.b.vec)

        return self._get_boundary_values(self.gfu_eval)

    def _derivative(self, h):
        # Bilinearform already defined from _eval

        # Translate arguments in Coefficient Function
        self.gfu_linearform_domain.vec.FV().NumPy()[:] = h
        self.gfu_linearform_codomain.Set(self.gfu_linearform_domain)

        # Define rhs
        self.f_deriv.Assemble()

        # Boundary term, often ignored

        # Solve system
        self.gfu_deriv.vec.data = self._solve(self.a, self.f_deriv.vec)

        return self._get_boundary_values(self.gfu_deriv)

    def _adjoint(self, argument):
        # Bilinearform already defined from _eval

        # Definition of Linearform
        # But it only needs to be defined on boundary
        self._set_boundary_values(argument)

        self.gfu_b.Set(self.gfu_in)
        self.b.Assemble()

        self.gfu_inner_adjoint.vec.data = self._solve(self.a, self.b.vec)

        self.gfu_adjoint.Set(self.gfu_inner_adjoint * self.gfu_eval)

        return self.gfu_adjoint.vec.FV().NumPy().copy()

    def _solve(self, bilinear, rhs, boundary=False):
        return bilinear.mat.Inverse(freedofs=self.fes_codomain.FreeDofs()) * rhs

    def _get_boundary_values(self, gfu):
        self.gfu_getbdr.Set(0)
        self.gfu_getbdr.Set(gfu, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        return self.gfu_getbdr.vec.FV().NumPy().copy()

    def _set_boundary_values(self, vals):
        self.gfu_setbdr.vec.FV().NumPy()[:] = vals
        self.gfu_in.Set(0)
        self.gfu_in.Set(self.gfu_setbdr, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        return
