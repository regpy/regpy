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
        # self.mesh=self.fes.mesh

        self.fes_codomain = codomain.fes
        #        if dim==1:
        #            self.mesh = Make1DMesh(meshsize)
        #            self.fes = H1(self.mesh, order=2, dirichlet="left|right")
        #        elif dim==2:
        #            self.mesh = MakeQuadMesh(meshsize)
        #            self.fes = H1(self.mesh, order=2, dirichlet="left|top|right|bottom")

        # grid functions for later use
        self.gfu = ngs.GridFunction(self.fes_codomain)  # solution, return value of _eval
        self.gfu_bdr = ngs.GridFunction(self.fes_codomain)  # grid function holding boundary values

        self.gfu_integrator = ngs.GridFunction(
            self.fes_domain)  # grid function for defining integrator (bilinearform)
        self.gfu_integrator_codomain = ngs.GridFunction(self.fes_codomain)
        self.gfu_rhs = ngs.GridFunction(
            self.fes_codomain)  # grid function for defining right hand side (Linearform)

        self.gfu_inner_domain = ngs.GridFunction(
            self.fes_domain)  # grid function for reading in values in derivative
        self.gfu_inner = ngs.GridFunction(
            self.fes_codomain)  # grid function for inner computation in derivative and adjoint
        self.gfu_deriv = ngs.GridFunction(self.fes_codomain)  # return value of derivative
        self.gfu_toret = ngs.GridFunction(
            self.fes_domain)  # grid function for returning values in adjoint and derivative

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
            self.f_deriv += ngs.SymbolicLFI(-self.gfu_rhs * ngs.grad(self.gfu) * ngs.grad(v))

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
        self.gfu_integrator.vec.FV().NumPy()[:] = diff
        self.gfu_integrator_codomain.Set(self.gfu_integrator)
        #       self.gfu_integrator.Set(diff)
        self.a.Assemble()

        # Assemble Linearform
        self.gfu_rhs.Set(self.rhs)
        self.f.Assemble()

        # Update rhs by boundary values
        self.r.data = self.f.vec - self.a.mat * self.gfu_bdr.vec

        # Solve system
        self.gfu.vec.data = self.gfu_bdr.vec.data + self._solve(self.a, self.r)

        return self.gfu.vec.FV().NumPy().copy()

    #            return self.gfu

    def _derivative(self, argument):
        # Bilinearform already defined from _eval

        # Translate arguments in Coefficient Function
        self.gfu_inner_domain.vec.FV().NumPy()[:] = argument
        # Interpolate to codomain
        self.gfu_inner.Set(self.gfu_inner_domain)

        # Define rhs
        if self.diffusion:
            rhs = self.gfu_inner
            self.gfu_rhs.Set(rhs)
            self.f_deriv.Assemble()

            self.gfu_deriv.vec.data = self._solve(self.a, self.f_deriv.vec)

        elif self.reaction:
            rhs = self.gfu_inner * self.gfu
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
        self.gfu_inner.vec.data = self._solve(self.a, self.f.vec)

        if self.diffusion:
            res = -ngs.grad(self.gfu) * ngs.grad(self.gfu_inner)
        elif self.reaction:
            res = -self.gfu * self.gfu_inner

        self.gfu_toret.Set(res)

        return self.gfu_toret.vec.FV().NumPy().copy()

    def _solve(self, bilinear, rhs, boundary=False):
        return bilinear.mat.Inverse(freedofs=self.fes_codomain.FreeDofs()) * rhs


class EIT(Operator):
    """Electrical Impedance Tomography Problem

    PDE: -div(s grad u)=0       in Omega
         s du/dn = g            on dOmega

    Evaluate: F: s mapsto trace(u)
    Derivative:
        -div (s grad v)=div (h grad u) (=:f)
        s dv/dn = 0

    Der: F'[s]: h mapsto trace(v)

    Denote A: f mapsto trace(v)

    Adjoint:
        -div (s grad w)=0
        w=q

    Adj: q mapsto w mapsto grad(u) grad(w)

    proof:
    (f, A* q)=(f, w)=(-div(s grad v), w)=(grad v, s grad w)=(v, -div s grad w)
    +int[div (v s grad w)]=int(div (v s grad w))=int_dOmega [trace(v) s dw/dn]
    =int_dOmega [trace(v) q]=(Af, q)_(dOmega)

    WARNING: The last steps only hold if the pde for adjoint is: s dw/dn=q on dOmega
    instead!
    """

    def __init__(self, domain, g, codomain=None):
        codomain = codomain or domain
        self.g = g
        # self.pts=pts

        # Define mesh and finite element space
        # geo=SplineGeometry()
        # geo.AddCircle((0,0), 1, bc="circle")
        # ngmesh = geo.GenerateMesh()
        # ngmesh.Save('ngmesh')
        #        self.mesh=MakeQuadMesh(10)
        # self.mesh=Mesh(ngmesh)

        self.fes_domain = domain.fes
        self.fes_codomain = codomain.fes

        # Variables for setting of boundary values later
        # self.ind=[v.point in pts for v in self.mesh.vertices]
        self.pts = [v.point for v in self.fes_codomain.mesh.vertices]
        self.ind = [np.linalg.norm(np.array(p)) > 0.95 for p in self.pts]
        self.pts_bdr = np.array(self.pts)[self.ind]

        self.fes_in = ngs.H1(self.fes_codomain.mesh, order=1)
        self.gfu_in = ngs.GridFunction(self.fes_in)

        # grid functions for later use
        self.gfu = ngs.GridFunction(self.fes_codomain)  # solution, return value of _eval
        self.gfu_bdr = ngs.GridFunction(
            self.fes_codomain)  # grid function holding boundary values, g/sigma=du/dn

        self.gfu_integrator = ngs.GridFunction(
            self.fes_domain)  # grid function for defining integrator (bilinearform)
        self.gfu_integrator_codomain = ngs.GridFunction(self.fes_codomain)
        self.gfu_rhs = ngs.GridFunction(
            self.fes_codomain)  # grid function for defining right hand side (linearform), f

        self.gfu_inner_domain = ngs.GridFunction(
            self.fes_domain)  # grid function for reading in values in derivative
        self.gfu_inner = ngs.GridFunction(
            self.fes_codomain)  # grid function for inner computation in derivative and adjoint
        self.gfu_deriv = ngs.GridFunction(
            self.fes_codomain)  # gridd function return value of derivative
        self.gfu_toret = ngs.GridFunction(
            self.fes_domain)  # grid function for returning values in adjoint and derivative

        self.gfu_dir = ngs.GridFunction(
            self.fes_domain)  # grid function for solving the dirichlet problem in adjoint
        self.gfu_error = ngs.GridFunction(
            self.fes_codomain)  # grid function used in _target to compute the error in forward computation
        self.gfu_tar = ngs.GridFunction(
            self.fes_codomain)  # grid function used in _target, holding the arguments
        self.gfu_adjtoret = ngs.GridFunction(self.fes_domain)

        self.Number = ngs.NumberSpace(self.fes_codomain.mesh)
        r, s = self.Number.TnT()

        u = self.fes_codomain.TrialFunction()  # symbolic object
        v = self.fes_codomain.TestFunction()  # symbolic object

        # Define Bilinearform, will be assembled later
        self.a = ngs.BilinearForm(self.fes_codomain, symmetric=True)
        self.a += ngs.SymbolicBFI(ngs.grad(u) * ngs.grad(v) * self.gfu_integrator_codomain)

        ########new
        self.a += ngs.SymbolicBFI(u * s + v * r, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        self.fes1 = ngs.H1(self.fes_codomain.mesh, order=2,
                           definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        self.gfu_getbdr = ngs.GridFunction(self.fes1)
        self.gfu_setbdr = ngs.GridFunction(self.fes_codomain)

        # Define Linearform, will be assembled later
        self.f = ngs.LinearForm(self.fes_codomain)
        self.f += ngs.SymbolicLFI(self.gfu_rhs * v)

        self.r = self.f.vec.CreateVector()

        self.b = ngs.LinearForm(self.fes_codomain)
        self.gfu_b = ngs.GridFunction(self.fes_codomain)
        self.b += ngs.SymbolicLFI(self.gfu_b * v.Trace(),
                                  definedon=self.fes_codomain.mesh.Boundaries("cyc"))

        self.f_deriv = ngs.LinearForm(self.fes_codomain)
        self.f_deriv += ngs.SymbolicLFI(self.gfu_rhs * ngs.grad(self.gfu) * ngs.grad(v))

        #        self.b2=LinearForm(self.fes)
        #        self.b2+=SymbolicLFI(div(v*grad(self.gfu))

        super().__init__(domain, codomain)

    def _eval(self, diff, differentiate=False):
        # Assemble Bilinearform
        self.gfu_integrator.vec.FV().NumPy()[:] = diff
        self.gfu_integrator_codomain.Set(self.gfu_integrator)
        self.a.Assemble()

        # Assemble Linearform, boundary term
        self.gfu_b.Set(self.g)
        self.b.Assemble()

        # Solve system
        self.gfu.vec.data = self._solve(self.a, self.b.vec)

        # res=sco.minimize((lambda u: self._target(u, self.b.vec)), np.zeros(self.fes_codomain.ndof), constraints={"fun": self._constraint, "type": "eq"})

        if differentiate:
            sigma = ngs.CoefficientFunction(self.gfu_integrator)
            self.gfu_bdr.Set(self.g / sigma)

        # self.gfu.vec.FV().NumPy()[:]=res.x
        return self._get_boundary_values(self.gfu)

    def _derivative(self, h, **kwargs):
        # Bilinearform already defined from _eval

        # Translate arguments in Coefficient Function
        self.gfu_inner_domain.vec.FV().NumPy()[:] = h
        self.gfu_inner.Set(self.gfu_inner_domain)

        # Define rhs (f)
        rhs = self.gfu_inner
        self.gfu_rhs.Set(rhs)
        self.f_deriv.Assemble()

        # Define boundary term
        # self.gfu_b.Set(-self.gfu_inner*self.gfu_bdr)
        # self.b.Assemble()

        self.gfu_deriv.vec.data = self._solve(self.a, self.f_deriv.vec)  # +self.b.vec)

        # res=sco.minimize((lambda u: self._target(u, self.f.vec)), np.zeros(self.N_domain), constraints={"fun": self._constraint, "type": "eq"})

        # self.gfu_deriv.vec.FV().NumPy()[:]=res.x
        #        return res.x
        #        return self.gfu_toret.vec.FV().NumPy().copy()
        return self._get_boundary_values(self.gfu_deriv)

    def _adjoint(self, argument):
        # Bilinearform already defined from _eval

        # Definition of Linearform
        # But it only needs to be defined on boundary
        self._set_boundary_values(argument)
        # self.gfu_dir.Set(self.gfu_in)

        # Note: Here the linearform f for the dirichlet problem is just zero
        # Update for boundary values
        # self.r.data=-self.a.mat * self.gfu_dir.vec

        # Solve system
        # self.gfu_toret.vec.data=self.gfu_dir.vec.data+self._solve(self.a, self.r)

        # self.gfu_adjtoret.Set(-grad(self.gfu_toret)*grad(self.gfu))
        # return self.gfu_adjtoret.vec.FV().NumPy().copy()

        self.gfu_b.Set(self.gfu_in)
        self.b.Assemble()

        self.gfu_toret.vec.data = self._solve(self.a, self.b.vec)

        self.gfu_adjtoret.Set(-ngs.grad(self.gfu_toret) * ngs.grad(self.gfu))

        return self.gfu_adjtoret.vec.FV().NumPy().copy()

    def _solve(self, bilinear, rhs, boundary=False):
        return bilinear.mat.Inverse(freedofs=self.fes_codomain.FreeDofs()) * rhs

    def _get_boundary_values(self, gfu):
        #        myfunc=CoefficientFunction(gfu)
        #        vals = np.asarray([myfunc(self.fes_codomain.mesh(*p)) for p in self.pts_bdr])
        #        return vals
        self.gfu_getbdr.Set(0)
        self.gfu_getbdr.Set(gfu, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        return self.gfu_getbdr.vec.FV().NumPy().copy()

    def _set_boundary_values(self, vals):
        #        self.gfu_in.vec.FV().NumPy()[self.ind]=vals
        #        return
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
        self.gfu = ngs.GridFunction(self.fes_codomain)  # solution, return value of _eval
        # self.gfu_bdr=ngs.GridFunction(self.fes_codomain) #grid function holding boundary values, g/sigma=du/dn

        self.gfu_bilinearform = ngs.GridFunction(
            self.fes_domain)  # grid function for defining integrator (bilinearform)
        self.gfu_bilinearform_codomain = ngs.GridFunction(
            self.fes_codomain)  # grid function for defining integrator of bilinearform

        self.gfu_linearform_domain = ngs.GridFunction(
            self.fes_codomain)  # grid function for defining linearform
        self.gfu_linearform_codomain = ngs.GridFunction(self.fes_domain)

        self.gfu_deriv_toret = ngs.GridFunction(
            self.fes_codomain)  # grid function: return value of derivative

        self.gfu_adj = ngs.GridFunction(
            self.fes_domain)  # grid function for inner computation in adjoint
        self.gfu_adj_toret = ngs.GridFunction(
            self.fes_domain)  # grid function: return value of adjoint

        self.gfu_b = ngs.GridFunction(
            self.fes_codomain)  # grid function for defining the boundary term

        u = self.fes_codomain.TrialFunction()  # symbolic object
        v = self.fes_codomain.TestFunction()  # symbolic object

        # Define Bilinearform, will be assembled later
        self.a = ngs.BilinearForm(self.fes_codomain, symmetric=True)
        self.a += ngs.SymbolicBFI(
            -ngs.grad(u) * ngs.grad(v) + u * v * self.gfu_bilinearform_codomain)

        # Interaction with Trace
        self.fes_bdr = ngs.H1(self.fes_codomain.mesh, order=self.fes_codomain.globalorder,
                              definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        self.gfu_getbdr = ngs.GridFunction(self.fes_bdr)
        self.gfu_setbdr = ngs.GridFunction(self.fes_codomain)

        # Boundary term
        self.b = ngs.LinearForm(self.fes_codomain)
        self.b += ngs.SymbolicLFI(-self.gfu_b * v.Trace(),
                                  definedon=self.fes_codomain.mesh.Boundaries("cyc"))

        # Linearform (only appears in derivative)
        self.f_deriv = ngs.LinearForm(self.fes_codomain)
        self.f_deriv += ngs.SymbolicLFI(-self.gfu_linearform_codomain * self.gfu * v)

        super().__init__(domain, codomain)

    def _eval(self, diff, differentiate=False):
        # Assemble Bilinearform
        self.gfu_bilinearform.vec.FV().NumPy()[:] = diff
        self.gfu_bilinearform_codomain.Set(self.gfu_bilinearform)
        self.a.Assemble()

        # Assemble Linearform of boundary term
        self.gfu_b.Set(self.g)
        self.b.Assemble()

        # Solve system
        self.gfu.vec.data = self._solve(self.a, self.b.vec)

        # if differentiate:
        #    sigma=CoefficientFunction(self.gfu_integrator)
        #    self.gfu_bdr.Set(self.g/sigma)

        return self._get_boundary_values(self.gfu)

    def _derivative(self, h):
        # Bilinearform already defined from _eval

        # Translate arguments in Coefficient Function
        self.gfu_linearform_domain.vec.FV().NumPy()[:] = h
        self.gfu_linearform_codomain.Set(self.gfu_linearform_domain)

        # Define rhs
        self.f_deriv.Assemble()

        # Define boundary term, often ignored
        # self.gfu_b.Set(-self.gfu_linearform_codomain*self.gfu_bdr)
        # self.b.Assemble()

        # Solve system
        self.gfu_deriv_toret.vec.data = self._solve(self.a, self.f_deriv.vec)

        return self._get_boundary_values(self.gfu_deriv_toret)

    def _adjoint(self, argument):
        # Bilinearform already defined from _eval

        # Definition of Linearform
        # But it only needs to be defined on boundary
        self._set_boundary_values(argument)

        self.gfu_b.Set(self.gfu_in)
        self.b.Assemble()

        self.gfu_adj.vec.data = self._solve(self.a, self.b.vec)

        self.gfu_adj_toret.Set(self.gfu_adj * self.gfu)

        return self.gfu_adj_toret.vec.FV().NumPy().copy()

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
