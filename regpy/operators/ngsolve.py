"""PDE forward operators using NGSolve
"""

import ngsolve as ngs
import numpy as np

from regpy.operators import Operator

class NGSolveOperator(Operator):
    def __init__(self, domain, codomain, linear = False):
        super().__init__(domain = domain, codomain = codomain, linear = linear)
        self.gfu_read_in = ngs.GridFunction(self.domain.fes)

    def _read_in(self, vector, gfu):
        """
        Reads in a coefficient vector of the domain and interpolates in the codomain.
        The result is saved in `gfu`.
        """
        self.gfu_read_in.vec.FV().NumPy()[:] = vector
        gfu.Set(self.gfu_read_in)


    '''Solves the dirichlet problem by ngsolve routines'''
    
    '''Solves the dirichlet problem by ngsolve routines'''
    def _solve_dirichlet_problem(self, bf, lf, gf, prec, prec_update=False):
        """
        Solves the dirichlet problem by ngsolve routines.
        """
        if prec_update:
            prec.Update()
        ngs.solvers.BVP(bf=bf, lf=lf, gf=gf, pre=prec,needsassembling=False, print=False)

class ProjectToBoundary(NGSolveOperator):

    def __init__(self, domain, codomain=None):
        codomain = codomain or domain
        super().__init__(domain, codomain)
        self.linear=True
        self.bdr = codomain.bdr
        self.gfu_codomain = ngs.GridFunction(self.codomain.fes)
        self.gfu_domain = ngs.GridFunction(self.domain.fes)
        try: 
            self.nr_bc = len(self.codomain.summands)
        except:
            self.nr_bc = 1

    def _eval(self, x):
        if self.nr_bc == 1:
            array = [x]
        else: 
            array = self.domain.split(x)
        toret = []
        for i in range(self.nr_bc):
            self.gfu_domain.vec.FV().NumPy()[:] = array[i]
            self.gfu_codomain.Set(self.gfu_domain, definedon=self.codomain.fes.mesh.Boundaries(self.bdr))
            toret.append(self.gfu_codomain.vec.FV().NumPy().copy())
        return np.array(toret).flatten()

    def _adjoint(self, g):
        toret = []
        if self.nr_bc == 1:
            g_tuple = [g]
        else: 
            g_tuple = self.codomain.split(g)
        for i in range(self.nr_bc):
            self.gfu_codomain.vec.FV().NumPy()[:] = g_tuple[i]
            self.gfu_domain.Set(self.gfu_codomain, definedon=self.codomain.fes.mesh.Boundaries(self.bdr))
            toret.append(self.gfu_domain.vec.FV().NumPy().copy())
        return np.array(toret).flatten()

class Coefficient(NGSolveOperator):
    
    """Diffusion and reaction coefficient problem
    
    Identification of a diffusion coefficient:
    
    PDE: -div(a grad u)=rhs       in Omega
         u = 0            on dOmega

    Evaluate: 
        F: a \mapsto u
    Derivative:
        -div (a grad v)=div (h grad u) in Omega
        v = 0                        on dOmega

    Der: F'[u]: h \mapsto v

    Adjoint:
        div (a grad w)=q  in Omega
        w=0              on dOmega

    Adj: F'[s]^*: q \mapsto -grad(u) grad(w)
    
    
    
    
    Identification of a reaction coefficient:
    
    PDE: -Delta u +c u=f       in Omega
         u = 0            on dOmega

    Evaluate: 
        F: c \mapsto u
        
    Derivative:
        -Delta v + c v=-h u in Omega
        v = 0                        on dOmega

    Der: F'[u]: h \mapsto v

    Adjoint:
        -Delta w+c w=q  in Omega
        w=0              on dOmega

    Adj: F'[s]^*: q \mapsto -u^* w    
    """

    def __init__(
        self, domain, rhs, bc=None, codomain=None,
        diffusion=False, reaction=True
    ):
        assert diffusion or reaction
        assert (diffusion and reaction) is False
        codomain = codomain or domain
        #Need to know the boundary to calculate Dirichlet bdr condition
        assert codomain.bdr is not None

        self.rhs = rhs
        super().__init__(domain, codomain)

        self.diffusion = diffusion
        self.reaction = reaction
        self.dim = domain.fes.mesh.dim

        bc = bc or 0

        # Define mesh and finite element space
        self.fes_domain = domain.fes
        self.fes_codomain = codomain.fes

        # grid functions for later use
        self.gfu_eval = ngs.GridFunction(self.fes_codomain)  # solution, return value of _eval
        self.gfu_deriv = ngs.GridFunction(self.fes_codomain)  # return value of derivative
        self.gfu_adjoint = ngs.GridFunction(self.fes_domain)  # grid function for returning values in adjoint

        self.gfu_bf = ngs.GridFunction(self.fes_codomain) # grid function for defining integrator (bilinearform)
        self.gfu_lf = ngs.GridFunction(self.fes_codomain)  # grid function for defining right hand side (Linearform)

        self.gfu_inner_adj = ngs.GridFunction(self.fes_codomain) #computations in adjoint
        self.gfu_inner_deriv = ngs.GridFunction(self.fes_codomain) #inner computations in derivative

        #Test and Trial Function
        u, v = self.fes_codomain.TnT()

        # Define Bilinearform, will be assembled later
        self.a = ngs.BilinearForm(self.fes_codomain, symmetric=True)
        if self.diffusion:
            self.a += ngs.grad(u) * ngs.grad(v) * self.gfu_bf * ngs.dx
        elif self.reaction:
            self.a += (ngs.grad(u) * ngs.grad(v) + u * v * self.gfu_bf) * ngs.dx

        # Define Linearform, will be assembled later
        self.f = ngs.LinearForm(self.fes_codomain)
        self.f += self.gfu_lf * v * ngs.dx

        if diffusion:
            self.f_deriv = ngs.LinearForm(self.fes_codomain)
            self.f_deriv += -self.gfu_lf * ngs.grad(self.gfu_eval) * ngs.grad(v) * ngs.dx

        # Precompute Boundary values and boundary valued corrected rhs
        #if self.dim == 1:
        #    self.gfu_eval.Set([bc_left, bc_right], definedon=self.fes_codomain.mesh.Boundaries("left|right"))
        #elif self.dim == 2:
        #    self.gfu_eval.Set([bc_left, bc_top, bc_right, bc_bottom], definedon=self.fes_codomain.mesh.Boundaries("left|top|right|bottom"))
        self.gfu_eval.Set(bc, definedon=self.fes_codomain.mesh.Boundaries(codomain.bdr))

        #Initialize Preconditioner for solving the Dirichlet problems
        self.prec = ngs.Preconditioner(self.a, 'local')

        #Initialize homogenous Dirichlet problems for derivative and adjoint
        self.gfu_deriv.Set(0)
        self.gfu_inner_adj.Set(0)

    def _eval(self, diff, differentiate=False, adjoint_derivative=False):
        # Assemble Bilinearform
        self._read_in(diff, self.gfu_bf)
        self.a.Assemble()

        # Assemble Linearform
        self.gfu_lf.Set(self.rhs)
        self.f.Assemble()

        # Solve system
        self._solve_dirichlet_problem(self.a, self.f, self.gfu_eval, self.prec, prec_update=True)

        return self.gfu_eval.vec.FV().NumPy().copy()

    def _derivative(self, argument):
        # Bilinearform already defined from _eval

        # Translate arguments in Coefficient Function and interpolate to codomain
        self._read_in(argument, self.gfu_inner_deriv)

        # Define rhs
        if self.diffusion:
            self.gfu_lf.Set(self.gfu_inner_deriv)
            self.f_deriv.Assemble()

            self._solve_dirichlet_problem(self.a, self.f_deriv, self.gfu_deriv, self.prec)

        elif self.reaction:
            self.gfu_lf.Set(self.gfu_inner_deriv * self.gfu_eval)
            self.f.Assemble()

            self._solve_dirichlet_problem(self.a, self.f, self.gfu_deriv, self.prec)

        return self.gfu_deriv.vec.FV().NumPy().copy()

    def _adjoint(self, argument):
        # Bilinearform already defined from _eval

        # Definition of Linearform
        self.gfu_lf.vec.FV().NumPy()[:] = argument
        self.f.Assemble()

        # Solve system
        self._solve_dirichlet_problem(self.a, self.f, self.gfu_inner_adj, self.prec)

        if self.diffusion:
            self.gfu_adjoint.Set( -ngs.grad(self.gfu_eval) * ngs.grad(self.gfu_inner_adj) )
        elif self.reaction:
            self.gfu_adjoint.Set( -self.gfu_eval * self.gfu_inner_adj )

        return self.gfu_adjoint.vec.FV().NumPy().copy()


class EIT(NGSolveOperator):
    r"""Electrical Impedance Tomography Problem
    
    PDE:
    \[
    -\textrm{div}(s \nabla u)+\alpha u=0 \;\text{ in } \Omega
    \]
    \[
         s \frac{\textrm{d}u}{\textrm{d}n} = g \;\text{ on } \partial\Omega
    \]
    Evaluate: \(F\colon s \mapsto \mathrm{tr}(u)\)
    Derivative:
    \[
    -\textrm{div}(s \nabla v)+\alpha v=\textrm{div}(h \nabla u) (=:f) 
    \]
    \[
         s \frac{\textrm{d}v}{\textrm{d}n} = 0 +(-h\frac{\textrm{d}u}{\textrm{d}n} \;\text{ [second term often omitted] } 
    \]

    Der: \(F'[s]\colon h \mapsto \textrm{tr}(v)\)

    Adjoint:
    \[
    -\textrm{div}(s \nabla w)+\alpha w=0 
    \]
    \[
         s \frac{\textrm{d}w}{\textrm{d}n} = q 
    \]
    
    Adj: \(F'[s]^*\colon q \mapsto -\nabla(u) \nabla(w)\)

    Proof:
    \[(F'h, q)=\int_{\partial\Omega} [\textrm{tr}(v) q] \]
    \[= \int_{\partial\Omega} [\textrm{tr}(v) s \frac{\textrm{d}w}{\textrm{d}n}] \] 
    \[= \int_{\Omega} [\textrm{div}(v s \nabla w )]\]
    Note \(\textrm{div}(s \nabla w) = \alpha*w\), thus above equation shows:
    \[(F'h, q) = (s \nabla v, \nabla w)+\alpha (v, w) \]
    \[= \int_\Omega [\textrm{div}( s \nabla v w)] +(-\textrm{div} (s \nabla v)), w)+\alpha (v, w)\]
    \[= \int_{\partial\Omega} [s dv/dn \textrm{tr}(w)]+(f, w)\]
    \[= (f, w)-\int_{\partial\Omega} [\textrm{tr}(w) h \frac{\textrm{d}u}{\textrm{d}n}]\]
    \[= (h, -\nabla u \nabla w) + \int_\Omega [\textrm{div}(h \nabla u w)]-\int_{\partial\Omega} [\textrm{tr}(w) h \frac{\textrm{d}u}{\textrm{d}n}] \]
    The last two terms are the same! It follows: \((F'h, q) = (h, -\nabla u \nabla w)\). Hence:
    Adjoint: \(q \mapsto -\nabla u \nabla w\)
    """

    def __init__(self, domain, g, codomain=None, alpha=0.01):
        codomain = codomain or domain
        #Need to know the boundary to calculate Neumann bdr condition
        assert codomain.bdr is not None
        super().__init__(domain, codomain)
        self.g = g
        self.nr_bc = len(self.g)

        self.fes_domain = domain.fes
        self.fes_codomain = codomain.fes

        #FES and Grid Function for reading in values
        self.fes_in = ngs.H1(self.fes_codomain.mesh, order=1)
        self.gfu_in = ngs.GridFunction(self.fes_in)

        # grid functions for later use
        self.gfu_eval = ngs.GridFunction(self.fes_codomain)  # solution, return value of _eval
        self.gfu_deriv = ngs.GridFunction(self.fes_codomain)  # grid function return value of derivative
        self.gfu_adjoint = ngs.GridFunction(self.fes_domain) #grid function return value of adjoint
        
        self.gfu_bf = ngs.GridFunction(self.fes_codomain) # grid function for defining integrator (bilinearform)
        self.gfu_lf = ngs.GridFunction(self.fes_codomain)  # grid function for defining right hand side (linearform), f
        self.gfu_b = ngs.GridFunction(self.fes_codomain)

        self.gfu_inner_adjoint = ngs.GridFunction(self.fes_codomain)  # grid function for inner computations in adjoint

        self.Number = ngs.NumberSpace(self.fes_codomain.mesh)
        #r, s = self.Number.TnT()

        u, v = self.fes_codomain.TnT()

        # Define Bilinearform, will be assembled later
        self.a = ngs.BilinearForm(self.fes_codomain, symmetric=True)
        self.a += (ngs.grad(u) * ngs.grad(v) * self.gfu_bf+alpha*u*v) * ngs.dx

        #Additional condition: The integral along the boundary vanishes
        #self.a += ngs.SymbolicBFI(u * s + v * r, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        #self.fes1 = ngs.H1(self.fes_codomain.mesh, order=4, definedon=self.fes_codomain.mesh.Boundaries("cyc"))

        # Define Linearform for evaluation, will be assembled later       
        self.b = ngs.LinearForm(self.fes_codomain)
        self.b += self.gfu_b*v*ngs.ds(codomain.bdr)

        # Define Linearform for derivative, will be assembled later
        self.f_deriv = ngs.LinearForm(self.fes_codomain)
        self.f_deriv += -self.gfu_lf * ngs.grad(self.gfu_eval) * ngs.grad(v) * ngs.dx

        # Initialize preconditioner for solving the Dirichlet problems by ngs.solvers.BVP
        self.prec = ngs.Preconditioner(self.a, 'direct')
    #Weak formulation:
    #0=int_Omega [-div(s grad u) v + alpha u v]=-int_dOmega [s du/dn trace(v)]+int_Omega [s grad u grad v + alpha u v]
    #Hence: int_Omega [s grad u grad v + alpha u v] = int_dOmega [g trace(v)]
    #Left term: Bilinearform self.a
    #Righ term: Linearform self.b
    def _eval(self, diff, differentiate=False, adjoint_derivative=False):
        # Assemble Bilinearform
        self._read_in(diff, self.gfu_bf)
        self.a.Assemble()

        # Assemble Linearform, boundary term
        toret = []
        for i in range(self.nr_bc):
            self.gfu_b.Set(self.g[i])
            self.b.Assemble()

        # Solve system
            if i == 0:
                self._solve_dirichlet_problem(bf=self.a, lf=self.b, gf=self.gfu_eval, prec=self.prec, prec_update=True)
            else: 
                self._solve_dirichlet_problem(bf=self.a, lf=self.b, gf=self.gfu_eval, prec=self.prec)

            toret.append(self.gfu_eval.vec.FV().NumPy()[:].copy())

        return np.array(toret).flatten()

    #Weak Formulation:
    #0 = int_Omega [-div(s grad v) w + alpha v w]-int_Omega [div (h grad u) w]
    #=-int_dOmega [s dv/dn \textrm{tr}(w)] + int_Omega [s grad v grad w + alpha v w]-int_dOmega [h du/dn \textrm{tr}(w)]+int_Omega [h grad u grad w]
    #=int_Omega [s grad v grad w + alpha v w]+int_Omega [h grad u grad w]
    #Hence: int_Omega [s grad v grad w + alpha v w] = int_Omega [-h grad u grad w]
    #Left Term: Bilinearform self.a, already defined in _eval
    #Right Term: Linearform f_deriv
    def _derivative(self, h, **kwargs):
        # Bilinearform already defined from _eval

        # Assemble Linearform
        toret = []
        for i in range(self.nr_bc):
            self._read_in(h[i], self.gfu_lf)
            self.f_deriv.Assemble()

            self.gfu_deriv.Set(0)
            self._solve_dirichlet_problem(bf=self.a, lf=self.f_deriv, gf=self.gfu_deriv, prec=self.prec)

            toret.append(self.gfu_deriv.vec.FV().NumPy()[:].copy())

        return np.array(toret).flatten()

    #Same problem as in _eval
    def _adjoint(self, argument):
        # Bilinearform already defined from _eval

        # Definition of Linearform
        # But it only needs to be defined on boundary
        if self.nr_bc==1:
            argument_tuple = [argument]
        else:
            argument_tuple = self.codomain.split(argument)
        toret = np.zeros(np.size(self.gfu_adjoint.vec.FV().NumPy()))
        for i in range(self.nr_bc):
            self.gfu_b.vec.FV().NumPy()[:] = argument_tuple[i]
            self.b.Assemble()

            self._solve_dirichlet_problem(bf=self.a, lf=self.b, gf=self.gfu_inner_adjoint, prec=self.prec)

            self.gfu_adjoint.Set(-ngs.grad(self.gfu_inner_adjoint) * ngs.grad(self.gfu_eval))

            toret += self.gfu_adjoint.vec.FV().NumPy().copy()

        return toret



class ReactionNeumann(NGSolveOperator):
    r"""
    Estimation of the reaction coefficient from boundary value measurements

    PDE: -div(grad(u)) + s*u = 0 in Omega
         du/dn = g on dOmega

    Evaluate: F: s \mapsto trace(u)
    Derivative:
        -div(grad(v))+s*v = -h*u (=:f)
        dv/dn = 0 

    Der: F'[s]: h \mapsto trace(v)

    Adjoint: 
        -div(grad(w))+s*w = 0
        dw/dn = q
    Adj: F'[s]^*: q \mapsto -u*w

    proof:
    (F'h, q) = int_dOmega [trace(v) q] = int_dOmega [trace(v) dw/dn] = int_Omega [div(v grad w)] 
    = int_Omega [grad v grad w] + int_Omega [v div( grad w)] = int_Omega [div(w grad v)] - int_Omega [div(grad v) w] + int_Omega [v div (grad w)]
    = int_dOmega [trace(w) dv/dn] - int_Omega [h u w] - int_Omega [s v w] + int_Omega [v s w]
    Note that dv/dn=0 on dOmega. Hence:
    (F'h, q) = -int_Omega[h u w] = (h, -u w)
    """
    def __init__(self, domain, g, codomain=None):
        codomain = codomain or domain
        #Need to know the boundary to calculate Neumann bdr condition
        assert codomain.bdr is not None
        super().__init__(domain, codomain)
        self.g = g
        self.nr_bc = len(self.g)

        self.fes_domain = domain.fes
        self.fes_codomain = codomain.fes

        # grid functions for later use
        self.gfu_eval = ngs.GridFunction(self.fes_codomain)  # solution, return value of _eval
        self.gfu_deriv = ngs.GridFunction(self.fes_codomain)  # grid function: return value of derivative
        self.gfu_adjoint = ngs.GridFunction(self.fes_domain)  # grid function: return value of adjoint

        self.gfu_bf = ngs.GridFunction(self.fes_codomain)  # grid function for defining integrator of bilinearform
        self.gfu_lf = ngs.GridFunction(self.fes_domain) # grid function for defining linearform
        self.gfu_b = ngs.GridFunction(self.fes_codomain)  # grid function for defining the boundary term

        self.gfu_inner_adjoint = ngs.GridFunction(self.fes_codomain)  # grid function for inner computation in adjoint

        #Test and Trial Function
        u, v = self.fes_codomain.TnT()

        # Define Bilinearform, will be assembled later
        self.a = ngs.BilinearForm(self.fes_codomain, symmetric=True)
        self.a += (ngs.grad(u) * ngs.grad(v) + u * v * self.gfu_bf) * ngs.dx

        # Boundary term
        self.b = ngs.LinearForm(self.fes_codomain)
        self.b += -self.gfu_b * v.Trace() * ngs.ds(codomain.bdr)

        # Linearform (only appears in derivative)
        self.f_deriv = ngs.LinearForm(self.fes_codomain)
        self.f_deriv += -self.gfu_lf * self.gfu_eval * v * ngs.dx

        # Initialize preconditioner for solving the Dirichlet problems by ngs.solvers.BVP
        self.prec = ngs.Preconditioner(self.a, 'direct')


    def _eval(self, diff, differentiate=False, adjoint_derivative=False):
        # Assemble Bilinearform
        self._read_in(diff, self.gfu_bf)
        self.a.Assemble()

        # Assemble Linearform of boundary term
        toret = []
        for i in range(self.nr_bc):
            self.gfu_b.Set(self.g[i])
            self.b.Assemble()

        # Solve system
            if i == 0:
                self._solve_dirichlet_problem(bf=self.a, lf=self.b, gf=self.gfu_eval, prec=self.prec, prec_update=True)
            else:
                self._solve_dirichlet_problem(bf=self.a, lf=self.b, gf=self.gfu_eval, prec=self.prec)

            toret.append(self.gfu_eval.vec.FV().NumPy()[:].copy())

        return np.array(toret).flatten()

    def _derivative(self, h):
        # Bilinearform already defined from _eval

        # Assemble Linearform of derivative
        toret = []
        for i in range(self.nr_bc):
            self._read_in(h, self.gfu_lf)
            self.f_deriv.Assemble()

            # Solve system
            self._solve_dirichlet_problem(bf=self.a, lf=self.f_deriv, gf=self.gfu_deriv, prec=self.prec)

            toret.append(self.gfu_deriv.vec.FV().NumPy()[:].copy())

        return np.array(toret).flatten()

    def _adjoint(self, argument):
        # Bilinearform already defined from _eval

        # Definition of Linearform
        # But it only needs to be defined on boundary
        if self.nr_bc==1:
            argument_tuple = [argument]
        else:
            argument_tuple = self.codomain.split(argument)
        toret = np.zeros(np.size(self.gfu_adjoint.vec.FV().NumPy()))
        for i in range(self.nr_bc):
            self.gfu_b.vec.FV().NumPy()[:] = argument_tuple[i]
            self.b.Assemble()

        # Solve system
            self._solve_dirichlet_problem(bf=self.a, lf=self.b, gf=self.gfu_inner_adjoint, prec=self.prec)

            self.gfu_adjoint.Set(self.gfu_inner_adjoint * self.gfu_eval)
        
            toret+=self.gfu_adjoint.vec.FV().NumPy().copy()

        return toret




