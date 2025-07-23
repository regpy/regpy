r"""PDE forward operators using NGSolve
"""

import ngsolve as ngs
import numpy as np

from regpy.operators import Operator

class NGSolveOperator(Operator):
    r"""The Base class for operators defined on `vecsps.ngsolve.NgsSpace`\s.

    Parameters
    ----------
    domain : NgsSpace 
        The space for the domain.
    codomain : NgsSpace
        The space for the codomain.
    linear : boolean, optional
        True if linear else False. (Defaults: False)
    """
    def __init__(self, domain, codomain, linear = False):
        super().__init__(domain = domain, codomain = codomain, linear = linear)
        self.gfu_read_in = ngs.GridFunction(self.domain.fes)

    '''Reads in a coefficient vector of the domain and interpolates in the codomain.
    The result is saved in gfu'''
    def _read_in(self, vector, gfu,definedonelements=None):
        r"""Read in of a numpy array into a ngsolve grid function. Note You can also read into
        ngsolve LinearForm.

        Parameters
        ----------
        vector : numpy.array
            Numpy array to be put into the `GridFunction`
        gfu : ngsolve.GridFunction or ngsolve.LinearForm
            ngsolve element to be written into.
        definedonelements : pyngcore.pyngcore.BitArray, optional
            BitArray representing the finite elements to be projected onto, by default None
        """
        gfu.vec.FV().NumPy()[:] = vector
        if definedonelements is not None:
            ngs.Projector(definedonelements, range=True).Project(gfu.vec)

    def _solve_dirichlet_problem(self, bf, lf, gf, prec):
        r"""Solves the problem 
        \begin{align*}
        b(u,v) = f(v) \;\forall v\; test\; functions,\\
        u|_\Gamma = g
        \end{align*}
        The boundary values are given by what `gf` has as values on the boundary. 

        Parameters
        ----------
        bf : ngs.BilinearForm
            the bilinear form of the problem.
        lf : ngs.LinearFrom
            the linear form of the problem. 
        gf : ngs.GridFunction
            The grid functions on which to solve the solution will be put into these and they have to satisfy 
            the boundary condition that you want.
        prec : ngs.Preconditioner
            preconditioner to be used with ngsolve.
        """
        gf.vec.data += bf.mat.Inverse(freedofs=self.codomain.fes.FreeDofs()) * (lf.vec - bf.mat * gf.vec)

        
class SecondOrderEllipticCoefficientPDE(NGSolveOperator):
    r"""Provides a general setup for the forward problems mapping PDE coefficients to their solutions.
    That is we assume that as variational formulation one can use 

    .. math::
        \forall v:    b_0(u,v) + b_a(u,v) = F(v)

    with :math:`b_0` a bilinear form independent of the coefficient :math:`a` and :math:`F` some linear form.  
    Furthermore :math:`b_a` the bilinear form that depends on :math:`a` has to be linear in :math:`a` so that 
    one can define a bilinear form :math:`c_u(a,v)=b_a(u,v)`. More over we may assume some Dirichlet 
    boundary conditions on :math:`u`.That is 

    .. math::
        F: a \mapsto u 


    The Frechét derivative :math:`F'[a]h` in direction :math:`h` is then given as the variational 
    solution :math:`u'` to 

    .. math::
        \forall v: b_0(u',v) + b_a(u',v) = -c_u(h,v)

    with :math:`u=F(a)`. That is

    .. math::
        F'[a]: h \mapsto u' 


    It's adjoint :math:`F'[a]^\ast g` is given as the linear form :math:`-c_u(\cdot,w)` for :math:`u=F(a)` and 
    :math:`w` solving the problem 

    .. math::
        \forall v:  b_0(v,w) + b_a(v,w) = <g,v>.

    That is

    .. math::
        F'[a]^\ast: g \mapsto -c_u(\cdot,w)


    Notes
    -----
    A subclass implemented by a user has to at least implement the subroutine `_bf` which is the 
    implementation of the bilinear form depending on the coefficient :math:`a`. Optional can the 
    independent bilinear form :math:`b_0` be implemented as formal integrator in `_bf_0` and the 
    linear form can be implemented in `_lf`. 

    Parameters
    ----------
    domain : NgsSpace
        The NgsSpace on which the coefficients defined are.
    sol_domain : NgsSpace
        The NgsSpace on which the PDE solutions defined are.
    bdr_val : array type, optional
        Boundary value of the PDE solution of the forward evaluation, by default None
    a_bdr_val : array type, optional
        Boundary value of the coefficients, by default None
    """
    def __init__(self, domain, sol_domain, bdr_val = None, a_bdr_val = None):
        super().__init__(domain, sol_domain, linear = False)
        self.gfu_a=ngs.GridFunction(self.domain.fes)
        self.gfu_h=ngs.GridFunction(self.domain.fes)
        self.gfu_a_bdr = ngs.GridFunction(self.domain.fes)
        if a_bdr_val is not None:
            assert self.domain.bdr is not None 
            assert self.domain.is_on_boundary(a_bdr_val)
            self._read_in(a_bdr_val,self.gfu_a_bdr)
        self.gfu_deriv=ngs.GridFunction(self.codomain.fes)
        self.gfu_adj_help=ngs.GridFunction(self.codomain.fes)
        if bdr_val is not None and bdr_val in self.codomain:
            self.gfu_eval=self.codomain.to_ngs(bdr_val)
        else:
            self.gfu_eval=ngs.GridFunction(self.codomain.fes)


        self.u_a, self.v_a = self.domain.fes.TnT()
        self.u, self.v = self.codomain.fes.TnT()
        

    def _eval(self, a, differentiate=False, adjoint_derivative=False):
        self._read_in(a, self.gfu_a,definedonelements=self.domain.fes.FreeDofs())
        self.gfu_a.vec.data = self.gfu_a.vec + self.gfu_a_bdr.vec
        self.bf_mat = ngs.BilinearForm(self.codomain.fes)
        self.bf_mat += self._bf(self.gfu_a,self.u,self.v) 
        if self._bf_0() is not None:
            self.bf_mat += self._bf_0()
        self.bf_mat.Assemble()
        self.bf_mat_inv = self.bf_mat.mat.Inverse(freedofs=self.codomain.fes.FreeDofs())
        self.gfu_eval.vec.data += self.bf_mat_inv * (self._lf().vec - self.bf_mat.mat * self.gfu_eval.vec)
        return self.gfu_eval.vec.FV().NumPy().copy()
    
    def _derivative(self, h):
        lf = self._c_u(h)
        self.gfu_deriv.vec.data += self.bf_mat_inv * (-lf.vec - self.bf_mat.mat * self.gfu_deriv.vec)
        return self.gfu_deriv.vec.FV().NumPy().copy()

    def _adjoint(self, g):
        lf = ngs.LinearForm(self.codomain.fes).Assemble()
        self._read_in(g,lf)
        self.gfu_adj_help.vec.data += self.bf_mat.mat.CreateTranspose().Inverse(freedofs=self.codomain.fes.FreeDofs()) * (lf.vec - self.bf_mat.mat.CreateTranspose() * self.gfu_adj_help.vec)
        lf_adj = ngs.LinearForm(self.domain.fes)
        lf_adj += -1*self._bf(self.v_a,self.gfu_eval,self.gfu_adj_help)
        lf_adj.Assemble()
        ngs.Projector(self.domain.fes.FreeDofs(), range=True).Project(lf_adj.vec)
        return lf_adj.vec.FV().NumPy().copy()

    def _bf(self,a,u,v):
        r"""Implementation of :math:`b_a` as `ngsolve.comp.SumOfIntegrals` that is something similar to
        `a*ngs.grad(u)*ngs.grad(v)*ngs.dx` where `u` ist used as trial functions and `v` as test 
        functions. This method has to be implemented by 

        Parameters
        ----------
        a : ngsolve.CoefficientFunction or ngsolve.GridFunction
            the coefficient in the PDE
        u : ngsolve.comp.ProxyFunction
            Trial functions for PDE
        v : ngsolve.comp.ProxyFunction
            Test functions for PDE

        Returns
        ------
        ngsolve.comp.SumOfIntegrals
            the formal Integration formula of the bilinear form depending on the coefficient
        """
        raise NotImplementedError
    
    def _bf_0(self):
        r"""Implementation of :math:`b_0` as `ngsolve.comp.SumOfIntegrals` is an optional method to be 
        overwritten with subclasses.  

        Returns
        -------
        ngsolve.comp.SumOfIntegrals
            the formal Integration formula of the bilinear form independent of the coefficient
        """
        return None
    
    def _lf(self):
        r"""The Linear form of the PDE :math:`F` implemented as a fixed Linear form. Note that the 
        Linear form has to be defined on the `codomain` as this is the domain of the solution of 
        the PDE. By default this is the empty Linear form. 

        Returns
        ------
        ngsolve.Linearform
            Linear form of the PDE :math:`F` as `ngsolve.LinearForm`
        """
        return ngs.LinearForm(self.codomain.fes).Assemble()
        
    def _c_u(self,h):
        self._read_in(h, self.gfu_h,definedonelements=self.domain.fes.FreeDofs())
        lf = ngs.LinearForm(self.codomain.fes)
        lf += self._bf(self.gfu_h,self.gfu_eval,self.v)
        return lf.Assemble()

    

class SolveSystem(NGSolveOperator):
    r"""Solve the system 
    \begin(align*)
    Lu = f \text{ in } \Omega, \\
    u = 0  \text{ in } \partial\Omega
    \begin{align*}
    given f. 

    Parameters
    ----------
    domain : NgsSpace
        the underlying NgsSpace.
    bf : ngs.BilinearForm
        The bilinear form describing :math:`L`
    """
    def __init__(self, domain, bf):
        super().__init__(domain=domain, codomain=domain, linear=True)
        self.bf=bf
        self.prec = ngs.Preconditioner(self.bf, 'local')
        self.gfu=ngs.GridFunction(self.domain.fes)
        self.gfu_adj=ngs.GridFunction(self.domain.fes)
        self.gfu_eval=ngs.GridFunction(self.domain.fes)
        u, v=self.domain.fes.TnT()
        
        self.f = ngs.LinearForm(self.domain.fes)
        self.f += self.gfu * v * ngs.dx
        
        self.f_adj = ngs.LinearForm(self.domain.fes)
        self.f_adj += self.gfu_adj * v * ngs.dx
        
    def _eval(self, argument):
        self._read_in(argument, self.gfu)
        self.f.Assemble()
        self._solve_dirichlet_problem(self.bf, self.f, self.gfu_eval, self.prec)
        return self.gfu_eval.vec.FV().NumPy().copy()
    
    def _adjoint(self, argument, return_numpy=True):
        f_read = ngs.LinearForm(self.domain.fes)
        #f_read.Assemble()
        f_read.vec.FV().NumPy()[:]=argument
        
        self.gfu_adj.vec.data=self.bf.mat.CreateTranspose().Inverse()*f_read.vec
        self.f_adj.Assemble()
        if return_numpy:
            return self.f_adj.vec.FV().NumPy()[:]
        else:
            return self.f_adj
        
        
class LinearForm(NGSolveOperator):
    def __init__(self, domain):
        super().__init__(domain=domain, codomain=domain, linear=True)
        self.gfu=ngs.GridFunction(self.domain.fes)
        self.gfu_adj=ngs.GridFunction(self.domain.fes)
        u, v=self.domain.fes.TnT()
        
        self.f = ngs.LinearForm(self.domain.fes)
        self.f += self.gfu * v * ngs.dx
        
        self.f_adj = ngs.LinearForm(self.domain.fes)
        self.f_adj += self.gfu_adj * v * ngs.dx
        
    def _eval(self, argument):
        self._read_in(argument, self.gfu)
        self.f.Assemble()
        return self.f.vec.FV().NumPy()[:]
    
    def _adjoint(self, argument):
        return self._eval(argument)
    
class LinearFormGrad(NGSolveOperator):
    
    def __init__(self, domain, gfu_eval):
        super().__init__(domain=domain, codomain=domain, linear=True)
        self.gfu=ngs.GridFunction(self.domain.fes)
        self.gfu_adj=ngs.GridFunction(self.domain.fes)
        self.gfu_eval=gfu_eval
        u, v=self.domain.fes.TnT()
        
        self.f = ngs.LinearForm(self.domain.fes)
        self.f += ngs.grad(self.gfu) * ngs.grad(self.gfu_eval) * v * ngs.dx
        
        self.f_adj = ngs.LinearForm(self.domain.fes)
        self.f_adj += self.gfu_adj * ngs.grad(self.gfu_eval) * ngs.grad(v) * ngs.dx
        
    def _eval(self, argument):
        self._read_in(argument, self.gfu)
        self.f.Assemble()
        return self.f.vec.FV().NumPy()[:]
    
    def _adjoint(self, argument):
        self._read_in(argument, self.gfu_adj)
        self.f_adj.Assemble()
        return self.f_adj.vec.FV().NumPy()[:]
        
class BilinearForm(NGSolveOperator):
    
        def __init__(self, domain, bf):
            super().__init__(domain=domain, codomain=domain, linear=True)
            self.bf=bf
            
            self.gfu=ngs.GridFunction(self.domain.fes)
            self.gfu_adj=ngs.GridFunction(self.domain.fes)
            
            self.f_eval = ngs.LinearForm(self.domain.fes)
            self.f_adj  = ngs.LinearForm(self.domain.fes)
            
        def _eval(self, argument):
            self.f_eval.vec.FV().NumPy()[:]=argument
            self.gfu.vec.data=self.bf.mat.Inverse()*self.f_eval.vec
            return self.gfu.vec.FV().NumPy()[:]
        
        def _adjoint(self, argument):
            self.f_adj.vec.FV().NumPy()[:]=argument.conj()
            self.gfu_adj.vec.data=self.bf.mat.CreateTranspose().Inverse()*self.f_adj.vec
            return self.gfu_adj.vec.FV().NumPy()[:].conj()
        
def _SolveSystem(domain, bf):
    
    LF=LinearForm(domain)
    BF=BilinearForm(LF.codomain, bf)
    return BF*LF







class Coefficient(NGSolveOperator):
    r"""Diffusion and reaction coefficient problem
    
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

        self.gfu_bf = ngs.GridFunction(self.fes_domain) # grid function for defining integrator (bilinearform)
        self.gfu_lf = ngs.GridFunction(self.fes_codomain)  # grid function for defining right hand side (Linearform)

        self.gfu_inner_adj = ngs.GridFunction(self.fes_codomain) #computations in adjoint
        self.gfu_inner_deriv = ngs.GridFunction(self.fes_domain) #inner computations in derivative

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
            
        if self.reaction:
            self.lf=LinearForm(self.codomain)
            
        self.gfu_eval.Set(bc, definedon=self.fes_codomain.mesh.Boundaries(codomain.bdr))

        #Initialize Preconditioner for solving the Dirichlet problems
        self.prec = ngs.Preconditioner(self.a, 'local')

        #Initialize homogenous Dirichlet problems for derivative and adjoint
        self.gfu_inner_adj.Set(0)
        self.gfu_inner_deriv.Set(0)

    def _eval(self, diff, differentiate=False, adjoint_derivative=False):
        # Assemble Bilinearform
        self._read_in(diff, self.gfu_bf)
        self.a.Assemble()
        if differentiate:
            self.bf=BilinearForm(self.codomain, self.a)
        
        # Assemble Linearform
        self.gfu_lf.Set(self.rhs)
        self.f.Assemble()

        # Solve system
        self._solve_dirichlet_problem(self.a, self.f, self.gfu_eval, self.prec)
        if differentiate and self.diffusion:
            self.lf=LinearFormGrad(self.codomain, self.gfu_eval)


        return self.gfu_eval.vec.FV().NumPy().copy()

    def _derivative(self, argument):
        # Bilinearform already defined from _eval

        # Translate arguments in Coefficient Function and interpolate to codomain
        self._read_in(argument, self.gfu_inner_deriv)
        
        if self.diffusion:
            self.gfu_deriv.Set(self.gfu_inner_deriv)
            return (self.bf*self.lf)(self.gfu_lf.vec.FV().NumPy()[:])

        elif self.reaction:
            self.gfu_deriv.Set(-self.gfu_inner_deriv * self.gfu_eval)
            return (self.bf*self.lf)(self.gfu_deriv.vec.FV().NumPy()[:])

    def _adjoint(self, argument):
        if self.reaction:
            self.gfu_inner_adj.vec.FV().NumPy()[:]=self.lf._adjoint(self.bf._adjoint(argument))
            self.gfu_adjoint.Set( -self.gfu_eval * self.gfu_inner_adj )
            return self.gfu_adjoint.vec.FV().NumPy().copy()
        
        elif self.diffusion:
            self.gfu_inner_adj.vec.FV().NumPy()[:]=self.lf._adjoint(self.bf._adjoint(argument))
            self.gfu_adjoint.Set( self.gfu_inner_adj )
            return self.gfu_adjoint.vec.FV().NumPy().copy()


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


class EIT(NGSolveOperator):
    r"""Electrical Impedance Tomography Problem
    PDE:

    .. math::
        -\textrm{div}(s \nabla u)+\alpha u&=0 \;\text{ in } \Omega \\
        s \frac{\textrm{d}u}{\textrm{d}n} &= g \;\text{ on } \partial\Omega

    Evaluate: :math:`F\colon s \mapsto \mathrm{tr}(u)`

    Derivative:

    .. math::
        -\textrm{div}(s \nabla v)+\alpha v&=\textrm{div}(h \nabla u) (=:f) \\
        s \frac{\textrm{d}v}{\textrm{d}n} &= 0 +(-h\frac{\textrm{d}u}{\textrm{d}n} \;\text{ [second term often omitted] } 


    Der: :math:`F'[s]\colon h \mapsto \textrm{tr}(v)`

    Adjoint

    .. math::
        -\textrm{div}(s \nabla w)+\alpha w&=0 \\
         s \frac{\textrm{d}w}{\textrm{d}n} &= q 

    
    Adj: :math:`F'[s]^*\colon q \mapsto -\nabla(u) \nabla(w)`

    Proof

    .. math::
        (F'h, q)&=\int_{\partial\Omega} [\textrm{tr}(v) q] \\
        &= \int_{\partial\Omega} [\textrm{tr}(v) s \frac{\textrm{d}w}{\textrm{d}n}] \\
        &= \int_{\Omega} [\textrm{div}(v s \nabla w )]

    Note :math:`\textrm{div}(s \nabla w) = \alpha*w`, thus above equation shows

    .. math::
        (F'h, q) &= (s \nabla v, \nabla w)+\alpha (v, w) \\
        &= \int_\Omega [\textrm{div}( s \nabla v w)] +(-\textrm{div} (s \nabla v)), w)+\alpha (v, w) \\
        &= \int_{\partial\Omega} [s dv/dn \textrm{tr}(w)]+(f, w) \\
        &= (f, w)-\int_{\partial\Omega} [\textrm{tr}(w) h \frac{\textrm{d}u}{\textrm{d}n}] \\
        &= (h, -\nabla u \nabla w) + \int_\Omega [\textrm{div}(h \nabla u w)]-\int_{\partial\Omega} [\textrm{tr}(w) h \frac{\textrm{d}u}{\textrm{d}n}] \\

    The last two terms are the same! It follows: :math:`(F'h, q) = (h, -\nabla u \nabla w)`. Hence:
    Adjoint: :math:`q \mapsto -\nabla u \nabla w`
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
                self._solve_dirichlet_problem(bf=self.a, lf=self.b, gf=self.gfu_eval, prec=self.prec)
            else: 
                self._solve_dirichlet_problem(bf=self.a, lf=self.b, gf=self.gfu_eval, prec=self.prec)

            toret.append(self.gfu_eval.vec.FV().NumPy()[:].copy())

        return np.array(toret).flatten()

#Weak Formulation:
#0 = int_Omega [-div(s grad v) w + alpha v w]-int_Omega [div (h grad u) w]
#=-int_dOmega [s dv/dn trace(w)] + int_Omega [s grad v grad w + alpha v w]-int_dOmega [h du/dn trace(w)]+int_Omega [h grad u grad w]
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

    PDE: :math:`-div(grad(u)) + s*u = 0 in Omega`

    .. math:: 
        du/dn = g on dOmega

    Evaluate: :math:`F: s \mapsto trace(u)`
    Derivative:

    .. math::
        -div(grad(v))+s*v = -h*u (=:f) \\
        dv/dn = 0 

    Der: :math:`F'[s]: h \mapsto trace(v)`

    Adjoint: 

    .. math::
        -div(grad(w))+s*w = 0 \\
        dw/dn = q
    
    Adj: :math:`F'[s]^*: q \mapsto -u*w`

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
                self._solve_dirichlet_problem(bf=self.a, lf=self.b, gf=self.gfu_eval, prec=self.prec)
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




