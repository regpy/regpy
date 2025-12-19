r"""PDE forward operators using NGSolve
"""
import types 

import ngsolve as ngs
from pyngcore.pyngcore import BitArray
import numpy as np

from regpy.vecsps.ngsolve import NgsVectorSpace,NgsBaseVector
from regpy.util import Errors

from .base import Operator

__all__ = ["NgsOperator", "NgsMatrixMultiplication", "SecondOrderEllipticCoefficientPDE", "SolveSystem", "LinearForm", "LinearFormGrad", "NgsGradOP", "ProjectToBoundary"]

class NgsOperator(Operator):
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
    def __init__(self, 
            domain : NgsVectorSpace, 
            codomain : NgsVectorSpace, 
            linear : bool = False)->None:
        if not isinstance(domain,NgsVectorSpace):
            raise TypeError(Errors.not_instance(domain,NgsVectorSpace,add_info="The domain of an NgsOperator needs to be an NgsVectorSpace"))
        if not isinstance(codomain,NgsVectorSpace):
            raise TypeError(Errors.not_instance(codomain,NgsVectorSpace,add_info="The domain of an NgsOperator needs to be an NgsVectorSpace"))
        super().__init__(domain = domain, codomain = codomain, linear = linear)
        self.gfu_read_in = ngs.GridFunction(self.domain.fes)

    '''Reads in a coefficient vector of the domain and interpolates in the codomain.
    The result is saved in gfu'''
    def _read_in(self, 
            vector, 
            gfu,
            definedonelements =None):
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

    def _solve_dirichlet_problem(self, 
            bf : ngs.comp.BilinearForm, 
            lf : ngs.comp.LinearForm, 
            gf : ngs.comp.GridFunction, 
            pre = None,
            solver = None,
            **kwargs) -> None:
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
        prec : BaseMatrix or class or Sting, default None
            preconditioner to be used with ngsolve.
        solver : class or None
            A solver instance that is passed to the ngs.solvers.BVP
        kwargs : dict
            Dictionary of possible arguments that can be passed to the ngs.solvers.BVP
        """
        ngs.solvers.BVP(bf, lf, gf, pre = pre, solver = solver, **kwargs)

class NgsMatrixMultiplication(NgsOperator):
    r"""An operator defined by an NGSolve bilinear form. This is a helper to define 
    Gram matrices by a bilinear form.  

    Parameters
    ----------
    domain : NgsVectorSpace
        The vector space.
    form : ngsolve.BilinearForm or ngsolve.BaseMatrix
        The bilinear form or matrix. A bilinear form will be assembled.
    """

    def __init__(self, domain, form):
        super().__init__(domain, domain, linear=True)
        if isinstance(form, ngs.BilinearForm):
            if domain.fes != form.space:
                raise ValueError(Errors.not_equal(domain.fes,form.space,add_info="The FES of the domain has to match with the one of the Bilinear form to construct an NgsMatrixMultiplication."))
            form.Assemble()
            mat = form.mat
        elif isinstance(form, ngs.BaseMatrix):
            mat = form
        else:
            raise TypeError('Invalid type: {}'.format(type(form)))
        self.mat = mat
        """The assembled matrix."""
        self._inverse = None
        self.res = self.domain.zeros()

    def _eval(self, x):
        self.res.vec.data = self.mat * x.vec
        return self.res.copy()

    def _adjoint(self, y):
        self.res.vec.data = self.mat.T * y.vec
        return self.res.copy

    @property
    def inverse(self):
        """The inverse as a `Matrix` instance."""
        if self._inverse is not None:
            return self._inverse
        elif isinstance(self.mat,ngs.la.ProductMatrix):
            raise NotImplementedError(Errors.generic_message(f"The inverse of a product ngsolve matrix is None. You have to figure out the inverse your self."))
        else:
            self._inverse = NgsMatrixMultiplication(
                self.domain,
                self.mat.Inverse(freedofs=self.domain.fes.FreeDofs())
            )
            self._inverse._inverse = self
            return self._inverse
    
    @inverse.setter
    def inverse(self, inv):
        if inv is None:
            self._inverse = None
            self.log.info("Setting the inverse of the operator {} to None".format(self))
        elif not isinstance(inv, Operator):
            raise TypeError(Errors.not_instance(
                inv,
                Operator,
                "The inverse has to be an Operator instance."
                ))
        self.log.info("Setting the inverse of the operator {} to {} overwriting the old {}.".format(self,inv,self._inverse))
        self._inverse = inv
        
class SecondOrderEllipticCoefficientPDE(NgsOperator):
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
    domain : NgsVectorSpace
        The NgsVectorSpace on which the coefficients defined are.
    sol_domain : NgsVectorSpace
        The NgsVectorSpace on which the PDE solutions defined are.
    bdr_val : NgsBaseVector, optional
        Boundary value of the PDE solution of the forward evaluation, by default None
    a_bdr_val : NgsBaseVector, optional
        Boundary value of the coefficients, by default None
    """
    def __init__(self, 
            domain : NgsVectorSpace, 
            sol_domain : NgsVectorSpace, 
            bdr_val : NgsBaseVector | types.NoneType = None, 
            a_bdr_val : NgsBaseVector | types.NoneType = None) -> None:
        super().__init__(domain, sol_domain, linear = False)
        self.gf_a=ngs.GridFunction(self.domain.fes)
        self.gf_h=ngs.GridFunction(self.domain.fes)
        self.a_bdr = a_bdr_val.vec if a_bdr_val is not None and self.domain.bdr is not None and self.domain.is_on_boundary(a_bdr_val) else self.domain.zeros().vec

        self.gf_deriv=ngs.GridFunction(self.codomain.fes)
        self.gf_adj_help=ngs.GridFunction(self.codomain.fes)
        self.gf_eval=ngs.GridFunction(self.codomain.fes)
        if bdr_val is not None and bdr_val in self.codomain:
            self.gf_eval.vec.data = bdr_val.vec
        
        self.u_a, self.v_a = self.domain.fes.TnT()
        self.u, self.v = self.codomain.fes.TnT()

        self.bf_mat = ngs.BilinearForm(self.codomain.fes)
        self.bf_mat += self._bf(self.gf_a,self.u,self.v) 
        if self._bf_0(self.u,self.v) is not None:
            self.bf_mat += self._bf_0(self.u,self.v)
        
        self.first = True
        self.adj_first = True

        self.lf = self._lf()

        self.c_u = ngs.LinearForm(self.codomain.fes)
        self.c_u += self._bf(self.gf_h,self.gf_eval,self.v)

        self.lf_adj = ngs.LinearForm(self.domain.fes)
        self.lf_adj += -1*self._bf(self.v_a,self.gf_eval,self.gf_adj_help)

        self._consts = {*self._consts, "u_a","v_a","u","v", "bf_mat", "bf_mat_inv", "bf_mat_adj", "bf_mat_adj_inv","lf","lf_adj","c_u",}

    def _eval(self, 
            a : NgsBaseVector, 
            differentiate : bool = False) -> NgsBaseVector:
        self.adj_first = True
        self.gf_a.vec.data = a.vec
        ngs.Projector(self.domain.fes.FreeDofs(), range=True).Project(self.gf_a.vec)
        self.gf_a.vec.data += self.a_bdr
        
        self.bf_mat.Assemble()
        self.bf_mat_inv = self.bf_mat.mat.Inverse(freedofs=self.codomain.fes.FreeDofs())
        self.gf_eval.vec.data += self.bf_mat_inv * (self.lf.vec - self.bf_mat.mat * self.gf_eval.vec)
        return self.codomain.from_ngs(self.gf_eval, copy = True)
    
    def _derivative(self, 
            h : NgsBaseVector) -> NgsBaseVector:
        lf = self._c_u(h.vec)
        self.gf_deriv.vec.data += self.bf_mat_inv * (-lf.vec - self.bf_mat.mat * self.gf_deriv.vec)
        return self.codomain.from_ngs(self.gf_deriv, copy = True)

    def _adjoint(self, 
            g : NgsBaseVector) -> NgsBaseVector:
        if self.adj_first:
            self.bf_mat_adj = self.bf_mat.mat.CreateTranspose()
            self.bf_mat_adj_inv = self.bf_mat_adj.Inverse(freedofs=self.codomain.fes.FreeDofs())
            self.adj_first = False
        self.gf_adj_help.vec.data += self.bf_mat_adj_inv * (g.vec - self.bf_mat_adj * self.gf_adj_help.vec)
        self.lf_adj.Assemble()
        return NgsBaseVector(ngs.Projector(self.domain.fes.FreeDofs(), range=True).Project(self.lf_adj.vec) + self.a_bdr,make_copy=True)

    def _bf(self,
            a : ngs.fem.CoefficientFunction,
            u : ngs.fem.CoefficientFunction,
            v : ngs.fem.CoefficientFunction) -> ngs.comp.BilinearForm:
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
    
    def _bf_0(self, u : ngs.fem.CoefficientFunction, v : ngs.fem.CoefficientFunction) -> ngs.comp.BilinearForm | types.NoneType:
        r"""Implementation of :math:`b_0` as `ngsolve.comp.SumOfIntegrals` is an optional method to be 
        overwritten with subclasses.  

        Parameters
        ----------
        u : ngsolve.comp.ProxyFunction
            Trial functions for PDE
        v : ngsolve.comp.ProxyFunction
            Test functions for PDE

        Returns
        -------
        ngsolve.comp.SumOfIntegrals
            the formal Integration formula of the bilinear form independent of the coefficient
        """
        return None
    
    def _lf(self) -> ngs.comp.LinearForm:
        r"""The Linear form of the PDE :math:`F` implemented as a fixed Linear form. Note that the 
        Linear form has to be defined on the `codomain` as this is the domain of the solution of 
        the PDE. By default this is the empty Linear form. 

        Returns
        ------
        ngsolve.Linearform
            Linear form of the PDE :math:`F` as `ngsolve.LinearForm`
        """
        return ngs.LinearForm(self.codomain.fes).Assemble()
        
    def _c_u(self,
            h : ngs.la.BaseVector) -> ngs.comp.BilinearForm:
        self.gf_h.vec.data = 1*h
        self.gf_h.vec.data = ngs.Projector(self.domain.fes.FreeDofs(), range=True).Project(self.gf_h.vec)
        return self.c_u.Assemble()
    

class SolveSystem(NgsOperator):
    r"""Solve the system 
    \begin(align*)
    Lu = f \text{ in } \Omega, \\
    u = 0  \text{ in } \partial\Omega
    \begin{align*}
    given f. 

    Parameters
    ----------
    domain : NgsVectorSpace
        the underlying NgsVectorSpace.
    bf : ngs.BilinearForm
        The bilinear form describing :math:`L`
    """
    def __init__(self, 
            domain : NgsVectorSpace, 
            bf : ngs.BilinearForm,
            use_prec : bool = True,
            **inverse_kwargs) -> None:
        if not isinstance(bf, ngs.BilinearForm):
            raise TypeError(Errors.not_instance(bf,ngs.BilinearForm,"To define a SolveSystem operator the bilinear form needs to be a proper ngs.BilinearFrom!"))
        super().__init__(domain=domain, codomain=domain, linear=True)
        self.bf=bf
        self.use_prec = use_prec
        if use_prec:
            self.prec = ngs.Preconditioner(self.bf, 'local')
        self.inverse_kwargs = inverse_kwargs
        self.gfu=ngs.GridFunction(self.domain.fes)
        self.gfu_adj=ngs.GridFunction(self.domain.fes)
        self.gfu_eval=ngs.GridFunction(self.domain.fes)
        _, v=self.domain.fes.TnT()
        
        self.f = ngs.LinearForm(self.domain.fes)
        self.f += self.gfu * v * ngs.dx
        
        self.f_adj = ngs.LinearForm(self.domain.fes)
        self.f_adj += self.gfu_adj * v * ngs.dx

        self._consts = {*self._consts, "bf", "prec", "f", "f_adj"}
        
    def _eval(self, 
            argument : NgsBaseVector) -> NgsBaseVector:
        self.gfu.vec.data = argument.vec
        self.f.Assemble()
        if self.use_prec:
            self._solve_dirichlet_problem(self.bf, self.f, self.gfu_eval, self.prec, **self.inverse_kwargs)
        else:
            self._solve_dirichlet_problem(self.bf, self.f, self.gfu_eval, **self.inverse_kwargs)
        return NgsBaseVector(self.gfu_eval.vec,make_copy=True)
    
    def _adjoint(self, 
            argument : NgsBaseVector) -> NgsBaseVector:
        self.gfu_adj.vec.data=self.bf.mat.Inverse(freedofs = self.domain.fes.FreeDofs()).T*argument.vec
        self.f_adj.Assemble()
        return NgsBaseVector(self.f_adj.vec,make_copy=True)
        
        
class LinearForm(NgsOperator):
    def __init__(self, 
            domain : NgsVectorSpace) -> None:
        super().__init__(domain=domain, codomain=domain, linear=True)
        self.gfu=ngs.GridFunction(self.domain.fes)
        self.gfu_adj=ngs.GridFunction(self.domain.fes)
        _, v=self.domain.fes.TnT()
        
        self.f = ngs.LinearForm(self.domain.fes)
        self.f += self.gfu * v * ngs.dx

        self._consts = {*self._consts, "f"}
        
    def _eval(self, 
            argument : NgsBaseVector) -> NgsBaseVector:
        self.gfu.vec.data = argument.vec
        self.f.Assemble()
        return NgsBaseVector(self.f.vec,make_copy=True)
    
    def _adjoint(self, 
            argument : NgsBaseVector) -> NgsBaseVector:
        return self._eval(argument)
    
class LinearFormGrad(NgsOperator):
    
    def __init__(self, 
        domain : NgsVectorSpace, 
        gfu_eval : ngs.GridFunction) -> None:

        super().__init__(domain=domain, codomain=domain, linear=True)
        self.gfu=ngs.GridFunction(self.domain.fes)
        self.gfu_adj=ngs.GridFunction(self.domain.fes)
        self.gfu_eval=gfu_eval
        _, v=self.domain.fes.TnT()
        
        self.f = ngs.LinearForm(self.domain.fes)
        self.f += ngs.grad(self.gfu) * ngs.grad(self.gfu_eval) * v * ngs.dx
        self._y = NgsBaseVector(self.f.vec)
        
        self.f_adj = ngs.LinearForm(self.domain.fes)
        self.f_adj += self.gfu_adj * ngs.grad(self.gfu_eval) * ngs.grad(v) * ngs.dx
        self._x = NgsBaseVector(self.f_adj.vec)

        self._consts = {*self._consts, "f", "f_adj"}
        
    def _eval(self,
        argument : NgsBaseVector) -> NgsBaseVector:
        self.gfu.vec.data = argument.vec
        self.f.Assemble()
        return self._y.copy()
    
    def _adjoint(self, 
        argument : NgsBaseVector) -> NgsBaseVector:
        self.gfu_adj.vec.data = argument.vec
        self.f_adj.Assemble()
        return self._x.copy()


class BilinearForm(NgsOperator):
    
    def __init__(self, 
        domain : NgsVectorSpace, 
        bf : ngs.BilinearForm) -> None:
        if not isinstance(bf, ngs.BilinearForm):
            raise TypeError(Errors.not_instance(bf,ngs.BilinearForm,"To define a BilinearForm operator the bilinear form needs to be a proper ngs.BilinearFrom!"))
        super().__init__(domain=domain, codomain=domain, linear=True)
        self.bf=bf
        
        self.gfu=ngs.GridFunction(self.domain.fes)
        self._y = NgsBaseVector(self.gfu.vec)
        self.gfu_adj=ngs.GridFunction(self.domain.fes)
        self._x = NgsBaseVector(self.gfu_adj.vec)
        
        self.f_eval = ngs.LinearForm(self.domain.fes)
        self.f_adj  = ngs.LinearForm(self.domain.fes)
        self._consts = {*self._consts, "bf","f_eval", "f_adj"}
        
    def _eval(self, 
        argument : NgsBaseVector) -> NgsBaseVector:
        self.gfu.vec.data=self.bf.mat.Inverse(freedofs = self.domain.fes.FreeDofs())*argument.vec
        return self._y.copy()
    
    def _adjoint(self, 
        argument : NgsBaseVector) -> NgsBaseVector:
        self.gfu_adj.vec.data=self.bf.mat.CreateTranspose().Inverse(freedofs = self.domain.fes.FreeDofs())*argument.vec
        return self._x.conj().copy()


class NgsGradOP(NgsOperator):
    r"""Gradient operator (bilinear-form based)

    This operator implements a *weak* gradient that is constructed to be exactly
    adjoint (in the sense of :mod:`regpy`) to its divergence operator.  It is **not**
    computed using :func:`ngs.grad` directly.

    Parameter
    ---------
    domain : NgsVectorSpace

    Notes
    -----

    - This operator does **not** compute the pointwise gradient.
    - Deviations near the boundary are expected.
    - Intended for variational formulations (e.g. regularization, weak Laplacians,
      adjoint-based optimization), not for visualization.


    Mathematical definition
    -----------------------

    Let :math:`U_h` be the scalar finite element space and :math:`W_h` the vector
    finite element space. The operator :math:`\mathrm{Grad}_h : U_h \to W_h` is
    defined by

    .. math::

        \langle \mathrm{Grad}_h u, w \rangle_{L^2}
        =
        \int_\Omega \nabla u \cdot w \, dx
        \qquad \forall w \in W_h.

    Thus, :math:`\mathrm{Grad}_h u` is the :math:`L^2`-projection of the pointwise
    gradient :math:`\nabla u` into the space :math:`W_h \subset H(\mathrm{div})`.

    Implementation
    --------------

    The bilinear form

    .. math::

        b(u, w) = \int_\Omega \nabla u \cdot w \, dx

    is assembled as a matrix :math:`B`. The operator is applied as
    :math:`\mathrm{Grad}_h u = B u`, and its adjoint is given by :math:`B^T`,
    which guarantees exact adjointness with respect to the Euclidean inner product
    used by :mod:`regpy`.

    Boundary conditions
    -------------------

    Boundary conditions are encoded in the choice of finite element spaces
    (e.g. Dirichlet conditions on :math:`u` or vanishing normal trace on
    :math:`w`). Under these assumptions, boundary terms vanish automatically and
    no explicit boundary integrals are added.


    """

    def __init__(self,domain):
        if isinstance(domain.fes,(ngs.H1,ngs.L2)):
            vec_fes = ngs.HDiv(domain.fes.mesh, order=domain.fes.globalorder, dirichlet=domain.bdr)
        else:
            raise ValueError("NgsTV is only implemented for H1 or L2 finite element spaces.",self)

        codomain = NgsVectorSpace(vec_fes, bdr=domain.bdr)
        super().__init__(domain, codomain, linear=True)

        u, v = domain.fes.TnT()
        w, z = codomain.fes.TnT()

        self.bf = ngs.BilinearForm(domain.fes, codomain.fes)
        self.bf += ngs.SymbolicBFI(ngs.grad(u) * z)
        self.bf.Assemble()

        self._consts = {*self._consts, "bf"}


    def _eval(self, x):
        return self.codomain.from_ngs(self.bf.mat * x.vec, copy=True)

    def _adjoint(self, y):
        return self.domain.from_ngs(self.bf.mat.T * y.vec, copy=True)

class ProjectToBoundary(NgsOperator):
    """Projects an element to the boundary of codomain.bdr. Given the domain is the codomain 
    this simplifies to taking ngs.Projector for the given vectors.
    Parameters
    ----------
    domain : NgsVectorSpace
        Domain from which to project.
    codomain : NgsVectorSpace, optional
        Codomain onto which to project. Defaults: domain
    """

    def __init__(self, 
            domain: NgsVectorSpace, 
            codomain: NgsVectorSpace | types.NoneType = None,
            bdr : types.NoneType | ngs.comp.Region | str | BitArray = None) -> None:
        codomain = codomain or domain
        self.same_domain = codomain == domain
        super().__init__(domain, codomain)
        self.linear=True
        if bdr is None:
            if codomain.bdr is None:
                raise ValueError(Errors.value_error(f"Either bdr is given or codomain has a specified boundary by regular expression!"))
            self.bdr_codomain = self.codomain.fes.GetDofs(self.codomain.fes.mesh.Boundaries(codomain.bdr))
            self.bdr_domain = self.domain.fes.GetDofs(self.codomain.fes.mesh.Boundaries(codomain.bdr))
        elif isinstance(bdr,ngs.comp.Region):
            self.bdr_codomain = self.codomain.fes.GetDofs(bdr)
            self.bdr_domain = self.domain.fes.GetDofs(bdr)
        elif isinstance(bdr,str):
            self.bdr_codomain = self.codomain.fes.GetDofs(self.codomain.fes.mesh.Boundaries(bdr))
            self.bdr_domain = self.domain.fes.GetDofs(self.domain.fes.mesh.Boundaries(bdr))
        elif isinstance(bdr, BitArray):
            self.bdr_domain = bdr
            self.bdr_codomain = bdr
            if not self.same_domain:
                raise ValueError(Errors.value_error(f"Cannot use BitArray as bdr when domain and codomain are not identical!"))
        else:
            raise TypeError(Errors.type_error(f"The given bdr can be either None, a ngsolve Region, a string regular expression or a BitArray. You gave bdr = {bdr}."))
        self.projector_domain = ngs.Projector(self.bdr_domain, range=True)
        self.projector_codomain = ngs.Projector(self.bdr_codomain, range=True)
        self.gfu_codomain = ngs.GridFunction(self.codomain.fes)
        self.gfu_domain = ngs.GridFunction(self.domain.fes)

        self._consts = {*self._consts, "bdr_domain","bdr_codomain","projector_domain","projector_codomain"}


    def _eval(self, 
        x : NgsBaseVector) -> NgsBaseVector:
        if self.same_domain:
            _x_eval = x.copy()
            self.projector_codomain.Project(_x_eval.vec)
            return _x_eval
        else:
            self.gfu_domain.vec.data = x.vec
            self.gfu_codomain.Set(self.gfu_domain)
            self.projector_codomain.Project(self.gfu_codomain.vec)
            return NgsBaseVector(self.gfu_codomain.vec, make_copy=True)

    def _adjoint(self, 
        x : NgsBaseVector) -> NgsBaseVector:
        if self.same_domain:
            _x_eval = x.copy()
            self.projector_domain.Project(_x_eval.vec)
            return _x_eval
        else:
            self.gfu_codomain.vec.data = x.vec
            self.gfu_domain.Set(self.gfu_codomain)
            self.projector_domain.Project(self.gfu_domain.vec)
            return NgsBaseVector(self.gfu_domain.vec, make_copy=True)
   

