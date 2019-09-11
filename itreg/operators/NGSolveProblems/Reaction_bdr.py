from itreg.operators import NonlinearOperator

import numpy as np
from ngsolve import *

class Reaction_Bdr(NonlinearOperator):


    def __init__(self, domain, g, codomain=None):

        codomain = codomain or domain
        self.N_domain=domain.coords.shape[1]
        self.g=g

        self.fes_domain=domain.fes
        self.fes_codomain=codomain.fes

        self.fes_in = H1(self.fes_codomain.mesh, order=1)
        self.gfu_in = GridFunction(self.fes_in)

        #grid functions for later use
        self.gfu = GridFunction(self.fes_codomain)  # solution, return value of _eval
        #self.gfu_bdr=GridFunction(self.fes_codomain) #grid function holding boundary values, g/sigma=du/dn

        self.gfu_bilinearform = GridFunction(self.fes_domain) #grid function for defining integrator (bilinearform)
        self.gfu_bilinearform_codomain = GridFunction(self.fes_codomain) #grid function for defining integrator of bilinearform
        
        self.gfu_linearform_domain = GridFunction(self.fes_codomain) #grid function for defining linearform
        self.gfu_linearform_codomain=GridFunction(self.fes_domain)
        
        self.gfu_deriv_toret=GridFunction(self.fes_codomain) #grid function: return value of derivative
        
        self.gfu_adj=GridFunction(self.fes_domain) #grid function for inner computation in adjoint
        self.gfu_adj_toret=GridFunction(self.fes_domain) #grid function: return value of adjoint

        self.gfu_b = GridFunction(self.fes_codomain) #grid function for defining the boundary term
        
        u = self.fes_codomain.TrialFunction()  # symbolic object
        v = self.fes_codomain.TestFunction()   # symbolic object

        #Define Bilinearform, will be assembled later
        self.a = BilinearForm(self.fes_codomain, symmetric=True)
        self.a += SymbolicBFI(-grad(u)*grad(v)+u*v*self.gfu_bilinearform_codomain)

        #Interaction with Trace
        self.fes_bdr=H1(self.fes_codomain.mesh, order=self.fes_codomain.globalorder, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        self.gfu_getbdr=GridFunction(self.fes_bdr)
        self.gfu_setbdr=GridFunction(self.fes_codomain)

        #Boundary term
        self.b=LinearForm(self.fes_codomain)
        self.b+=SymbolicLFI(-self.gfu_b*v.Trace(), definedon=self.fes_codomain.mesh.Boundaries("cyc"))

        #Linearform (only appears in derivative)
        self.f_deriv=LinearForm(self.fes_codomain)
        self.f_deriv += SymbolicLFI(-self.gfu_linearform_codomain*self.gfu*v)

        super().__init__(domain, codomain)

    def _eval(self, diff, differentiate, **kwargs):
        #Assemble Bilinearform
        self.gfu_bilinearform.vec.FV().NumPy()[:]=diff
        self.gfu_bilinearform_codomain.Set(self.gfu_bilinearform)
        self.a.Assemble()

        #Assemble Linearform of boundary term
        self.gfu_b.Set(self.g)
        self.b.Assemble()

        #Solve system
        self.gfu.vec.data=self._solve(self.a, self.b.vec)

        #if differentiate:
        #    sigma=CoefficientFunction(self.gfu_integrator)
        #    self.gfu_bdr.Set(self.g/sigma)
        
        return self._get_boundary_values(self.gfu)


    def _derivative(self, h, **kwargs):
        #Bilinearform already defined from _eval

        #Translate arguments in Coefficient Function
        self.gfu_linearform_domain.vec.FV().NumPy()[:]=h
        self.gfu_linearform_codomain.Set(self.gfu_linearform_domain)

        #Define rhs
        self.f_deriv.Assemble()

        #Define boundary term, often ignored
        #self.gfu_b.Set(-self.gfu_linearform_codomain*self.gfu_bdr)
        #self.b.Assemble()

        #Solve system
        self.gfu_deriv_toret.vec.data=self._solve(self.a, self.f_deriv.vec)

        return self._get_boundary_values(self.gfu_deriv_toret)



    def _adjoint(self, argument, **kwargs):
        #Bilinearform already defined from _eval

        #Definition of Linearform
        #But it only needs to be defined on boundary
        self._set_boundary_values(argument)

        self.gfu_b.Set(self.gfu_in)
        self.b.Assemble()

        self.gfu_adj.vec.data=self._solve(self.a, self.b.vec)

        self.gfu_adj_toret.Set(self.gfu_adj*self.gfu)

        return self.gfu_adj_toret.vec.FV().NumPy().copy()



    def _solve(self, bilinear, rhs, boundary=False):
        return bilinear.mat.Inverse(freedofs=self.fes_codomain.FreeDofs()) * rhs

    def _get_boundary_values(self, gfu):
        self.gfu_getbdr.Set(0)
        self.gfu_getbdr.Set(gfu, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        return self.gfu_getbdr.vec.FV().NumPy().copy()

    def _set_boundary_values(self, vals):
        self.gfu_setbdr.vec.FV().NumPy()[:]=vals
        self.gfu_in.Set(0)
        self.gfu_in.Set(self.gfu_setbdr, definedon=self.fes_codomain.mesh.Boundaries("cyc"))
        return
