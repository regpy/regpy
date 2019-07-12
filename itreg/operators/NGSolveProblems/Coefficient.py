#TODO: Use netgen for visualization instead of own functions
#TODO: Further optimization, introduction of solve function, preconditioner, fewer gridfunctions ....
#TODO: Maybe use gridfunctions and not coefficient-vectors as input and output

from itreg.operators import NonlinearOperator

import numpy as np
#import netgen.gui
from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.meshes import Make1DMesh, MakeQuadMesh


class Coefficient(NonlinearOperator):
    

    def __init__(self, domain, rhs, bc_left=None, bc_right=None, bc_top=None, bc_bottom=None, codomain=None, diffusion=True, reaction=False, dim=1):
        assert dim in (1, 2)
        assert diffusion or reaction
        
        codomain = codomain or domain
        self.rhs=rhs

        self.diffusion=diffusion
        self.reaction=reaction
        self.dim=domain.fes.mesh.dim
        
        bc_left=bc_left or 0
        bc_right=bc_right or 0
        bc_top=bc_top or 0
        bc_bottom=bc_bottom or 0

        
        #Define mesh and finite element space
        self.fes_domain=domain.fes
        #self.mesh=self.fes.mesh
        
        self.fes_codomain=codomain.fes
#        if dim==1:
#            self.mesh = Make1DMesh(meshsize)
#            self.fes = H1(self.mesh, order=2, dirichlet="left|right")
#        elif dim==2:
#            self.mesh = MakeQuadMesh(meshsize)
#            self.fes = H1(self.mesh, order=2, dirichlet="left|top|right|bottom")

        #grid functions for later use 
        self.gfu = GridFunction(self.fes_codomain)  # solution, return value of _eval
        self.gfu_bdr=GridFunction(self.fes_codomain) #grid function holding boundary values
        
        self.gfu_integrator = GridFunction(self.fes_domain) #grid function for defining integrator (bilinearform)
        self.gfu_integrator_codomain = GridFunction(self.fes_codomain)
        self.gfu_rhs = GridFunction(self.fes_codomain) #grid function for defining right hand side (Linearform)
        
        self.gfu_inner_domain(self.fes_domain) #grid function for reading in values in derivative
        self.gfu_inner=GridFunction(self.fes_codomain) #grid function for inner computation in derivative and adjoint
        self.gfu_deriv=GridFunction(self.fes_domain) #return value of derivative
        self.gfu_toret=GridFunction(self.fes_domain) #grid function for returning values in adjoint and derivative
       
        u = self.fes_codomain.TrialFunction()  # symbolic object
        v = self.fes_codomain.TestFunction()   # symbolic object 

        #Define Bilinearform, will be assembled later        
        self.a = BilinearForm(self.fes_codomain, symmetric=True)
        if self.diffusion:
            self.a += SymbolicBFI(grad(u)*grad(v)*self.gfu_integrator_codomain)
        elif self.reaction:
            self.a += SymbolicBFI(grad(u)*grad(v)+u*v*self.gfu_integrator_codomain)

        #Define Linearform, will be assembled later        
        self.f=LinearForm(self.fes_codomain)
        self.f += SymbolicLFI(self.gfu_rhs*v)
        
        if diffusion:
            self.f_deriv=LinearForm(self.fes_codomain)
            self.f_deriv += SymbolicLFI(-self.gfu_rhs*grad(v))
        
        #Precompute Boundary values and boundary valued corrected rhs
        if self.dim==1:
            self.gfu_bdr.Set([bc_left, bc_right], definedon=self.fes_codomain.mesh.Boundaries("left|right"))
        elif self.dim==2:
            self.gfu_bdr.Set([bc_left, bc_top, bc_right, bc_bottom], definedon=self.fes_codomain.mesh.Boundaries("left|top|right|bottom"))
        self.r=self.f.vec.CreateVector()
        
        super().__init__(domain, codomain)
        
    def _eval(self, diff, differentiate, **kwargs):
        #Assemble Bilinearform
        self.gfu_integrator.vec.FV().NumPy()[:]=diff 
        self.gfu_integrator_codomain.Set(self.gfu_integrator)
#       self.gfu_integrator.Set(diff)
        self.a.Assemble()
            
        #Assemble Linearform
        self.gfu_rhs.Set(self.rhs)
        self.f.Assemble()
                       
        #Update rhs by boundary values            
        self.r.data = self.f.vec - self.a.mat * self.gfu_bdr.vec
            
        #Solve system
        self.gfu.vec.data=self.gfu_bdr.vec.data+self._Solve(self.a, self.r)

        return self.gfu.vec.FV().NumPy().copy()
#            return self.gfu

    def _derivative(self, argument, **kwargs):
        #Bilinearform already defined from _eval

        #Translate arguments in Coefficient Function            
        self.gfu_inner_domain.vec.FV().NumPy()[:]=argument
        #Interpolate to codomain
        self.gfu_inner.Set(self.gfu_inner_domain)
 
        #Define rhs 
        if self.diffusion:              
            rhs=self.gfu_inner*grad(self.gfu)
            self.gfu_rhs.Set(rhs)
            self.f_deriv.Assemble()
            
            self.gfu_deriv.vec.data=self._Solve(self.a, self.f_deriv.vec)
            
        elif self.reaction:
            rhs=self.gfu_inner*self.gfu                
            self.gfu_rhs.Set(rhs)
            self.f.Assemble()
            
            self.gfu_deriv.vec.data=self._Solve(self.a, self.f.vec)
            
        return self.gfu_toret.vec.FV().NumPy().copy()

            

    def _adjoint(self, argument, **kwargs):  
        #Bilinearform already defined from _eval
            
        #Definition of Linearform
        self.gfu_rhs.vec.FV().NumPy()[:]=argument
#       self.gfu_rhs.Set(rhs)
        self.f.Assemble()

        #Solve system
        self.gfu_inner.vec.data=self._Solve(self.a, self.f.vec)

        if self.diffusion:
            res=-grad(self.gfu)*grad(self.gfu_inner)
        elif self.reaction:
            res=-self.gfu*self.gfu_inner               
            
        self.gfu_toret.Set(res)
            
        return self.gfu_toret.vec.FV().NumPy().copy()
        
    def _Solve(self, bilinear, rhs, boundary=False):
        return bilinear.mat.Inverse(freedofs=self.fes_codomain.FreeDofs()) * rhs


