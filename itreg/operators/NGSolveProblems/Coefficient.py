#TODO: Use netgen for visualization instead of own functions
#TODO: Further optimization, introduction of solve function, preconditioner, fewer gridfunctions ....
#TODO: Maybe use gridfunctions and not coefficient-vectors as input and output

from itreg.operators import NonlinearOperator

import numpy as np
import math as mt
import scipy as scp
import scipy.ndimage
import netgen.gui
from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.meshes import Make1DMesh, MakeQuadMesh
import matplotlib.pyplot as plt


class Coefficient(NonlinearOperator):
    

    def __init__(self, domain, meshsize, rhs, bc_left=None, bc_right=None, bc_top=None, bc_bottom=None, codomain=None, diffusion=True, reaction=False, dim=1):
        codomain = codomain or domain
        self.rhs=rhs
        self.bc_left=bc_left
        self.bc_right=bc_right
        self.bc_top=bc_top
        self.bc_bottom=bc_bottom
        self.diffusion=diffusion
        self.reaction=reaction
        self.dim=dim
        
        
        #Boundary values
        if bc_left is None:
            bc_left=rhs[0]
        if bc_right is None:
            bc_right=rhs[-1]
        
        #Define mesh and finite element space
        if dim==1:
            self.mesh = Make1DMesh(meshsize)
            self.fes = H1(self.mesh, order=2, dirichlet="left|right")
        elif dim==2:
            self.mesh = MakeQuadMesh(meshsize)
            self.fes = H1(self.mesh, order=2, dirichlet="left|top|right|bottom")

        #grid functions for later use 
        #TODO: These elements should not be part of params, but of data or self
        self.gfu = GridFunction(self.fes)  # solution
        self.gfu_bdr=GridFunction(self.fes)
        self.gfu_adj=GridFunction(self.fes) #solution for computation of adjoint
        self.gfu_adj_sol=GridFunction(self.fes) #return value for adjoint
        self.gfu_integrator = GridFunction(self.fes) #grid function for defining integrator
        self.gfu_rhs = GridFunction(self.fes) #grid function for defining right hand side
        u = self.fes.TrialFunction()  # symbolic object
        v = self.fes.TestFunction()   # symbolic object 

        #Define Bilinearform, will be assembled later        
        self.a = BilinearForm(self.fes, symmetric=True)
        if diffusion:
            self.a += SymbolicBFI(grad(u)*grad(v)*self.gfu_integrator)
        elif reaction:
            self.a += SymbolicBFI(grad(u)*grad(v)+u*v*self.gfu_integrator)

        #Define Linearform, will be assembled later        
        self.f=LinearForm(self.fes)
        self.f += SymbolicLFI(self.gfu_rhs*v)
        
        #Compute Boundary values
        if self.dim==1:
            self.gfu_bdr.Set([self.bc_left, self.bc_right], definedon=self.mesh.Boundaries("left|right"))
        elif self.dim==2:
            self.gfu_bdr.Set([self.bc_left, self.bc_top, self.bc_right, self.bc_bottom], definedon=self.mesh.Boundaries("left|top|right|bottom"))
        self.r=self.f.vec.CreateVector()
        
        
        super().__init__(domain, codomain)
        
    def _eval(self, diff, differentiate, **kwargs):
        #Assemble Bilinearform, L_a           
        for i in range(self.fes.ndof):
            self.gfu_integrator.vec[i]=diff[i]
#       params.Base_fes.gfu_integrator.Set(diff)
        self.a.Assemble()
            
        #Assemble Linearform
#       for j in range(201):          
#           params.Base_fes.gfu_rhs.vec[j]=params.rhs[j]
        self.gfu_rhs.Set(self.rhs)
        self.f.Assemble()
        
        #Set boundary values         
#       gfu=GridFunction(params.fes)
#        if self.dim==1:
#            self.gfu.Set([self.bc_left, self.bc_right], definedon=self.mesh.Boundaries("left|right"))
#        elif self.dim==2:
#            self.gfu.Set([self.bc_left, self.bc_top, self.bc_right, self.bc_bottom], definedon=self.mesh.Boundaries("left|top|right|bottom"))
                       
        #Update rhs by boundary values            
#        r = self.f.vec.CreateVector()
        self.r.data = self.f.vec - self.a.mat * self.gfu_bdr.vec
            
        #Solve system
#        gfu.vec.data += params.a.mat.Inverse(freedofs=params.fes.FreeDofs()) * r
        self.gfu.vec.data=self.gfu_bdr.vec.data+self._Solve(self.a, self.r)

        #data.u has not to be computed as values are stored in params.gfu, data.diff nicht nötig
        #da nur für a gebraucht, welches bekannt ist
#        if differentiate:
#           data.u=gfu
#            self._diff=diff
        return self.gfu.vec.FV().NumPy().copy()
#            return gfu

    def _derivative(self, argument, **kwargs):
        #Define Bilinearform 
        #not needed, bilinearform already defined from operator evaluation
#        for i in range(201):
#            params.gfu_integrator.vec[i]=data.diff[i]
#        params.a.Assemble()

        #Translate arguments in Coefficient Function            
        gfu_h=GridFunction(self.fes)
        for i in range(self.fes.ndof):
            gfu_h.vec[i]=argument[i]
        h=CoefficientFunction(gfu_h)
 
        #Define rhs 
        if self.diffusion:              
            rhs=div(h*grad(self.gfu))
        elif self.reaction:
            rhs=h*self.gfu                
        self.gfu_rhs.Set(rhs)
        self.f.Assemble()
            
        gfu=GridFunction(self.fes)
#       gfu.vec.data= params.a.mat.Inverse(freedofs=params.fes.FreeDofs()) * f.vec
        gfu.vec.data=self._Solve(self.a, self.f.vec)
            
        return gfu.vec.FV().NumPy().copy()

            

    def _adjoint(self, argument, **kwargs):  
        #Definition of Bilinearform
        #Not needed as Bilinearform is already defined from operator evaluation            
#        for i in range(201):
#             params.gfu_integrator.vec[i]=data.diff[i]
##       params.gfu_integrator.Set(data.diff)
#        params.a.Assemble()
            
        #Definition of Linearform
        for j in range(self.fes.ndof):          
            self.gfu_rhs.vec[j]=argument[j]
#       params.gfu_rhs.Set(rhs)
        self.f.Assemble()

        #Solve system
#        params.gfu_adj.vec.data= params.a.mat.Inverse(freedofs=params.fes.FreeDofs()) * params.f.vec
        self.gfu_adj.vec.data=self._Solve(self.a, self.f.vec)

        if self.diffusion:
            res=-grad(self.gfu)*grad(self.gfu_adj)
        elif self.reaction:
            res=-self.gfu*self.gfu_adj               
            
        self.gfu_adj_sol.Set(res)
            
        return self.gfu_adj_sol.vec.FV().NumPy().copy()
#       return gfu2
        
    def _Solve(self, bilinear, rhs):
        return bilinear.mat.Inverse(freedofs=self.fes.FreeDofs()) * rhs

    