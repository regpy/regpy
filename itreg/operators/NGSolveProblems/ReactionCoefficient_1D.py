from itreg.operators import NonlinearOperator, OperatorImplementation, Params
from itreg.util import instantiate

from .PDEBase import PDEBase

import numpy as np
import math as mt
import scipy as scp
import scipy.ndimage
import netgen.gui
from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.meshes import Make1DMesh
import matplotlib.pyplot as plt


class ReactionCoefficient(NonlinearOperator):
    

    def __init__(self, domain, meshsize, rhs, bc_left=None, bc_right=None, range=None):
        range = range or domain
        
        #Boundary values
        if bc_left is None:
            bc_left=rhs[0]
        if bc_right is None:
            bc_right=rhs[-1]
        
        #Define mesh and finite element space
        mesh = Make1DMesh(meshsize)
        fes = H1(mesh, order=2, dirichlet="left|right")

        #grid functions for later use 
        #TODO: These elements should not be part of params, but of data or self
        gfu = GridFunction(fes)  # solution
        gfu_adj=GridFunction(fes) #solution for computation of adjoint
        gfu_adj_sol=GridFunction(fes) #return value for adjoint
        gfu_integrator = GridFunction(fes) #grid function for defining integrator
        gfu_rhs = GridFunction(fes) #grid function for defining right hand side
        u = fes.TrialFunction()  # symbolic object
        v = fes.TestFunction()   # symbolic object 

        #Define Bilinearform, will be assembled later        
        a = BilinearForm(fes, symmetric=True)
        a += SymbolicBFI(grad(u)*grad(v)+u*v*gfu_integrator)

        #Define Linearform, will be assembled later        
        f=LinearForm(fes)
        f += SymbolicLFI(gfu_rhs*v)
        
        Base=PDEBase()
        
        super().__init__(Params(domain, range, rhs=rhs, bc_left=bc_left, bc_right=bc_right, mesh=mesh, fes=fes, gfu=gfu,
             gfu_adj=gfu_adj, gfu_adj_sol=gfu_adj_sol, gfu_integrator=gfu_integrator, gfu_rhs=gfu_rhs, a=a, f=f, Base=Base))
        
    @instantiate
    class operator(OperatorImplementation):
        def eval(self, params, diff, data, differentiate, **kwargs):
            #Assemble Bilinearform, L_a           
            for i in range(params.fes.ndof):
                params.gfu_integrator.vec[i]=diff[i]
#            params.Base_fes.gfu_integrator.Set(diff)
            params.a.Assemble()
            
            #Assemble Linearform
#            for j in range(201):          
#                params.Base_fes.gfu_rhs.vec[j]=params.rhs[j]
            params.gfu_rhs.Set(params.rhs)
            params.f.Assemble()
        
           #Set boundary values         
#            gfu=GridFunction(params.fes)
            gfu=params.gfu
            gfu.Set([params.bc_left, params.bc_right], definedon=params.mesh.Boundaries("left|right"))
                       
            #Update rhs by boundary values            
            r = params.f.vec.CreateVector()
            r.data = params.f.vec - params.a.mat * gfu.vec
            
            #Solve system
#            gfu.vec.data += params.a.mat.Inverse(freedofs=params.fes.FreeDofs()) * r
            gfu.vec.data+=params.Base.Solve(params, params.a, r)

            #data.u has not to be computed as values are stored in params.gfu
            if differentiate:
#                data.u=gfu
                data.diff=diff
            return gfu.vec.FV().NumPy().copy()
#            return gfu

    @instantiate
    class derivative(OperatorImplementation):
        def eval(self, params, argument, data, **kwargs):
            #Define Bilinearform 
            #not needed, bilinearform already defined from operator evaluation
#            for i in range(201):
#                params.gfu_integrator.vec[i]=data.diff[i]
#            params.a.Assemble()

            #Translate arguments in Coefficient Function            
            gfu_h=GridFunction(params.fes)
            for i in range(params.fes.ndof):
                gfu_h.vec[i]=argument[i]
            h=CoefficientFunction(gfu_h)
 
            #Define rhs               
            rhs=h*params.gfu
            params.gfu_rhs.Set(rhs)
            params.f.Assemble()
            
            gfu=GridFunction(params.fes)
#            gfu.vec.data= params.a.mat.Inverse(freedofs=params.fes.FreeDofs()) * f.vec
            gfu.vec.data=params.Base.Solve(params, params.a, params.f.vec)
           
            return gfu.vec.FV().NumPy().copy()

            

        def adjoint(self, params, argument, data, **kwargs):  
            #Definition of Bilinearform
            #Not needed as Bilinearform is already defined from operator evaluation            
#            for i in range(201):
#                params.gfu_integrator.vec[i]=data.diff[i]
##            params.gfu_integrator.Set(data.diff)
#            params.a.Assemble()
            
            #Definition of Linearform
            for j in range(params.fes.ndof):          
                params.gfu_rhs.vec[j]=argument[j]
#            params.gfu_rhs.Set(rhs)
            params.f.Assemble()

            #Solve system
#            params.gfu_adj.vec.data= params.a.mat.Inverse(freedofs=params.fes.FreeDofs()) * params.f.vec
            params.gfu_adj.vec.data=params.Base.Solve(params, params.a, params.f.vec)

            res=-params.gfu*params.gfu_adj
            
            params.gfu_adj_sol.Set(res)
            
            return params.gfu_adj_sol.vec.FV().NumPy().copy()
#            return gfu2

    
