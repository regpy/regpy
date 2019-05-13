# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:29:40 2019

@author: Hendrik Müller
"""

from itreg.operators import NonlinearOperator, OperatorImplementation, Params
from itreg.util import instantiate
from .Reaction_Base_Functions import Reaction_Base_Functions

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
    

    def __init__(self, domain, rhs, bc_left=None, bc_right=None, codomain=None, spacing=1):
        codomain = codomain or domain
        if bc_left is None:
            bc_left=rhs[0]
        if bc_right is None:
            bc_right=rhs[-1]
        mesh = Make1DMesh(domain.coords.shape[0])
        fes = H1(mesh, order=2, dirichlet="bottom|right|top|left")
        u = fes.TrialFunction()  # symbolic object
        v = fes.TestFunction()   # symbolic object        
        Base=Reaction_Base_1D()
        v_star=Base.tilde_g_builder(domain, bc_left, bc_right)
        super().__init__(Params(domain, codomain, rhs=rhs, bc_left=bc_left, bc_right=bc_right, mesh=mesh, fes=fes, u=u, v=v, spacing=spacing, Base=Base, v_star=v_star))
        
    @instantiate
    class operator(OperatorImplementation):
        def eval(self, params, c, data, differentiate, **kwargs):
            r_c=params.Base.rc(params, c)
            rhs=params.Base.FunctoSymbolic(params, r_c)
            myfunc=params.Base.FunctoSymbolic(params, c)
#            fes = H1(params.mesh, order=2, dirichlet="bottom|right|top|left")


#            u = fes.TrialFunction()  # symbolic object
#            v = fes.TestFunction()   # symbolic object
#            gfu = GridFunction(fes)  # solution
            
#            a = BilinearForm(fes, symmetric=True)
#            a += SymbolicBFI(grad(params.u)*grad(params.v)+myfunc*params.u*params.v)
#            a.Assemble()
            
#            f = LinearForm(fes)
#            f += SymbolicLFI(rhs*params.v)
#            f.Assemble()
            #solve the system
#            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
            gfu=params.Base.Base_Functions.Solve(params, myfunc, rhs)

            if differentiate:
                data.u=params.Base.SymbolictoFunc(params, gfu)+params.v_star
                data.c=c
            return params.Base.SymbolictoFunc(params, gfu)+params.v_star
    @instantiate
    class derivative(OperatorImplementation):
        def eval(self, params, x, data, **kwargs):
            rhs=params.Base.FunctoSymbolic(params, -data.u*x)
            myfunc=params.Base.FunctoSymbolic(params, data.c)
#            fes = H1(params.mesh, order=2, dirichlet="bottom|right|top|left")


#            u = fes.TrialFunction()  # symbolic object
#            v = fes.TestFunction()   # symbolic object
#            gfu = GridFunction(fes)  # solution
            
#            a = BilinearForm(fes, symmetric=True)
#            a += SymbolicBFI(grad(u)*grad(v)+myfunc*u*v)
#            a.Assemble()
            
#            f = LinearForm(fes)
#            f += SymbolicLFI(rhs*v)
#            f.Assemble()
            #solve the system
#            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
            gfu=params.Base.Base_Functions.Solve(params, myfunc, rhs)
            return SymbolictoFunc(params, gfu)
            

        def adjoint(self, params, y, data, **kwargs):
            rhs=params.Base.FunctoSymbolic(params, y)
            
            myfunc=params.Base.FunctoSymbolic(params, data.c)
#            fes = H1(params.mesh, order=2, dirichlet="bottom|right|top|left")


#            u = fes.TrialFunction()  # symbolic object
#            v = fes.TestFunction()   # symbolic object
#            gfu = GridFunction(fes)  # solution
            
#            a = BilinearForm(fes, symmetric=True)
#            a += SymbolicBFI(grad(u)*grad(v)+myfunc*u*v)
#            a.Assemble()
            
#            f = LinearForm(fes)
#            f += SymbolicLFI(rhs*v)
#            f.Assemble()
            #solve the system
#            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
            gfu=params.Base.Base_Functions.Solve(params, myfunc, rhs)
           
            return -data.u*params.Base.SymbolictoFunc(params, gfu)
            
class Reaction_Base_1D:
    def __init__(self):
        self.Base_Functions=Reaction_Base_Functions()
        return  

#    def Solve(self, params, myfunc, rhs):
#        gfu = GridFunction(params.fes)  # solution
            
#        a = BilinearForm(params.fes, symmetric=True)
#        a += SymbolicBFI(grad(params.u)*grad(params.v)+myfunc*params.u*params.v)
#        a.Assemble()
            
#        f = LinearForm(params.fes)
#        f += SymbolicLFI(rhs*params.v)
#        f.Assemble()
        #solve the system
#        gfu.vec.data = a.mat.Inverse(freedofs=params.fes.FreeDofs()) * f.vec        
#        return gfu
            
    def tilde_g_builder(self, domain, bc_left, bc_right):
        tilde_g=np.interp(domain.coords, np.asarray([domain.coords[0], domain.coords[-1]]), np.asarray([bc_left, bc_right]))
        v_star=tilde_g
        return v_star    


    def FunctoSymbolic(self, params, func):
        V = H1(params.mesh, order=1, dirichlet="left|right")
        u = GridFunction(V)
        N=params.domain.coords.shape[0]
        for i in range(0, N):
            u.vec[i]=func[i]           
        return CoefficientFunction(u)
    
    

           
    def SymbolictoFunc(self, params, Symfunc):
        N=params.domain.size_support
        Symfunc=CoefficientFunction(Symfunc)
        func=np.zeros(N)
        for i in range(0, N):
            mip=params.mesh(params.domain.coords[i])
            func[i]=Symfunc(mip)
    
        return func
           
    def rc(self, params, c):
        res=self.mylaplace(params, params.v_star)
        return params.rhs+res-c*params.v_star    



    def mylaplace(self, params, func):
        N=params.domain.size_support
        der=np.zeros(N)
        for i in range(1, N-1):
            der[i]=(func[i+1]+func[i-1]-2*func[i])/(1/N**2)
        der[0]=der[1]
        der[-1]=der[-2]
        return der  
