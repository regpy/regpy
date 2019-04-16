# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:08:07 2019

@author: Hendrik Müller
"""

from itreg.operators import NonlinearOperator, OperatorImplementation, Params
from itreg.util import instantiate

import numpy as np
import math as mt
import scipy as scp
import scipy.ndimage
import netgen.gui
from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.meshes import Make1DMesh
import matplotlib.pyplot as plt


class DiffusionCoefficient(NonlinearOperator):
    

    def __init__(self, domain, rhs, bc_left=None, bc_right=None, range=None, spacing=1):
        range = range or domain
        if bc_left is None:
            bc_left=rhs[0]
        if bc_right is None:
            bc_right=rhs[-1]
        mesh = Make1DMesh(domain.coords.shape[0])
        super().__init__(Params(domain, range, rhs=rhs, bc_left=bc_left, bc_right=bc_right, mesh=mesh, spacing=spacing))
        
    @instantiate
    class operator(OperatorImplementation):
        def eval(self, params, diff, data, differentiate, **kwargs):
            r_diff=rdiff(params, diff)
            rhs=FunctoSymbolic(params, r_diff)
            myfunc=FunctoSymbolic(params, diff)
            fes = H1(params.mesh, order=2, dirichlet="bottom|right|top|left")


            u = fes.TrialFunction()  # symbolic object
            v = fes.TestFunction()   # symbolic object
            gfu = GridFunction(fes)  # solution
            
            a = BilinearForm(fes, symmetric=True)
            a += SymbolicBFI(grad(u)*grad(v)*myfunc)
            a.Assemble()
            
            f = LinearForm(fes)
            f += SymbolicLFI(rhs*v)
            f.Assemble()
            #solve the system
            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec

            if differentiate:
                data.u=SymbolictoFunc(params, gfu)+tilde_g_builder(params)
                data.diff=diff
            return SymbolictoFunc(params, gfu)+tilde_g_builder(params)

    @instantiate
    class derivative(OperatorImplementation):
        def eval(self, params, x, data, **kwargs):
            prod=mygradient(params, data.u)
            rhs=FunctoSymbolic(params, mydiv(params, prod))
            myfunc=FunctoSymbolic(params, data.diff)
            fes = H1(params.mesh, order=2, dirichlet="bottom|right|top|left")


            u = fes.TrialFunction()  # symbolic object
            v = fes.TestFunction()   # symbolic object
            gfu = GridFunction(fes)  # solution
            
            a = BilinearForm(fes, symmetric=True)
            a += SymbolicBFI(myfunc*grad(u)*grad(v))
            a.Assemble()
            
            f = LinearForm(fes)
            f += SymbolicLFI(rhs*v)
            f.Assemble()
            #solve the system
            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
            return -SymbolictoFunc(params, gfu)
            

        def adjoint(self, params, y, data, **kwargs):
            rhs=FunctoSymbolic(params, y)
            
            myfunc=FunctoSymbolic(params, data.diff)
            fes = H1(params.mesh, order=2, dirichlet="left|right")


            u = fes.TrialFunction()  # symbolic object
            v = fes.TestFunction()   # symbolic object
            gfu = GridFunction(fes)  # solution
            
            a = BilinearForm(fes, symmetric=True)
            a += SymbolicBFI(grad(u)*grad(v)*myfunc)
            a.Assemble()
            
            f = LinearForm(fes)
            f += SymbolicLFI(rhs*v)
            f.Assemble()
            #solve the system
            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
            
            
            return mygradient(params, data.u)*mygradient(params, SymbolictoFunc(params, gfu))
            
            
            
def tilde_g_builder(params):
    tilde_g=np.interp(params.domain.coords, np.asarray([params.domain.coords[0], params.domain.coords[-1]]), np.asarray([params.bc_left, params.bc_right]))
    v_star=tilde_g
    return v_star  



def FunctoSymbolic(params, func):
        V = H1(params.mesh, order=1, dirichlet="left|right")
        u = GridFunction(V)
        N=params.domain.coords.shape[0]
        for i in range(0, N):
            u.vec[i]=func[i]           
        return CoefficientFunction(u)
    
    

           
def SymbolictoFunc(params, Symfunc):
    N=params.domain.coords.shape[0]
    Symfunc=CoefficientFunction(Symfunc)
    func=np.zeros(N)
    for i in range(0, N):
        mip=params.mesh(params.domain.coords[i])
        func[i]=Symfunc(mip)
    
    return func
        

def rdiff(params, diff):
    res=mydiv(params, diff*mygradient(params, tilde_g_builder(params)))
    return params.rhs+res    


def mygradient(params, func):
    N=params.domain.coords.shape[0]
    return np.gradient(func)*N

def mydiv(params, func): 
    N=params.domain.coords.shape[0]
    return np.gradient(func)*N