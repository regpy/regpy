# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:14:15 2019

@author: hendr
"""
from ngsolve import *

class Diffusion_Base_Functions:
    def __init__(self):
        return 
    
    def Solve(self, params, myfunc, rhs):
        gfu = GridFunction(params.fes)  # solution
            
        a = BilinearForm(params.fes, symmetric=True)
        a += SymbolicBFI(grad(params.u)*grad(params.v)*myfunc)
        a.Assemble()
            
        f = LinearForm(params.fes)
        f += SymbolicLFI(rhs*params.v)
        f.Assemble()
        #solve the system
        gfu.vec.data=a.mat.Inverse(freedofs=params.fes.FreeDofs()) * f.vec  
        return gfu

        