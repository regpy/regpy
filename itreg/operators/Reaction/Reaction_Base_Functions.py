# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:00:58 2019

@author: hendr
"""

from ngsolve import *

class Reaction_Base_Functions:
    def __init__(self):
        return 
    
    def Solve(self, params, myfunc, rhs):
        gfu = GridFunction(params.fes)  # solution
            
        a = BilinearForm(params.fes, symmetric=True)
        a += SymbolicBFI(grad(params.u)*grad(params.v)+myfunc*params.u*params.v)
        a.Assemble()
            
        f = LinearForm(params.fes)
        f += SymbolicLFI(rhs*params.v)
        f.Assemble()
        #solve the system
        pre = Preconditioner(a, 'local')
        a.Assemble()
        inv = CGSolver(a.mat, pre.mat, maxsteps=1000)
        gfu.vec.data = inv * f.vec         
        return gfu        