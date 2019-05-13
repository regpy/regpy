from itreg.operators import NonlinearOperator, OperatorImplementation, Params
from itreg.util import instantiate
from .Diffusion_Base_Functions import Diffusion_Base_Functions

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
        Base=Diffusion_Base_1D()
        v_star=Base.tilde_g_builder(domain, bc_left, bc_right)
        super().__init__(Params(domain, codomain, rhs=rhs, bc_left=bc_left, bc_right=bc_right, mesh=mesh, fes=fes, u=u, v=v, spacing=spacing, Base=Base, v_star=v_star))
        
    @instantiate
    class operator(OperatorImplementation):
        def eval(self, params, diff, data, differentiate, **kwargs):
            r_diff=params.Base.rdiff(params, diff)
            rhs=params.Base.FunctoSymbolic(params, r_diff)
            myfunc=params.Base.FunctoSymbolic(params, diff)
#            fes = H1(params.mesh, order=2, dirichlet="bottom|right|top|left")

#            u = fes.TrialFunction()  # symbolic object
#            v = fes.TestFunction()   # symbolic object

#            gfu = GridFunction(params.fes)  # solution
            
#            a = BilinearForm(params.fes, symmetric=True)
#            a += SymbolicBFI(grad(params.u)*grad(params.v)*myfunc)
#            a.Assemble()
            
#            f = LinearForm(params.fes)
#            f += SymbolicLFI(rhs*params.v)
#            f.Assemble()
            #solve the system
#            gfu.vec.data = a.mat.Inverse(freedofs=params.fes.FreeDofs()) * f.vec
            gfu=params.Base.Base_Functions.Solve(params, myfunc, rhs)

            if differentiate:
                data.u=params.Base.SymbolictoFunc(params, gfu)+params.v_star
                data.diff=diff
            return params.Base.SymbolictoFunc(params, gfu)+params.v_star

    @instantiate
    class derivative(OperatorImplementation):
        def eval(self, params, x, data, **kwargs):
            prod=params.Base.mygradient(params, data.u)
            rhs=params.Base.FunctoSymbolic(params, params.Base.mydiv(params.Base, params, prod))
            myfunc=params.Base.FunctoSymbolic(params, data.diff)
#            fes = H1(params.mesh, order=2, dirichlet="bottom|right|top|left")
#            fes=params.fes


#            u = fes.TrialFunction()  # symbolic object
#            v = fes.TestFunction()   # symbolic object
#            gfu = GridFunction(params.fes)  # solution
            
#            a = BilinearForm(params.fes, symmetric=True)
#            a += SymbolicBFI(myfunc*grad(params.u)*grad(params.v))
#            a.Assemble()
            
#            f = LinearForm(params.fes)
#            f += SymbolicLFI(rhs*params.v)
#            f.Assemble()
            #solve the system
#            gfu.vec.data = a.mat.Inverse(freedofs=params.fes.FreeDofs()) * f.vec
            gfu=params.Base.Base_Functions.Solve(params, myfunc, rhs)
            return params.Base.SymbolictoFunc(params, gfu)
            

        def adjoint(self, params, y, data, **kwargs):
            rhs=params.Base.FunctoSymbolic(params, y)
            
            myfunc=params.Base.FunctoSymbolic(params, data.diff)
#            fes = H1(params.mesh, order=2, dirichlet="left|right")
#            fes=params.fes


#            u = fes.TrialFunction()  # symbolic object
#            v = fes.TestFunction()   # symbolic object

#            gfu = GridFunction(params.fes)  # solution
            
#            a = BilinearForm(params.fes, symmetric=True)
#            a += SymbolicBFI(grad(params.u)*grad(params.v)*myfunc)
#            a.Assemble()
            
#            f = LinearForm(params.fes)
#            f += SymbolicLFI(rhs*params.v)
#            f.Assemble()
            #solve the system
#            gfu.vec.data = a.mat.Inverse(freedofs=params.fes.FreeDofs()) * f.vec
            gfu=params.Base.Base_Functions.Solve(params, myfunc, rhs)
            
            
            return -params.Base.mygradient(params, data.u)*params.Base.mygradient(params, params.Base.SymbolictoFunc(params, gfu))
            
            
class Diffusion_Base_1D:
    def __init__(self):
        self.Base_Functions=Diffusion_Base_Functions()
        return 
    
#    def Solve(self, params, myfunc, rhs):
#        gfu = GridFunction(params.fes)  # solution
            
#        a = BilinearForm(params.fes, symmetric=True)
#        a += SymbolicBFI(grad(params.u)*grad(params.v)*myfunc)
#        a.Assemble()
            
#        f = LinearForm(params.fes)
#        f += SymbolicLFI(rhs*params.v)
#        f.Assemble()
        #solve the system
#        gfu.vec.data=a.mat.Inverse(freedofs=params.fes.FreeDofs()) * f.vec  
#        return gfu
    
    def tilde_g_builder(self, domain, bc_left, bc_right):
        tilde_g=np.interp(domain.coords, np.asarray([domain.coords[0], domain.coords[-1]]), np.asarray([bc_left, bc_right]))
        return tilde_g  



    def FunctoSymbolic(self, params, func):
        V = H1(params.mesh, order=1, dirichlet="left|right")
        u = GridFunction(V)
        N=params.domain.shape[0]
        for i in range(0, N):
            u.vec[i]=func[i]           
        return CoefficientFunction(u)

    

           
    def SymbolictoFunc(self, params, Symfunc):
        N=params.domain.coords.shape[0]
        Symfunc=CoefficientFunction(Symfunc)
        func=np.zeros(N)
        for i in range(0, N):
            mip=params.mesh(params.domain.coords[i])
            func[i]=Symfunc(mip)
    
        return func
        

    def rdiff(self, params, diff):
        res=self.mydiv(params, diff*self.mygradient(params, params.v_star))
        return params.rhs+res    


    def mygradient(self, params, func):
        N=params.domain.shape[0]
        return np.gradient(func)*N

    def mydiv(self, params, func): 
        N=params.domain.shape[0]
        return np.gradient(func)*N
