from itreg.operators import NonlinearOperator, OperatorImplementation, Params
from itreg.util import instantiate

import numpy as np
import math as mt
import scipy as scp
import scipy.ndimage
import netgen.gui
from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.meshes import Make1DMesh, MakeQuadMesh
import matplotlib.pyplot as plt
import netgen.gui
import ngsolve
from netgen.meshing import *


class ReactionCoefficient(NonlinearOperator):
    

    def __init__(self, domain, rhs, bc_top=None, bc_bottom=None, bc_left=None, bc_right=None, range=None, spacing=1):
        range = range or domain
        if bc_top is None:
            bc_top=rhs[0, :]
        if bc_bottom is None:
            bc_bottom=rhs[-1, :]
        if bc_left is None:
            bc_left=rhs[:, 0]
        if bc_right is None:
            bc_right=rhs[:, -1]
        N=rhs.shape[0]-1
        mesh=MakeQuadMesh(N,N)
        fes = H1(mesh, order=2, dirichlet="bottom|right|top|left")
        u = fes.TrialFunction()  # symbolic object
        v = fes.TestFunction()   # symbolic object        
        Base=Reaction_Base_2D()
        v_star=Base.tilde_g_builder(domain, bc_top, bc_bottom, bc_left, bc_right)
        super().__init__(Params(domain, range, rhs=rhs, bc_top=bc_top, bc_bottom=bc_bottom, bc_left=bc_left, bc_right=bc_right, mesh=mesh, fes=fes, u=u, v=v, spacing=spacing, Base=Base, v_star=v_star))
        
    @instantiate
    class operator(OperatorImplementation):
        def eval(self, params, c, data, differentiate, **kwargs):
#            B=B_builder(params, c)
            r_c=params.Base.rc(params, c)
#            rhs=rhs_builder(params, r_c)
            rhs=params.Base.FunctoSymbolic(params, r_c)
            
#            coeff= np.linalg.solve(B, rhs)
            myfunc=params.Base.FunctoSymbolic(params, c)
            
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
#            pre = Preconditioner(a, 'local')
#            a.Assemble()
#            inv = CGSolver(a.mat, pre.mat, maxsteps=1000)
#            gfu.vec.data = inv * f.vec
            #gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
            gfu=params.Base.Base_Functions.Solve(params, myfunc, rhs)

            if differentiate:
                data.u=params.Base.SymbolictoFunc(params, gfu)+params.v_star
                data.c=c
            return params.Base.SymbolictoFunc(params, gfu)+params.v_star

    @instantiate
    class derivative(OperatorImplementation):
        def eval(self, params, x, data, **kwargs):
            rhs=params.Base.FunctoSymbolic(params, -data.u*x)
            #mip=params.mesh(0.5, 0.5)
            #print(rhs(mip))
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
#            pre = Preconditioner(a, 'local')
#            a.Assemble()
#            inv = CGSolver(a.mat, pre.mat, maxsteps=1000)
#            gfu.vec.data = inv * f.vec
            #gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
            gfu=params.Base.Base_Functions.Solve(params, myfunc, rhs)
            return params.Base.SymbolictoFunc(params, gfu)
            

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
#            pre = Preconditioner(a, 'local')
#            a.Assemble()
#            inv = CGSolver(a.mat, pre.mat, maxsteps=1000)
#            gfu.vec.data = inv * f.vec
            #gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
            gfu=params.Base.Base_Functions.Solve(params, myfunc, rhs)
            
            return -data.u*params.Base.SymbolictoFunc(params, gfu)
            

class Reaction_Base_2D:
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
#        pre = Preconditioner(a, 'local')
#        a.Assemble()
#        inv = CGSolver(a.mat, pre.mat, maxsteps=1000)
#        gfu.vec.data = inv * f.vec         
#        return gfu
            
    def tilde_g_builder(self, domain, bc_top, bc_bottom, bc_left, bc_right):
        tilde_g=np.zeros((domain.shape[0], domain.shape[0]))
        tilde_g[0, :]=bc_top
        tilde_g[-1, :]=bc_bottom
        tilde_g[:, 0]=bc_left
        tilde_g[:, -1]=bc_right
        for i in range(1, domain.shape[0]-1):
            tilde_g[:, i]=np.interp(domain.coords[1, :], np.asarray([domain.coords[1, 0], domain.coords[1, -1]]), np.asarray([bc_top[i], bc_bottom[i]]))
        return tilde_g    


    def FunctoSymbolic(self, params, func):
        V = H1(params.mesh, order=1, dirichlet="left|right")
        u = GridFunction(V)
        N=params.domain.coords.shape[1]
        for i in range(0, N):
            for j in range(0, N):
                u.vec[i+N*j]=func[i, j]           
        return CoefficientFunction(u)
    
    

           
    def SymbolictoFunc(self, params, Symfunc):
        N=params.domain.coords.shape[1]
        Symfunc=CoefficientFunction(Symfunc)
        func=np.zeros((N, N))
        for i in range(0, N):
            for j in range(0, N):
                mip=params.mesh(params.domain.coords[0, i], params.domain.coords[1, j])
                func[i, j]=Symfunc(mip)
    
        return func
           
    def rc(self, params, c):
        res=self.mylaplace(params, params.v_star)
        return params.rhs+res-c*params.v_star    



    def mylaplace(self, params, func):
        N=params.domain.coords.shape[1]
        der=np.zeros((N,N))
        for i in range(1, N-1):
            for j in range(1, N-1):
                der[i, j]=(func[i+1, j]+func[i-1, j]-4*func[i, j]+func[i, j+1]+func[i, j-1])/(1/N**2)
        for i in range(1, N-1):       
            der[0, i]=(func[0, i+1]+func[0, i-1]-2*func[0, i])*N**2
            der[-1, i]=(func[-1, i+1]+func[-1, i-1]-2*func[-1, i])*N**2
            der[i,0]=(func[ i+1, 0]+func[i-1, 0]-2*func[i, 0])*N**2
            der[i,-1]=(func[ i+1, -1]+func[i-1, -1]-2*func[i, -1])*N**2
        der[0,0]=(der[0,1]+der[1, 0])/2
        der[-1, 0]=(der[-2, 0]+der[-1, 1])/2
        der[-1, -1]=(der[-2, -1]+der[-1, -2])/2
        der[0, -1]=(der[1, -1]+der[0, -2])/2
        return der  

    def construct_mesh(self, N):
        ngmesh = Mesh()
        ngmesh.SetGeometry(unit_square)
        ngmesh.dim = 2
        pnums = []
        for i in range(N + 1):
            for j in range(N + 1):
                pnums.append(ngmesh.Add(MeshPoint(Pnt(i / N, j / N, 0))))
            
        ngmesh.Add (FaceDescriptor(surfnr=1,domin=1,bc=1))
        ngmesh.SetMaterial(1, "mat")
        for j in range(N):
            for i in range(N):
                ngmesh.Add(Element2D(1, [pnums[i + j * (N + 1)],
                                   pnums[i + (j + 1) * (N + 1)],
                                   pnums[i + 1 + (j + 1) * (N + 1)],
                                   pnums[i + 1 + j * (N + 1)]]))
        
        for i in range(N):
            ngmesh.Add(Element1D([pnums[N + i * (N + 1)],
                           pnums[N + (i + 1) * (N + 1)]], index=1))
            ngmesh.Add(Element1D([pnums[0 + i * (N + 1)], pnums[0 + (i + 1) * (N + 1)]], index=1))
    
        # vertical boundaries
        for i in range(N):
            ngmesh.Add(Element1D([pnums[i], pnums[i + 1]], index=2))
            ngmesh.Add(Element1D([pnums[i + N * (N + 1)], pnums[i + 1 + N * (N + 1)]], index=2))
       
        mesh = ngsolve.Mesh(ngmesh)
        return mesh