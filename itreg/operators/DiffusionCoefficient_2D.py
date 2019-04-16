# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:54:23 2019

@author: Hendrik Müller
"""

from . import NonlinearOperator, OperatorImplementation, Params
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
import netgen.gui
import ngsolve
from netgen.meshing import *



class DiffusionCoefficient(NonlinearOperator):
    

    def __init__(self, domain, rhs, bc_left=None, bc_right=None, bc_bottom=None, bc_top=None, range=None, spacing=1):
        if bc_top is None:
            bc_top=rhs[0, :]
        if bc_bottom is None:
            bc_bottom=rhs[-1, :]
        if bc_left is None:
            bc_left=rhs[:, 0]
        if bc_right is None:
            bc_right=rhs[:, -1]   
        
        
        N=rhs.shape[0]-1
        mesh=construct_mesh(N)
        
        range = range or domain
        super().__init__(Params(domain, range, rhs=rhs, bc_left=bc_left, bc_right=bc_right, bc_top=bc_top, bc_bottom=bc_bottom, mesh=mesh, spacing=spacing))
        
    @instantiate
    class operator(OperatorImplementation):
        def eval(self, params, diff, data, differentiate, **kwargs):
#            B=B_builder(params, c)
            r_diff=rdiff(params, diff)
#            print(r_diff)
#            rhs=rhs_builder(params, r_c)
            rhs=FunctoSymbolic(params, r_diff)
            #mip=params.mesh(1, 0.5)
            #print(rhs(mip))
#            coeff= np.linalg.solve(B, rhs)
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
            pre = Preconditioner(a, 'local')
            a.Assemble()
            inv = CGSolver(a.mat, pre.mat, maxsteps=1000)
            gfu.vec.data = inv * f.vec
            #gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
            

            if differentiate:
                data.u=SymbolictoFunc(params, gfu)+tilde_g_builder(params)
                data.diff=diff
            return SymbolictoFunc(params, gfu)+tilde_g_builder(params)

    @instantiate
    class derivative(OperatorImplementation):
        def eval(self, params, x, data, **kwargs):
            prod=mygradient(params, data.u)
            prod[:, :, 0]=prod[:, :, 0]*x
            prod[:, :, 1]=prod[:, :, 1]*x
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
            pre = Preconditioner(a, 'local')
            a.Assemble()
            inv = CGSolver(a.mat, pre.mat, maxsteps=1000)
            gfu.vec.data = inv * f.vec
            #gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
            return -SymbolictoFunc(params, gfu)
            

        def adjoint(self, params, y, data, **kwargs):
            rhs=FunctoSymbolic(params, y)
            
            myfunc=FunctoSymbolic(params, data.diff)
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
            pre = Preconditioner(a, 'local')
            a.Assemble()
            inv = CGSolver(a.mat, pre.mat, maxsteps=1000)
            gfu.vec.data = inv * f.vec
            #gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
            return mygradient(params, data.u)[:, :, 0]*mygradient(params, SymbolictoFunc(params, gfu))[:, :, 0]+mygradient(params, data.u)[:, :, 1]*mygradient(params, SymbolictoFunc(params, gfu))[:, :, 1]
            
            
            
def tilde_g_builder(params):
    tilde_g=np.zeros((params.domain.coords.shape[1], params.domain.coords.shape[1]))
    tilde_g[0, :]=params.bc_top
    tilde_g[-1, :]=params.bc_bottom
    tilde_g[:, 0]=params.bc_left
    tilde_g[:, -1]=params.bc_right
    for i in range(1, params.domain.coords.shape[1]-1):
        tilde_g[:, i]=np.interp(params.domain.coords[1, :], np.asarray([params.domain.coords[1, 0], params.domain.coords[1, -1]]), np.asarray([params.bc_top[i], params.bc_bottom[i]]))
    v_star=tilde_g
    return v_star     



def FunctoSymbolic(params, func):
        V = H1(params.mesh, order=1, dirichlet="left|right")
        u = GridFunction(V)
        N=params.domain.coords.shape[1]
        for i in range(0, N):
            for j in range(0, N):
                u.vec[i+N*j]=func[i, j]           
        return CoefficientFunction(u)
    
    

           
def SymbolictoFunc(params, Symfunc):
    N=params.domain.coords.shape[1]
    Symfunc=CoefficientFunction(Symfunc)
    func=np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            mip=params.mesh(params.domain.coords[0, i], params.domain.coords[1, j])
            func[i, j]=Symfunc(mip)
    
    return func
        

def rdiff(params, diff):
    N=params.domain.coords.shape[1]
#    print(tilde_g_builder(params).shape)
#define gradient by myself
    prod=mygradient(params, tilde_g_builder(params))
    prod[:, :, 0]=diff*prod[:, :, 0]
    prod[:, :, 1]=diff*prod[:, :, 1]
    res=mydiv(params, prod)
#    print(type(res))
    return params.rhs+res      


def mygradient(params, func):
    N=params.domain.coords.shape[1]
    der=np.zeros((N, N, 2))
    for i in range(0, N):
        der[i, :, 0]=np.gradient(func[i, :])*N
        der[:, i, 1]=np.gradient(func[:, i])*N
    return der

def mydiv(params, func): 
    N=params.domain.coords.shape[1]
    der=np.zeros((N, N))
    for i in range(0, N):
        der[i, :]=der[i, :]+np.gradient(func[i, :, 0])*N
        der[:, i]=der[:, i]+np.gradient(func[:, i, 1])*N
    return der

def construct_mesh(N):
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