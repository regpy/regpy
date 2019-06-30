# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:15:13 2019

@author: hendr
"""

from . import Space

import numpy as np
import netgen.gui
from ngsolve import *

class H1_NGSolve(Space):

     def __init__(self, mesh, order=1, dirichlet="bottom|right|top|left"):
         self.fes=H1(mesh, order=order, dirichlet=dirichlet)
         self.mesh=mesh
         N, coords=find_coordinates(mesh)
         self.parameters_domain=start_sobolev(N, coords, mesh)
         super().__init__(N, np.prod(N), coords)

     def gram(self, x):
         v=np.zeros(self.parameters_domain.N)
         np.put(v, self.parameters_domain.ind_support, x)
         v=np.fft.ifftn(np.fft.fftn(v)*self.parameters_domain.Fourierweights)
         v=np.reshape(v, (np.prod(self.parameters_domain.N), 1), order='F')
         return v[self.parameters_domain.ind_support]

     def gram_inv(self, x):
         v=np.zeros(self.parameters_domain.N)
         np.put(v, self.parameters_domain.ind_support, x)        
         v=np.fft.ifftn(np.fft.fftn(v)/self.parameters_domain.Fourierweights) 
         v=np.reshape(v, (np.prod(self.parameters_domain.N), 1), order='F')         
         return v[self.parameters_domain.ind_support]
     
class parameters_domain_sobolev_ngsolve:
     def __init__(self):
         return
         
def start_sobolev(N, coords, mesh):
     par_dom=parameters_domain_sobolev_ngsolve
     sobo_index=1
     dimension=mesh.dim
     par_dom.N=N
     if dimension==1:
         par_dom.x_coo=coords
     if dimension==2 or dimension==3:
         par_dom.x_coo=coords[0,:]
         par_dom.y_coo=coords[1,:]
     if dimension==3:
         par_dom.z_coo=coords[2,:]
         
     if dimension==1:
         par_dom.X=par_dom.x_coo
     if dimension==2:
         par_dom.Y, par_dom.X=np.meshgrid(par_dom.x_coo, par_dom.y_coo)
     if dimension==3:
         par_dom.Z, par_dom.Y, par_dom.X=np.meshgrid(par_dom.x_coo, par_dom.y_coo, par_dom.z_coo)
         

     par_dom.rho=1
    
#     par_dom.ind_support=grid.ind_support
     if dimension==1:
         par_dom.Fourierweights=np.fft.fftshift((1+(4*par_dom.rho/par_dom.x_coo.shape[0])**2*par_dom.X*par_dom.X)**(sobo_index))         
     if dimension==2:
         par_dom.Fourierweights=np.fft.fftshift((1+(4*par_dom.rho/par_dom.x_coo.shape[0])**2*par_dom.X*par_dom.X+(4*par_dom.rho/par_dom.y_coo.shape[0])**2*par_dom.Y*par_dom.Y)**(sobo_index))
     if dimension==3:
         par_dom.Fourierweights=np.fft.fftshift((1+(4*par_dom.rho/par_dom.x_coo.shape[0])**2*par_dom.X*par_dom.X+(4*par_dom.rho/par_dom.y_coo.shape[0])**2*par_dom.Y*par_dom.Y+(4*par_dom.rho/par_dom.z_coo.shape[0])**2*par_dom.Z*par_dom.Z)**(sobo_index))
     return par_dom
 
def find_coordinates(mesh):
    if mesh.dim==1:
        N=np.size(mesh.ngmesh.Points())
    if mesh.dim==2:
        N=(2, np.size(mesh.ngmesh.Points()))
    if mesh.dim==3:
        N=(3, np.size(mesh.ngmesh.Points()))
    coords=np.zeros(N)
    i=0
    for p in mesh.ngmesh.Points():
        if mesh.dim==1:
            coords[i]=p.p[0]
        if mesh.dim==2:
            coords[0][i]=p.p[0]
            coords[1][i]=p.p[1]
        if mesh.dim==3:
            coords[0][i]=p.p[0]
            coords[1][i]=p.p[1] 
            coords[2][i]=p.p[2]
        i+=1
    return [N, coords]
    