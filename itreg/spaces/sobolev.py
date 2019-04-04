# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:01:52 2019

@author: hendr
"""

from . import Space

import numpy as np

class Sobolev(Space):

     def __init__(self, grid, sobo_index):
         self.parameters_domain=start_sobolev(grid, sobo_index)
         super().__init__(grid.shape, np.size(grid.ind_support), grid.coords)

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
     
class parameters_domain_sobolev:
     def __init__(self):
         return
         
def start_sobolev(grid, sobo_index):
     par_dom=parameters_domain_sobolev
     par_dom.sobo_index=sobo_index
     dimension=np.size(grid.shape)
     par_dom.N=grid.shape
     par_dom.x_coo=grid.x_coo
     par_dom.y_coo=grid.y_coo
     if dimension==3:
         par_dom.z_coo=grid.z_coo
     par_dom.Y, par_dom.X=grid.Y, grid.X
     if dimension==3:
         par_dom.Z=grid.Z
     rho=grid.rho
     par_dom.ind_support=grid.ind_support
     if dimension==2:
         par_dom.Fourierweights=np.fft.fftshift((1+(4*grid.rho/grid.shape[0])**2*grid.X*grid.X+(4*grid.rho/grid.shape[1])**2*grid.Y*grid.Y)**(sobo_index))
     if dimension==3:
         par_dom.Fourierweights=np.fft.fftshift((1+(4*grid.rho/grid.shape[0])**2*grid.X*grid.X+(4*grid.rho/grid.shape[1])**2*grid.Y*grid.Y+(4*grid.rho/grid.shape[2])**2*grid.Z*grid.Z)**(sobo_index))
     return par_dom