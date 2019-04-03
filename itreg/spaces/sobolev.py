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
         super().__init__(np.size(grid.ind_support), grid.coords)

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
     par_dom.Y, par_dom.X=grid.Y, grid.X
     rho=grid.rho
     par_dom.ind_support=grid.ind_support
     par_dom.Fourierweights=np.fft.fftshift((1+(4*grid.rho/grid.shape[0])**2*grid.X*grid.X+(4*grid.rho/grid.shape[1])**2*grid.Y*grid.Y)**(sobo_index))
     return par_dom