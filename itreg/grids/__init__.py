# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:27:59 2019

@author: hendr
"""
from itreg.util import classlogger

import numpy as np

class Grid:
    log=classlogger
    
    def __init__(self, coords, shape):
        self.coords=coords
        self.shape=shape
        
    def support_circle(self, rho):
        self.rho=rho
        self.x_coo=self.coords[0,:]
        self.y_coo=self.coords[1,:]
        self.Y, self.X=np.meshgrid(self.y_coo, self.x_coo)
        self.ind_support=np.asarray(np.reshape(self.X, np.prod(self.shape), order='F')**2+np.reshape(self.Y, np.prod(self.shape), order='F')**2<=rho**2).nonzero()
        return
        
from .grids import Square
