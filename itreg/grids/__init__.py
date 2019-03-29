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
        self.ind_support=np.linspace(0, np.prod(shape), np.prod(shape))
        
    def support_interval(self, lower_bound, upper_bound):
        del self.ind_support
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.x_coo=self.coords
        self.ind_support=np.where(np.logical_and(self.x_coo<=upper_bound, self.x_coo>=lower_bound))[1]
        
    def support_square(self, lower_bound, upper_bound):
        del self.ind_support
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.x_coo=self.coords[0,:]
        self.y_coo=self.coords[1,:]
        self.Y, self.X=np.meshgrid(self.y_coo, self.x_coo)
        self.ind_support=np.asarray(np.logical_and.reduce(
                (np.reshape(self.X, np.prod(self.shape), order='F')<=upper_bound, 
                 np.reshape(self.X, np.prod(self.shape), order='F')>=lower_bound, 
                 np.reshape(self.Y, np.prod(self.shape), order='F')<=upper_bound, 
                 np.reshape(self.Y, np.prod(self.shape), order='F')>=lower_bound))).nonzero()
        
    def support_cube(self, lower_bound, upper_bound):
        del self.ind_support
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.x_coo=self.coords[0,:]
        self.y_coo=self.coords[1,:]
        self.z_coo=self.coords[2,:]
        self.Z, self.Y, self.X=np.meshgrid(self.z_coo, self.y_coo, self.x_coo)
        self.ind_support=np.asarray(np.logical_and.reduce(
                (np.reshape(self.X, np.prod(self.shape), order='F')<=upper_bound, 
                 np.reshape(self.X, np.prod(self.shape), order='F')>=lower_bound, 
                 np.reshape(self.Y, np.prod(self.shape), order='F')<=upper_bound, 
                 np.reshape(self.Y, np.prod(self.shape), order='F')>=lower_bound,
                 np.reshape(self.Z, np.prod(self.shape), order='F')<=upper_bound,
                 np.reshape(self.Z, np.prod(self.shape), order='F')>=lower_bound))).nonzero()        
        
        
    def support_circle(self, rho):
        del self.ind_support
        self.rho=rho
        self.x_coo=self.coords[0,:]
        self.y_coo=self.coords[1,:]
        self.Y, self.X=np.meshgrid(self.y_coo, self.x_coo)
        self.ind_support=np.asarray(np.reshape(self.X, np.prod(self.shape), order='F')**2+np.reshape(self.Y, np.prod(self.shape), order='F')**2<=rho**2).nonzero()
        return
    
    def support_sphere(self, rho):
        del self.ind_support
        self.rho=rho
        self.x_coo=self.coords[0,:]
        self.y_coo=self.coords[1,:]
        self.z_coo=self.coords[2,:]
        self.Z, self.Y, self.X=np.meshgrid(self.z_coo, self.y_coo, self.x_coo)
        self.ind_support=np.asarray(np.reshape(self.X, np.prod(self.shape), order='F')**2+np.reshape(self.Y, np.prod(self.shape), order='F')**2+np.reshape(self.Z, np.prod(self.shape), order='F')**2<=rho**2).nonzero()
        return
    
from .grids import Square_1D, Square_2D, Square_3D
