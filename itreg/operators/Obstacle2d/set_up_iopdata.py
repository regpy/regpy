# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 20:40:58 2019

@author: Björn Müller
"""
import numpy as np
import scipy.linalg as scla

from .functions.bessj0 import bessj0
from .functions.bessj1 import bessj1
from .functions.bessy0 import bessy0
from .functions.bessy1 import bessy1

def setup_iop_data(bd,kappa):
    """ computes data needed to set up the boundary integral matrices
     to avoid repeated computations"""
    dim = len(bd.z)
#    dat.kappa = kappa
    
    """compute matrix of distances of grid points"""
    t1=np.matlibrepmat(bd.z[0,:].T,1,dim)-np.matlib.repmat(bd.z[1,:],dim,1)
    t2=np.matlib.repmat(bd.z[2,:].T,1,dim)-np.matlib.repmat(bd.z[2,:],dim,1)
    kdist = kappa*np.sqrt(t1**2 + t2**2)
    bessj0_kdist = bessj0(kdist) 
    bessH0 = bessj0_kdist+complex(0, 1)*bessy0(kdist,bessj0_kdist)
    #bessH0 = besselh(0,1,dat.kdist)
    
    bessj1_kdist= bessj1(kdist)
    bessH1quot = (bessj1_kdist+ complex(0,1)*bessy1(kdist,bessj1_kdist))/kdist
    #bessH1quot = besselh(1,1,dat.kdist) / dat.kdist
    for j in range(0, dim):
        bessH0[j,j]=1

    
    """set up prototyp of the singularity of boundary integral operators"""
    t=2*np.pi*np.arange(1, dim)/dim
    logsin = scla.toeplitz([1, np.log(4*np.sin(t/2)**2)])
    
    """quadrature weight for weight function log(4*(sin(t-tau))**2)"""
    sign=np.ones(dim)
    sign[np.arange(1, dim, 2)]=-1
    t = 2*np.pi*np.arange(0, dim)/dim 
    s=0
    for m in range(0, dim/2-1):
        s=s+np.cos((m+1)*t)/(m+1)
    logsin_weights = scla.toeplitz(-2*(s + sign/dim)/dim)
    
    #euler constant 'eulergamma'
    Euler_gamma =  0.577215664901532860606512
    
    return dat_object(kappa, Euler_gamma, logsin_weights, logsin, bessH0, bessH1quot, \
                      kdist, )
    
    
class dat_object(object):
    def __init__(self, kappa, Euler_gamma, logsin_weights, logsin, bessH0, bessH1quot, \
                      kdist, ):
        self.kappa=kappa
        self.Euler_gamma=Euler_gamma
        self.logsin_weights=logsin_weights
        self.logsin=logsin
        self.bessH1quot=bessH1quot
        self.kdist=kdist
