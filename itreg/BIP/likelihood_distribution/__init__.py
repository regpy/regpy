# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:35:14 2019

@author: Bjoern Mueller
"""

from itreg.util import classlogger

import numpy as np

class likelihood:
    
    log = classlogger

    def __init__(self):
        self.likelihood=None
        self.hessian=None
        self.op=None
        
    def gaussian(self, x):
        if self.gamma_d is None:
            raise ValueError('Error: No gamma_d is given')
        return -1/2*np.log(2*np.pi*self.gamma_d_abs)-1/2*np.dot(self.op(x)-self.rhs, self.op.range.gram(np.dot(self.gamma_d, self.op(x))-self.rhs))
    
    def l1(self, x):
        return np.log(self.l1_A)-self.l1_sigma*sum(abs(self.op(x)-self.rhs))
    
    def opnorm(self, x):
        return np.log(self.norm_A)-self.norm_sigma*self.op.range.norm(self.op(x)-self.rhs)
    
#    def tikhonov(self, x): 
#        y=self.op(x)-self.rhs

 #       return - 0.5 * (self.op.range.inner(y, y)+self.regpar*self.op.domain.inner(x, x))
    
    
    '''TODO: The Gram matrices have to be included.
    '''    
    
    
    
    def gradient_gaussian(self, x):
        y, deriv=self.op.linearize(x)
        return -deriv.adjoint(self.op.range.gram_inv(np.dot(np.linalg.inv(self.gamma_d), (self.rhs-y))))
    
    def gradient_l1(self, x):
        y, deriv=self.op.linearize(x)
        return deriv.adjoint(np.sign(y-self.rhs))
    
    def gradient_opnorm(self, x):
        y, deriv=self.op.linearize(x)
        return deriv.adjoint(np.sign(y-self.rhs))
    
    
    
     
    def hessian_gaussian(self, m, x):
        grad_mx=self.gradient_gaussian(m+x)
        grad_m=self.gradient_gaussian(m)
        return grad_mx-grad_m
    
    def hessian_l1(self, m, x):
        grad_mx=self.gradient_l1(m+x)
        grad_m=self.gradient_l1(m)
        return grad_mx-grad_m
    
    def hessian_opnorm(self, m, x):
        grad_mx=self.hessian(m+x)
        grad_m=self.hessian(m)
        return grad_mx-grad_m