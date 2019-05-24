# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:15:28 2019

@author: Bjoern Mueller
"""

from itreg.util import classlogger

import numpy as np

class prior:
    
    log = classlogger

    def __init__(self):
        self.prior=None
        self.hessian=None
        self.op=None
        
    def gaussian(self, x):
        if self.gamma_prior is None:
            raise ValueError('Error: No gamma_prior is given')

        return -np.log(np.sqrt(2*np.pi*self.gamma_prior_abs))-1/2*np.dot(x-self.m_0, self.op.domain.gram(np.dot(self.gamma_prior, x-self.m_0)))
    
    def l1(self, x):
        return self.l1_A-self.l1_sigma*sum(abs(x))
    
    def mean(self, x):
        res=0
        if self.x_lower is not None:
            if not self.x_lower.all()<=x.all():
                res=float('inf')
        if self.x_upper is not None:
            if not self.x_upper.all()>=x.all():
                res=float('inf')
        return res
    
    def tikhonov(self, x):
        y=self.op(x)-self.rhs
        return - 0.5 * (self.op.range.inner(y, y)+self.regpar*self.op.domain.inner(x, x))




    
    def gradient_gaussian(self, x):
        return self.op.domain.gram(np.dot(self.gamma_prior, x-self.m_0))
    
    def gradient_l1(self, x):
        return -self.l1_sigma*np.sign(x)
    
    def gradient_mean(self, x):
        return 0
    
    def gradient_tikhonov(self, x):
        y, deriv=self.op.linearize(x)
        y-=self.rhs
        return -(deriv.adjoint(self.op.range.gram(y))+self.regpar*self.op.domain.gram(x))
    
    
    
    
    
    
    
    def hessian_gaussian(self, m, x):
        return np.dot(self.gamma_prior, x)
    
    def hessian_l1(self, m, x):
        grad_mx=self.gradient_l1(m+x)
        grad_m=self.gradient_l1(m)
        return grad_mx-grad_m
    
    def hessian_mean(self, m, x):
        y, deriv=self.op.linearize(m+x)
        grad_mx=0
        y, deriv=self.op.linearize(m)
        grad_m=0
        return grad_mx-grad_m
    
    def hessian_tikhonov(self, m, x):
        grad_mx=self.gradient_tikhonov(m+x)
        grad_m=self.gradient_tikhonov(m)
        return grad_mx-grad_m