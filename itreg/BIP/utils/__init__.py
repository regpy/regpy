# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:25:55 2019

@author: Hendrik Müller
"""

from itreg.util import classlogger

import numpy as np

class prior:
    
    log = classlogger

    def __init__(self):
        self.prior=None
        
    def gaussian(self, x):
        if self.gamma_prior is None:
            raise ValueError('Error: No gamma_prior is given')
        return -np.log(np.sqrt(2*np.pi*self.gamma_prior_abs))-1/2*np.dot(x-self.m_0, np.dot(self.gamma_prior, x-self.m_0))
    
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
    
    def hessian_gaussian(self, m, x):
        return np.dot(self.gamma_prior, x)
    
    def hessian_l1(self, m, x):
        grad_mx=np.sign(m+x)
        grad_m=np.sign(m)
        return grad_mx-grad_m
    
    def hessian_mean(self, m, x):
        y, deriv=self.op.linearize(m+x)
        grad_mx=0
        y, deriv=self.op.linearize(m)
        grad_m=0
        return grad_mx-grad_m
    
    
    
    
class likelihood:
    
    log=classlogger
    
    def __init__(self):
        self.likelihood=None
        
    def gaussian(self, x):
        if self.gamma_d is None:
            raise ValueError('Error: No gamma_d is given')
        return -1/2*np.log(2*np.pi*self.gamma_d_abs)-1/2*np.dot(self.op(x)-self.rhs, np.dot(self.gamma_d, self.op(x))-self.rhs)
    
    def l1(self, x):
        return np.log(self.l1_A)-self.l1_sigma*sum(abs(self.op(x)-self.rhs))
    
    def opnorm(self, x):
        return np.log(self.norm_A)-self.norm_sigma*self.op.range.norm(self.op(x)-self.rhs)
    
     
    def hessian_gaussian(self, m, x):
        y, deriv=self.op.linearize(m+x)
        grad_mx=-2*deriv.adjoint(np.dot(np.linalg.inv(self.gamma_d), (self.rhs-y)))
        y, deriv=self.op.linearize(m)
        grad_m=-2*deriv.adjoint(np.dot(np.linalg.inv(self.gamma_d), (self.rhs-y)))
        return grad_mx-grad_m
    
    def hessian_l1(self, m, x):
        y, deriv=self.op.linearize(m+x)
        grad_mx=deriv.adjoint(np.sign(y-self.rhs))
        y, deriv=self.op.linearize(m)
        grad_m=deriv.adjoint(np.sign(y-self.rhs))
        return grad_mx-grad_m
    
    def hessian_opnorm(self, m, x):
        y, deriv=self.op.lineariez(m+x)
        grad_mx=deriv.adjoint()
        y, deriv=self.op.linearize(m)
        grad_m=deriv.adjoint(np.sign(y-self.rhs))
        return grad_mx-grad_m
    
    
    
    
    
    
    
class Monte_Carlo:
    
    log=classlogger
    
    def __init__(self):
        self.sigma=None
        self.mu=None
        self.num=0
        
    def random_samples(self):
        R=np.random.normal(0, 1, self.gamma_post_half.shape[0]) 
        if self.gamma_prior_half is not None:
            m_prior=self.m_0+np.dot(self.gamma_prior_half, R)
        else:
            m_prior=None
        m_post=self.m_MAP+np.dot(self.gamma_post_half, R)
        return m_prior, m_post
        
    
    

        
        