# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:19:29 2019

@author: Bjoern Mueller
"""


import numpy as np
import random as rd
import scipy.optimize


'''TODO: The Gram matrices have to be included.
'''    

class User_defined_prior(object):
    
    def __init__(self, op, logprob, gradient, hessian, m_0):
        super().__init__()
        self.likelihood=logprob
        self.op=op
        self.hessian=hessian
        self.gradient=gradient
        self.m_0=m_0
        
class gaussian(object):
       
    def __init__(self, op, gamma_d, rhs):
        super().__init__()
        if gamma_d is None:
           raise ValueError('Error: No data covariance matrix')
        if rhs is None:
            raise ValueError('Error: No right hand side is given')
        self.op=op
        self.rhs=rhs
        self.gamma_d=gamma_d
        self.gamma_d_abs=np.linalg.det(self.gamma_d)
        self.likelihood=self.gaussian
        self.gradient=self.gradient_gaussian
        self.hessian=self.hessian_gaussian
        
    def gaussian(self, x):
        if self.gamma_d is None:
            raise ValueError('Error: No gamma_d is given')
        return -1/2*np.log(2*np.pi*self.gamma_d_abs)-\
            1/2*np.dot(self.op(x)-self.rhs, self.op.range.gram(np.dot(self.gamma_d, self.op(x))-self.rhs))
    
    def gradient_gaussian(self, x):
        y, deriv=self.op.linearize(x)
        return -deriv.adjoint(self.op.range.gram_inv(np.dot(np.linalg.inv(self.gamma_d), (self.rhs-y))))
    
    def hessian_gaussian(self, m, x):
        grad_mx=self.gradient_gaussian(m+x)
        grad_m=self.gradient_gaussian(m)
        return grad_mx-grad_m

        
class l1(object):
    
    def __init__(self, op, l1_sigma, rhs):
        super().__init__()
        if l1_sigma is None:
            raise ValueError('Error: Not all necessary parameters are specified')
        self.op=op
        self.l1_sigma=l1_sigma
        self.rhs=rhs
        self.likelihood=self.l1
        self.gradient=self.gradient_l1
        self.hessian=self.hessian_l1
        
    def l1(self, x):
        return (-1)*self.l1_sigma*np.sum(abs(self.op(x)-self.rhs))

    
    def gradient_l1(self, x):
        y, deriv=self.op.linearize(x)
        return deriv.adjoint(np.sign(y-self.rhs))
    
    def hessian_l1(self, m, x):
        grad_mx=self.gradient_l1(m+x)
        grad_m=self.gradient_l1(m)
        return grad_mx-grad_m
        
#class opnorm(object):
    
#    def __init__(self, op, norm_A, norm_sigma):
#        super().__init__()
#        if norm_A or norm_sigma is None:
#            raise ValueError('Error: Not all necessary parameters are specified')
#        self.op=op
#        self.norm_A=norm_A
#        self.norm_sigma=norm_sigma
#        self.likelihood=self.opnorm  
#        self.gradient=self.gradient_opnorm
#        self.hessian=self.hessian_opnorm
        
#    def gradient_opnorm(self, x):
#        y, deriv=self.op.linearize(x)
#        return deriv.adjoint(np.sign(y-self.rhs))
        
#    def hessian_opnorm(self, m, x):
#        grad_mx=self.hessian(m+x)
#        grad_m=self.hessian(m)
#        return grad_mx-grad_m
        
class unity(object):
    def __init__(self, op):
        super().__init__()
        self.op=op
        self.likelihood=(lambda x: 0)
        self.gradient=(lambda x: 0)
        self.hessian=(lambda x: 0)

