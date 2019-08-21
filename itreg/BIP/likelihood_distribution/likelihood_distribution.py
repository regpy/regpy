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

class User_defined_likelihood(object):
    
    def __init__(self, setting, logprob, gradient, hessian, m_0):
        super().__init__()
        self.likelihood=logprob
        self.setting=setting
        self.hessian=hessian
        self.gradient=gradient
        self.m_0=m_0
        
class gaussian(object):
       
    def __init__(self, setting, gamma_d, rhs, offset=None, inv_offset=None):
        super().__init__()
        if gamma_d is None:
           raise ValueError('Error: No data covariance matrix')
        if rhs is None:
            raise ValueError('Error: No right hand side is given')
        self.setting=setting
        self.rhs=rhs
        self.gamma_d=gamma_d
        self.inv_offset=inv_offset or 1e-5
        self.D, self.U=np.linalg.eig(self.gamma_d)
        self.gamma_d_half_inv=np.dot(self.U.T, np.dot(np.diag(np.sqrt(1/self.D)+self.inv_offset), self.U))
        self.gamma_d_inv=np.dot(self.U.T, np.dot(np.diag(1/self.D+self.inv_offset), self.U))
        self.gamma_d_abs=np.linalg.det(self.gamma_d)
        self.likelihood=self.gaussian
        self.gradient=self.gradient_gaussian
        self.hessian=self.hessian_gaussian
        self.offset=offset or 1e-10
        self.len_codomain=np.prod(self.setting.op.codomain.shape)
        
    
        
    def gaussian(self, x):      
        misfit=np.dot((self.setting.op(x)-self.rhs).reshape(self.len_codomain), self.gamma_d_half_inv)
#        return -1/2*np.log(2*np.pi*self.gamma_d_abs+self.offset)-\
#            1/2*np.dot(misfit.reshape(self.len_codomain), np.conjugate(misfit.reshape(self.len_codomain))).real
        return -1/2*np.dot(misfit.reshape(self.len_codomain), np.conjugate(misfit.reshape(self.len_codomain))).real
    
    def gradient_gaussian(self, x):
#        y, deriv=self.setting.op.linearize(x)
        y=self.setting.op._eval(x, differentiate=True)
        misfit=(y-self.rhs).reshape(self.len_codomain)
        res=np.dot(self.gamma_d_inv, misfit)
        res=self.setting.op._adjoint(misfit)
        return -self.setting.op._adjoint(self.setting.codomain.gram_inv(res.reshape(self.setting.op.codomain.shape))).real

    
    def hessian_gaussian(self, m, x):
        grad_mx=self.gradient_gaussian(m+x)
        grad_m=self.gradient_gaussian(m)
        return grad_mx-grad_m

        
class l1(object):
    
    def __init__(self, setting, l1_sigma, rhs):
        super().__init__()
        if l1_sigma is None:
            raise ValueError('Error: Not all necessary parameters are specified')
        self.setting=setting
        self.l1_sigma=l1_sigma
        self.rhs=rhs
        self.likelihood=self.l1
        self.gradient=self.gradient_l1
        self.hessian=self.hessian_l1
        
    def l1(self, x):
        return (-1)*self.l1_sigma*np.sum(abs(self.setting.op(x)-self.rhs))

    
    def gradient_l1(self, x):
        y, deriv=self.setting.op.linearize(x)
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
    def __init__(self, setting):
        super().__init__()
        self.setting=setting
        self.likelihood=(lambda x: 0)
        self.gradient=(lambda x: self.setting.op.domain.zeros())
        self.hessian=(lambda x, y: self.setting.op.domain.zeros())
        
class tikhonov(object):
    

    def __init__(self, setting, rhs):
        self.setting=setting
        self.likelihood=self.tikhonov
        self.gradient=self.gradient_tikhonov
        self.hessian=self.hessian_tikhonov
        self.rhs=rhs
        

    def tikhonov(self, x):
        y=self.setting.op(x)-self.rhs
        return -0.5 * self.setting.codomain.inner(y, y)

    
    

    def gradient_tikhonov(self, x):
        y, deriv=self.setting.op.linearize(x)
        y-=self.rhs
        return -deriv.adjoint(self.setting.codomain.gram(y))
    
    def hessian_tikhonov(self, m, x):
        #print(m+x)
        grad_mx=self.gradient_tikhonov(m+x)
        grad_m=self.gradient_tikhonov(m)
        return grad_mx-grad_m

