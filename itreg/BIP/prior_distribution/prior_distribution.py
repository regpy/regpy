# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:47:06 2019

@author: Bjoern Mueller
"""

from . import prior
import numpy as np
import random as rd
import scipy.optimize

class User_defined_prior(prior):
    
    def __init__(self, op, logprob, gradient, hessian, m_0):
        super().__init__()
        self.prior=logprob
        self.hessian=hessian
        self.gradient=gradient
        self.op=op
        self.m_0=m_0
        
class gaussian(prior):  
    def __init__(self, gamma_prior, op, m_0=None):
        super().__init__()
        if gamma_prior is None:
                raise ValueError('Error: No prior covariance matrix')
        self.op=op
        if m_0 is None:
            self.m_0=np.zeros(self.op.domain.coords.shape[0])
        else:
            self.m_0=m_0
        self.gamma_prior=gamma_prior
        self.gamma_prior_abs=np.linalg.det(self.gamma_prior)
        D, S=np.linalg.eig(self.gamma_prior)
        self.gamma_prior_half=np.dot(S.transpose(), np.dot(np.diag(np.sqrt(D)), S))
        self.hessian=self.hessian_gaussian
        self.gradient=self.gradient_gaussian
        self.prior=self.gaussian

class l1(prior):
    def __init__(self, l1_A, l1_sigma, op):
        super().__init__()
        if l1_A or l1_sigma is None:
            raise ValueError('Error: Not all necessary parameters are specified')
        self.l1_A=l1_A
        self.l1_sigma=l1_sigma
        self.op=op
        self.hessian=self.hessian_l1
        self.gradient=self.gradient_l1
        self.prior=self.l1
        
class mean(prior):       
    def __init__(self, op, x_lower=None, x_upper=None):
        super().__init__()
        self.op=op
        self.x_lower=x_lower
        self.x_upper=x_upper
        self.prior=self.mean
        self.gradient=self.gradient_mean
        self.hessian=self.hessian_mean
        
class unity(prior):
    def __init__(self, op):
        super().__init__()
        self.op=op
        self.prior=(lambda x: 0)
        self.gradient=(lambda x: 0)
        self.hessian=(lambda x: 0)
        
class tikhonov(prior):
    def __init__(self, op, regpar):
        self.op=op
        self.regpar=regpar
        self.prior=self.tikhonov
        self.gradient=self.gradient_tikhonov
        self.hessian=self.hessian_tikhonov