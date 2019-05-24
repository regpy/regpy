# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:19:29 2019

@author: Bjoern Mueller
"""

from . import likelihood
import numpy as np
import random as rd
import scipy.optimize

class User_defined_prior(likelihood):
    
    def __init__(self, op, logprob, gradient, hessian, m_0):
        super().__init__()
        self.likelihood=logprob
        self.op=op
        self.hessian=hessian
        self.gradient=gradient
        self.m_0=m_0
        
class gaussian(likelihood):
       
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
        
class l1(likelihood):
    
    def __init__(self, op, l1_A, l1_sigma):
        super().__init__()
        if l1_A or l1_sigma is None:
            raise ValueError('Error: Not all necessary parameters are specified')
        self.op=op
        self.l1_A=l1_A
        self.l1_sigma=l1_sigma
        self.likelihood=self.l1
        self.gradient=self.gradient_l1
        self.hessian=self.hessian_l1
        
class opnorm(likelihood):
    
    def __init__(self, op, norm_A, norm_sigma):
        super().__init__()
        if norm_A or norm_sigma is None:
            raise ValueError('Error: Not all necessary parameters are specified')
        self.op=op
        self.norm_A=norm_A
        self.norm_sigma=norm_sigma
        self.likelihood=self.opnorm  
        self.gradient=self.gradient_opnorm
        self.hessian=self.hessian_opnorm
        
class unity(likelihood):
    def __init__(self, op):
        super().__init__()
        self.op=op
        self.likelihood=(lambda x: 0)
        self.gradient=(lambda x: 0)
        self.hessian=(lambda x: 0)

