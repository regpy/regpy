# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:02:37 2019

@author: Hendrik Müller
"""

from . import prior, likelihood, Monte_Carlo
import numpy as np
import random as rd
import scipy.optimize

class prior_distribution(prior):
    
     def __init__(self, op, prior_type, m_0, gamma_prior=None, l1_A=None, l1_sigma=None, x_upper=None, x_lower=None):
        super().__init__()
        self.prior_type=prior_type
        self.op=op
        self.m_0=m_0
        
        
        
        if self.prior_type is None:
            raise ValueError('Error: No prior is given')
        elif self.prior_type=='gaussian':
            if gamma_prior is None:
                raise ValueError('Error: No prior covariance matrix')
            self.gamma_prior=gamma_prior
            self.gamma_prior_abs=np.linalg.det(self.gamma_prior)
            D, S=np.linalg.eig(self.gamma_prior)
            self.gamma_prior_half=np.dot(S.transpose(), np.dot(np.diag(np.sqrt(D)), S))
            self.hessian=self.hessian_gaussian
            self.prior=self.gaussian

            
        elif self.prior_type == 'l1':
            if l1_A or l1_sigma is None:
                raise ValueError('Error: Not all necessary parameters are specified')
            self.l1_A=l1_A
            self.l1_sigma=l1_sigma
            self.hessian=self.hessian_l1
            self.prior=self.l1
            
        elif self.prior_type == 'mean':
            self.prior=self.mean
            self.hessian=self.hessian_mean
            
        else:
            raise ValueError('Error: prior_type is not known')
        
        
        
        
        
        
        
        



class likelihood_distribution(likelihood):
    
        def __init__(self, op, likelihood_type, rhs, gamma_d=None, l1_A=None, l1_sigma=None, norm_A=None, norm_sigma=None):
            super().__init__()
            self.op=op
            self.rhs=rhs
            self.likelihood_type=likelihood_type
            
            if self.likelihood_type=='gaussian':
                if gamma_d is None:
                   raise ValueError('Error: No data covariance matrix')
                self.gamma_d=gamma_d
                self.gamma_d_abs=np.linalg.det(self.gamma_d)
                self.likelihood=self.gaussian
                self.hessian=self.hessian_gaussian
                
                
            elif self.likelihood_type=='l1':
                if l1_A or l1_sigma is None:
                    raise ValueError('Error: Not all necessary parameters are specified')
                self.l1_A=l1_A
                self.l1_sigma=l1_sigma
                self.likelihood=self.l1
                self.hessian=self.hessian_l1
                
            elif self.likelihood_type=='opnorm':
                if norm_A or norm_sigma is None:
                    raise ValueError('Error: Not all necessary parameters are specified')
                self.norm_A=norm_A
                self.norm_sigma=norm_sigma
                self.likelihood=self.opnorm  
                self.hessian=self.hessian_opnorm
                
            else:
                raise ValueError('Error: likelihood_type is not known')
                
                
                
class Monte_Carlo_evaluation(Monte_Carlo):
    
        def __init__(self, op, distribution, gamma_post_half, m_0, m_MAP, maxnum=None, maxhits=None, gamma_prior_half=None):
            super().__init__()
#            self.maxnum=maxnum or 10**5
            self.maxnum=maxnum
#            self.maxhits=maxhits or 1000
            self.maxhits=maxhits
            self.op=op
            self.m_MAP=m_MAP
            if m_0 is not None:
                self.m_0=m_0
            else: 
                self.m_0=np.zeros(self.op.params.domain.coords.shape[0])
            self.distribution=distribution
            if gamma_prior_half is not None:
                self.gamma_prior_half=gamma_prior_half
            else:
                self.gamma_prior_half=None
            self.gamma_post_half=gamma_post_half
            
        def maximize(self):
            res=scipy.optimize.minimize(-self.distribution, self.m_0)
            self.mu=res.x
            self.distribution_max=self.distribution(self.mu)
            
            
        def Metropolis(self):
            number=0
            hits=0
            N=self.op.params.domain.coords.shape[0]
            vec=np.zeros((self.max_hits, N))
            self.sigma=np.zeros(N)
            self.maximize()
            while number<self.maxnumber:
                x=np.random.rand(N)
                q=rd.random()
                if self.distribution(x)/self.distribution_max>q:
                    vec[hits]=x
                    hits+=1
                number+=1
            for i in range(0, N):
                self.mu[i]=np.mean(vec[i, 0:hits])
                self.sigma[i]=np.std(vec[i, 0:hits])
                
        def MC(self, maxhits):
            vec=np.zeros((self.m_0.shape[0], maxhits))
            for i in range(0, maxhits):
                _, vec[:, i]=self.random_samples()
            return vec
                
                
            
        
    


    