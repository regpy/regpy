# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:02:45 2019

@author: Hendrik Müller
"""

from . import Solver_BIP
from itreg.BIP.utils.BIP_utils import prior_distribution
from itreg.BIP.utils.BIP_utils import likelihood_distribution
from itreg.BIP.utils.BIP_utils import Monte_Carlo_evaluation


import logging
import numpy as np
import scipy.sparse.linalg as scsla
import scipy.optimize
import random as rd
import matplotlib.pyplot as plt

class BayesianIP(Solver_BIP):
   



    def __init__(self, op, prior_type, likelihood_type, rhs, m_0, gamma_prior=None, gamma_d=None, l1_A=None, l1_sigma=None, x_upper=None, x_lower=None, norm_A=None, norm_sigma=None, maxnum=None, maxhits=None, stepsize=None):
        super().__init__()
        self.op = op
        self.rhs = rhs
        self.m_0=m_0
        self.gamma_prior=gamma_prior
        self.gamma_d=gamma_d
        self.l1_A=l1_A
        self.l1_sigma=l1_sigma
        self.x_upper=x_upper
        self.x_lower=x_lower
        self.norm_A=norm_A
        self.norm_sigma=norm_sigma
        self.maxnum=maxnum or 10**5
        self.maxhits=maxhits or 1000
        self.prior=prior_distribution(op, prior_type, m_0, gamma_prior, l1_A, l1_sigma, x_upper, x_lower)
#        self.prior=prior_distribution(op, prior_type, gamma_prior=None, l1_A=None, l1_sigma=None, x_upper=None, x_lower=None)
#        self.prior=prior_distribution(self, op, prior_type)
        self.likelihood=likelihood_distribution(op, likelihood_type, rhs, gamma_d, l1_A, l1_sigma, norm_A, norm_sigma)
#        def distribution(self, x):
#            return  self.prior.prior(x)+self.likelihood.likelihood(x)
#        self.distribution=distribution
        self.distribution=(lambda x: self.prior.prior(x)+self.likelihood.likelihood(x))
        

       
#here the step size is the percentage number we reduce the prior covariance in each step.
        self.stepsize = stepsize or 1.2


#find m_MAP by Maximum-Likelihood
        res=scipy.optimize.minimize((lambda x: -self.distribution(x)), self.m_0)
        self.m_MAP=res.x
        self.x=self.m_MAP
        self.y=self.op(self.x)
        N=self.m_0.shape[0]
#define the prior-preconditioned Hessian
        self.Hessian_prior=np.zeros((N, N))
        self.gamma_prior_inv=np.zeros((N, N))
        for i in range(0, N):
            self.gamma_prior_inv[:, i]=self.prior.hessian(self.m_0, np.eye(N)[:, i])
        D, S=np.linalg.eig(np.linalg.inv(self.gamma_prior_inv))
#        print(D)
        self.gamma_prior_half=np.dot(S.transpose(), np.dot(np.diag(np.sqrt(D)), S))
        for i in range(0, N):
            self.Hessian_prior[:, i]=np.dot(self.gamma_prior_half, self.likelihood.hessian(self.m_0, np.dot(self.gamma_prior_half, np.eye(N)[:, i])))
#construct randomized SVD of Hessian_prior      
#        print(self.Hessian_prior)
        self.L, self.V=randomized_SVD(self, self.Hessian_prior)
#define gamma_post
        self.gamma_post=np.dot(self.gamma_prior_half, np.dot(self.V, np.dot(np.diag(1/(self.L+1)), np.dot(self.V.transpose(), self.gamma_prior_half))))  
        self.gamma_post_half=np.dot(self.gamma_prior_half, (np.dot(self.V, np.dot(np.diag(1/np.sqrt(self.L+1)-1), self.V.transpose()))+np.eye(self.gamma_prior.shape[0])))
#define prior, posterior sampling
        self.evaluation=Monte_Carlo_evaluation(op, self.distribution, self.gamma_post_half, self.m_0, self.m_MAP,  maxnum=self.maxnum, maxhits=self.maxhits, gamma_prior_half=self.gamma_prior_half)
        self.m_prior, self.m_post=self.evaluation.random_samples()
        
        
        
        
        
    def plotting(self, number):
        vec=self.MC(maxhits=number)
        for i in range(0, number):
            plt.plot(vec[:, i])
        
        
        
        
        
        
        

        
        
        
        






def randomized_SVD(self, H):
    r=np.linalg.matrix_rank(H)
    N=self.gamma_prior_half.shape[0]
#    r=N
    X=np.zeros((N, r))
    for i in range(0, N):
        for j in range(0, r):
            X[i,j]=rd.random()
        
    #second step, compute sample matrix
    Y=np.dot(H, X)
    
    #third step, QR-decomposition of Y
    Q, R=np.linalg.qr(Y)
    
    #solve linear system to obtain the matrix B: Q^TY=B(Q^T X)
    B_trans=np.linalg.solve(np.dot(X.transpose(), Q), np.dot(Y.transpose(), Q))
    B=B_trans.transpose()
    
    #Perform SVD of B
    L, U=np.linalg.eig(B)
    #compute singular vectors
    V=np.dot(Q, U)
    #L are the singular values and V the corresponding singular vectors.
    return L, V

