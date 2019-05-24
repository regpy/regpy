# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:01:42 2019

@author: Bjoern Mueller
"""

from __future__ import print_function
"""
Prototypes for important MCMC algorithms
"""




from . import Solver_BIP
from . import PDF
from . import State
from .MonteCarlo_basics import MetropolisHastings
from .MonteCarlo_basics import RandomWalk
from .MonteCarlo_basics import AdaptiveRandomWalk
from .MonteCarlo_basics import Leapfrog
from .MonteCarlo_basics import HamiltonianMonteCarlo
from .MonteCarlo_basics import GaussianApproximation

from itreg.solvers.landweber import Landweber


import logging
import numpy as np
import scipy.sparse.linalg as scsla
import scipy.optimize
import random as rd

import numpy as np
import matplotlib.pylab as plt

from scipy.special import logsumexp

from copy import deepcopy

class settings(PDF):
    """Bayesian inverse problems with Tikhonov-like exponential
    """
#    __slots__ = ('mu', 'cov', 'prec')

    def __init__(self, op, rhs, prior, likelihood, solver, stopping_rule,
                 sampler, T, n_iter=None, stepsize=None,  n_steps=None, m_0=None):



        
        self.op=op
        self.rhs=rhs
        self.prior=prior
        self.likelihood=likelihood
        self.T=T
        self.log_prob=(lambda x: (self.prior.prior(x)+self.likelihood.likelihood(x))/self.T)
        self.gradient=(lambda x: (self.prior.gradient(x)+self.likelihood.gradient(x))/self.T)
        
        self.solver=solver
        self.stopping_rule=stopping_rule
        
        """The initial state is computed by the classical solver
        """
        
        self.initial_state = State()
        self.initial_state.positions, _=self.solver.run(self.stopping_rule)
        self.initial_state.log_prob = self.log_prob(self.initial_state.positions)
        self.first_state=self.initial_state.positions
        
#parameters for Random Walk
        self.n_iter=n_iter or 2e4
        self.stepsize=stepsize or 5e-1
        self.n_steps=n_steps or 10
        self.m_0=m_0 or np.zeros(self.initial_state.positions.shape[0])
        if sampler == 'RandomWalk':
            self.sampler=RandomWalk(self, self.stepsize)
        elif sampler == 'AdaptiveRandomWalk':
            self.sampler=AdaptiveRandomWalk(self, self.stepsize)
        elif sampler == 'HamiltonianMonteCarlo':
            self.sampler=HamiltonianMonteCarlo(self, self.stepsize, self.n_steps)
        elif sampler == 'GaussianApproximation':
            self.sampler=GaussianApproximation(self)
        else:
            raise ValueError('sampler is not specified. Choose one of the following: AdaptiveRandomWalk, HamiltonianMonteCarlo, AdaptiveRandomWalk, GaussianApproximation')
        
        
        

        self.states=self.sampler.run(self.initial_state, self.n_iter)
        
        
        if sampler!='GaussianApproximation':
            accepted = [i for i in range(int(n_iter)) if self.states[i]!=self.states[i+1]]
            print('acceptance_rate : {0:.1f} %'.format(100. * len(accepted) / self.n_iter))
            print('stepsize        : {0:.1f}'.format(self.sampler.stepsize))
        
        self.points = np.array([state.positions for state in self.states])
        
        if not False:
            #N=self.initial_state.positions.shape[0]
            #self.reco=np.zeros(N)
            #for i in range(0, N):
            #    self.reco[i]=np.mean(self.points[int(self.n_iter/2): , i])
            #self.reco_data=self.op(self.reco)
            self.reco = np.mean([s.positions for s in self.states[-int(self.n_iter/2):]], axis=0)
            self.std = np.std([s.positions for s in self.states[-int(self.n_iter/2):]], axis=0)
            self.reco_data=self.op(self.reco)
            
            
            
            
            
            

#    def log_prob(self, state):
        
#        ## TODO: ignoring normalization constant for now

##        x = state.positions - self.mu
#        y=self.op(state.positions)-self.rhs

#        return - 0.5 * (self.op.range.inner(y, y)+self.regpar*self.op.domain.inner(state.positions, state.positions))

#    def gradient(self, state):
#        y, deriv=self.op.linearize(state.positions)
#        y-=self.rhs
#        return -self.op.domain.gram_inv(deriv.adjoint(self.op.range.gram(y))+self.regpar*state.positions)
    
    
        
        
        
        




















class Gaussian(PDF):
    
    """2-D-Gaussian
    """
    __slots__ = ('mu', 'cov', 'prec')
    
    def __init__(self, mu, cov_data, cov_sol, op, rhs, regpar):

        assert np.linalg.det(cov_data) > 0., \
               'Covariance matrix must be positive definite'
               
        assert np.linalg.det(cov_sol) > 0., \
               'Covariance matrix must be positive definite'
               
        self.mu=mu
        self.cov_data=cov_data
        self.cov_sol=cov_sol
        self.prec_data=np.linalg.inv(self.cov_data)
        self.prec_sol=np.linalg.inv(self.cov_sol)

        
        self.op=op
        self.rhs=rhs
        self.regpar=regpar
        
        if not False: 
            self.plotting()
    
    def plotting(self):
         ## plot results

        x = np.linspace(-1, 1, 100) * 3 * sigma[0]
        y = np.linspace(-1, 1, 100) * 3 * sigma[1]
    
        ## marginal distributions
    
        p_x = -0.5 * (x-mu[0])**2 / sigma[0]**2
        p_x-= logsumexp(p_x) + np.log(x[1]-x[0])
        p_y = -0.5 * (y-mu[1])**2 / sigma[1]**2
        p_y-= logsumexp(p_y) + np.log(y[1]-y[0])
        
        grid = np.reshape(np.meshgrid(x, y), (2, len(x)*len(y))).T
        log_prob = -0.5 * np.sum(grid*grid.dot(pdf.prec),1)
    
        burnin = int(0.2*n_iter)
        nbins = 31
        hist, xbins, ybins = np.histogram2d(*points[burnin:].T, bins=(nbins,nbins))
    
        fig, ax = plt.subplots(2,3,figsize=(9,6))
        ax = list(ax.flat)
    
        ax[0].contour(x, y, np.exp(log_prob.reshape(len(x),-1)))
        ax[0].scatter(*mu, color='r', s=200)
        ax[0].scatter(*points.T, s=10, c=np.linspace(0.,1.,len(points)), alpha=0.7)
    
        ax[1].contour(x, y, np.exp(log_prob.reshape(len(x),-1)))
        ax[1].scatter(*mu, color='r', s=200)
        ax[1].scatter(*points[burnin:].T, s=10, c=np.linspace(0.,1.,len(points)-burnin), alpha=0.7)
    
        ax[2].matshow(hist, origin='lower', extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]))
        ax[2].contour(x, y, np.exp(log_prob.reshape(len(x),-1)), cmap=plt.cm.gray_r, alpha=0.5)
        ax[2].set_aspect(1./ax[2].get_data_ratio())
    
        ax[3].hist(points[burnin:,0], bins=31, normed=True, color='k', alpha=0.5)
        ax[3].plot(x, np.exp(p_x), color='r', lw=3)
    
        ax[4].hist(points[burnin:,1], bins=31, normed=True, color='k', alpha=0.5)
        ax[4].plot(y, np.exp(p_y), color='r', lw=3)
    
        ax[5].plot([state.log_prob for state in states])
    
        fig.tight_layout()
