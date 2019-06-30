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
#from .MonteCarlo_basics import AdaptiveRandomWalk
from .MonteCarlo_basics import Leapfrog
from .MonteCarlo_basics import HamiltonianMonteCarlo
from .MonteCarlo_basics import GaussianApproximation
from .MonteCarlo_basics import fixed_stepsize

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

class Settings(PDF):
    """Bayesian inverse problems with Tikhonov-like exponential
    """
#    __slots__ = ('mu', 'cov', 'prec')

    def __init__(self, op, rhs, prior, likelihood, solver, stopping_rule,
                  T, n_iter=None, stepsize_rule=None,  
                 n_steps=None, m_0=None, initial_stepsize=None):



        
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
        self.stepsize_rule=stepsize_rule or fixed_stepsize
        self.n_steps=n_steps or 10
        self.m_0=m_0 or np.zeros(self.initial_state.positions.shape[0])
        self.stepsize=initial_stepsize or 1e-1
#        self.sampler=sampler
#        if sampler == 'RandomWalk':
#            self.stepsize=initial_stepsize or 1e-1
#            self.sampler=RandomWalk(self, self.stepsize)
#        elif sampler == 'AdaptiveRandomWalk':
#            self.stepsize=initial_stepsize or 1e-1
#            self.sampler=AdaptiveRandomWalk(self, self.stepsize)
#        elif sampler == 'HamiltonianMonteCarlo':
#            self.stepsize=initial_stepsize or 1e-1
#            self.sampler=HamiltonianMonteCarlo(self, self.stepsize, self.n_steps)
#        elif sampler == 'GaussianApproximation':
#            self.sampler=GaussianApproximation(self)
#        else:
#            raise ValueError('sampler is not specified. Choose one of the following: AdaptiveRandomWalk, HamiltonianMonteCarlo, AdaptiveRandomWalk, GaussianApproximation')
        
        
    def run(self, sampler, statemanager, n_iter):
        for i in range(int(n_iter)):
            accepted = sampler.next()
#            print(sampler.stepsize)
            statemanager.statemanager(sampler.state, accepted)
            
            
        self.points = np.array([state.positions for state in statemanager.states])
        
        
        
        #accepted = [i for i in range(int(n_iter)) if statemanager.states[i]!=statemanager.states[i+1]]
        #print('acceptance_rate : {0:.1f} %'.format(100. * len(accepted) / n_iter))
        print('acceptance_rate : {0:.1f} %'.format(100. *statemanager.N/n_iter))
        if type(sampler.stepsize)==float:
            print('stepsize        : {0:.5f}'.format(sampler.stepsize))
        else:
            print('sampler.stepsize')
        
            
        if not False:
            self.reco = np.mean([s.positions for s in statemanager.states[-int(statemanager.N/2):]], axis=0)
            self.std = np.std([s.positions for s in statemanager.states[-int(statemanager.N/2):]], axis=0)
            self.reco_data=self.op(self.reco)
                
            
            

