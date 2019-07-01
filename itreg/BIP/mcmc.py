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

    def __init__(self, setting, rhs, prior, likelihood, solver, stopping_rule,
                  T, n_iter=None, stepsize_rule=None,  
                 n_steps=None, m_0=None, initial_stepsize=None):



        
        self.setting=setting
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
        
        
    def run(self, sampler, statemanager):
        printed=[]
        for i in range(int(self.n_iter)):
            accepted = sampler.next()
#            print(sampler.stepsize)
            statemanager.statemanager(sampler.state, accepted)
            printed.append(100*round(i/self.n_iter, 2))
            
            
        print('\n'.join(map(str, printed)))    
        self.points = np.array([state.positions for state in statemanager.states])
        
        
        
        #accepted = [i for i in range(int(n_iter)) if statemanager.states[i]!=statemanager.states[i+1]]
        #print('acceptance_rate : {0:.1f} %'.format(100. * len(accepted) / n_iter))
        print('acceptance_rate : {0:.1f} %'.format(100. *statemanager.N/self.n_iter))
        if type(sampler.stepsize)==float:
            print('stepsize        : {0:.5f}'.format(sampler.stepsize))
        else:
            print('sampler.stepsize')
        
            
        if not False:
            self.reco = np.mean([s.positions for s in statemanager.states[-int(statemanager.N/2):]], axis=0)
            self.std = np.std([s.positions for s in statemanager.states[-int(statemanager.N/2):]], axis=0)
            self.reco_data=self.setting.op(self.reco)
            
            
            
def plot_lastiter(self, exact_solution, exact_data, data):
    plt.plot(self.op.params.domain.coords, exact_solution.T, label='exact solution')
    plt.plot(self.op.params.domain.coords, self.reco, label='reco')
    plt.title('solution')
    plt.legend()
    plt.show()
    
    plt.plot(self.op.params.range.coords, exact_data, label='exact data')
    plt.plot(self.op.params.range.coords, data, label='data')
    plt.plot(self.op.params.range.coords, self.reco_data, label='reco data')
    plt.legend()
    plt.title('data')
    plt.show()
    
def plot_mean(statemanager, exact_solution, n_list=None, n_iter=None, variance=None):
    variance=variance or np.array([3])
#    print(variance[0])
#    if type(variance) is int or float:
#        variance=np.array([variance])
    n_plots=np.size(variance)
    if n_list is None and n_iter is None:
        raise ValueError('Specify the evaluation points')
    if n_list is not None:
        assert n_list.max()<statemanager.N
        a = np.array([s.positions for s in statemanager.states[n_list]])        
    else:
        assert n_iter<statemanager.N
        a = np.array([s.positions for s in statemanager.states[-n_iter:]])
    for i in range(0, n_plots):
        v = a.std(axis=0)*variance[i]
        m = a.mean(axis=0)
        plt.plot(np.array([m+v]).T, label='mean +' +str(variance[i])+ '*variance')
        plt.plot(np.array([m-v]).T, label='mean- '+str(variance[i])+'*variance')
    plt.plot(exact_solution.T, label='exact solution')
    plt.legend()
    plt.show()
    
def plot_verlauf(statemanager, pdf=None, exact_solution=None, plot_real=False):
    arr=[s.log_prob for s in statemanager.states]
    maximum=np.asarray([arr]).max()
    plt.plot(range(0, statemanager.N), arr/maximum, label='iterated log_prob')
    if plot_real is True:
        if pdf is None or exact_solution is None:
            raise ValueError('Specify the log_prob of exact solution')
        plt.plot(range(0, statemanager.N), pdf.log_prob(exact_solution)*np.ones(statemanager.N)/maximum, label='exact solution')
    plt.xlabel('iterations')
    plt.ylabel('log probability')
    plt.legend()
    plt.show()
    
def plot_iter(pdf, statemanager, position):  
    assert type(position)==int
    plt.plot(range(0, statemanager.N), [s.positions[position] for s in statemanager.states])
    plt.xlabel('iterations')
    plt.ylabel('value at x='+str(round(pdf.op.params.domain.coords[position], 2)))
    plt.show()
    
                
            
if False:

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

    ax[3].hist(points[burnin:,0], bins=31, density=True, color='k', alpha=0.5)
    ax[3].plot(x, np.exp(p_x), color='r', lw=3)

    ax[4].hist(points[burnin:,1], bins=31, density=True, color='k', alpha=0.5)
    ax[4].plot(y, np.exp(p_y), color='r', lw=3)

    ax[5].plot([state.log_prob for state in states])

    fig.tight_layout()
    plt.show()            

