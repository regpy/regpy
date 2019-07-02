# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:03:29 2019

@author: Bjoern Mueller
"""

"""Änderungen in line 75, 109, 145-146, 156-157
"""
from itreg.util import classlogger
import numpy as np

from . import State
from . import HMCState
from . import PDF

from itreg.BIP.utils.SVD_methods import Lanzcos_SVD
from itreg.BIP.utils.SVD_methods import randomized_SVD

from copy import deepcopy


def fixed_stepsize(stepsize, current_state, proposed_state, accepted):
    return stepsize

def adaptive_stepsize(stepsize, current_state, proposed_state, accepted, stepsize_factor):
    stepsize *= stepsize_factor if accepted else 1 / stepsize_factor
    return stepsize



class MetropolisHastings(object):
    log=classlogger
    """MetropolisHastings for *symmetric* proposal kernel
    """
    def __init__(self, pdf, statemanager, initial_stepsize=1, stepsize_rule=fixed_stepsize):
        assert isinstance(pdf, PDF), 'Instance of PDF expected'
        self.pdf = pdf
        self.stepsize=initial_stepsize
        self.stepsize_rule=stepsize_rule
        self.state=statemanager.initial_state
        
    def propose(self, current_state):
        raise NotImplementedError

    def accept(self, current_state, proposed_state):

#        assert isinstance(current_state, State), 'State expected'
#        assert isinstance(proposed_state, State), 'State expected'

        log_odds = proposed_state.log_prob - \
                   current_state.log_prob
                   
        accepted=np.log(np.random.random()) < log_odds
#        print(accepted)
        self.stepsize=self.stepsize_rule(self.stepsize, current_state, proposed_state, accepted)
        #print(accepted)

        return np.log(np.random.random()) < log_odds
    
    def next(self):
        proposed_state = self.propose(self.state)
        accept = self.accept(self.state, proposed_state)
        if accept:
            self.state = proposed_state
        return accept
    
    


class statemanager(object):
    """Describes what to do with the states
    """
    def __init__(self, initial_state, parameter_list=None):
        if parameter_list is not None:

            if 'log_prob' not in parameter_list:
                parameter_list.append('log_prob')
            if 'positions' not in parameter_list:
                parameter_list.append('positions')
            class state(object):
                __slots__ = tuple(parameter_list)
            self.initial_state=state
            self.initial_state.log_prob=initial_state.log_prob
            self.initial_state.positions=initial_state.positions
        else:
            self.initial_state=initial_state
        self.states=[self.initial_state]
        self.N=1
        
    def statemanager(self, state, accepted):
#        assert isinstance(self.initial_state, State), 'State expected'
        if accepted:
            self.states.append(state)
            self.N+=1
    
class RandomWalk(MetropolisHastings):
    """Gaussian proposal kernel
    """
    def __init__(self, pdf, statemanager, stepsize=1e-1, stepsize_rule=fixed_stepsize):
        assert stepsize > 0., 'Positive number expected'
        super(RandomWalk, self).__init__(pdf, statemanager)
        self.stepsize = float(stepsize)
        self.stepsize_rule=stepsize_rule

    def propose(self, current_state):

        proposed_state = deepcopy(current_state)
        random_step = np.random.standard_normal(proposed_state.positions.shape)
        proposed_state.positions += self.stepsize * random_step
        #proposed_state.log_prob = self.pdf.log_prob(proposed_state)
        proposed_state.log_prob = self.pdf.log_prob(proposed_state.positions)
#        print(current_state.log_prob)

        return proposed_state




class Leapfrog(object):
    """Leapfrog integrator
    """
    def __init__(self, pdf, stepsize=1e-5, n_steps=10):

        assert stepsize > 0
        assert n_steps  > 0
        
        self.pdf      = pdf
        self.stepsize = float(stepsize)
        self.n_steps  = int(n_steps)

    def run(self, positions, momenta):

        def dH_q(q, p):
            state = State()
            state.positions = q
            return -self.pdf.gradient(state.positions)
            #return -self.pdf.gradient(state)

        def dH_p(q, p):
            return p

        q  = positions.copy()
        p  = momenta.copy()
        dt = self.stepsize
        
        ## half-step

        p -= 0.5 * dt * dH_q(q, p)

        for t in range(self.n_steps-1):

            q += dt * dH_p(q, p)
            p -= dt * dH_q(q, p)

        q += dt * dH_p(q, p)
        p -= 0.5 * dt * dH_q(q, p)

        return q, p

class HamiltonianMonteCarlo(RandomWalk):

    def __init__(self, pdf, statemanager, stepsize=1e-5, stepsize_rule=fixed_stepsize, n_steps=10):

        super(HamiltonianMonteCarlo, self).__init__(pdf, statemanager, stepsize, stepsize_rule)

        self.integrator = Leapfrog(self.pdf, self.stepsize, n_steps)

    def propose(self, current_state):

        current_state.momenta  = np.random.standard_normal(current_state.positions.shape)
        current_state.log_prob = -0.5 * np.sum(current_state.momenta**2) + \
                                 self.pdf.log_prob(current_state.positions)
        #current_state.log_prob = -0.5 * np.sum(current_state.momenta**2) + \
        #                         self.pdf.log_prob(current_state)

        proposed_state = deepcopy(current_state)

        q, p = self.integrator.run(proposed_state.positions, proposed_state.momenta)

        proposed_state.positions, proposed_state.momenta = q, p

        proposed_state.log_prob = -0.5 * np.sum(proposed_state.momenta**2) + \
                                  self.pdf.log_prob(proposed_state.positions)
#        print(proposed_state.log_prob)
        #proposed_state.log_prob = -0.5 * np.sum(proposed_state.momenta**2) + \
        #                          self.pdf.log_prob(proposed_state)

        return proposed_state


    
    
class GaussianApproximation(object):    
    def __init__(self, pdf):
        
#find m_MAP by Maximum-Likelihood
        """TODO: Insert one of the implemented solvers instead of scipy.optimize.minimize
        Is done in mcmc_second_variant.
        Insert approximated code to compute gamma_prior_half^{1/2}
        """
        self.pdf=pdf
        self.stepsize='randomly chosen'
        self.y_MAP=self.pdf.op(self.pdf.initial_state.positions)
        N=self.pdf.initial_state.positions.shape[0]
#define the prior-preconditioned Hessian
        self.Hessian_prior=np.zeros((N, N))
        self.gamma_prior_inv=np.zeros((N, N))
        for i in range(0, N):
            self.gamma_prior_inv[:, i]=self.pdf.prior.hessian(self.pdf.m_0, np.eye(N)[:, i])
        D, S=np.linalg.eig(np.linalg.inv(self.gamma_prior_inv))
        self.gamma_prior_half=np.dot(S.transpose(), np.dot(np.diag(np.sqrt(D)), S))
        for i in range(0, N):
            self.Hessian_prior[:, i]=np.dot(self.gamma_prior_half, self.pdf.likelihood.hessian(self.pdf.m_0, np.dot(self.gamma_prior_half, np.eye(N)[:, i])))
#construct randomized SVD of Hessian_prior      
        self.L, self.V=randomized_SVD(self, self.Hessian_prior)
#define gamma_post
        self.gamma_post=np.dot(self.gamma_prior_half, np.dot(self.V, np.dot(np.diag(1/(self.L+1)), np.dot(self.V.transpose(), self.gamma_prior_half))))  
        self.gamma_post_half=np.dot(self.gamma_prior_half, (np.dot(self.V, np.dot(np.diag(1/np.sqrt(self.L+1)-1), self.V.transpose()))+np.eye(self.pdf.prior.gamma_prior.shape[0])))
#define prior, posterior sampling
        

        

        
    def random_samples(self):
        R=np.random.normal(0, 1, self.gamma_post_half.shape[0]) 
#        m_prior=self.pdf.m_0+np.dot(self.gamma_prior_half, R)
        m_post=self.pdf.initial_state.positions+np.dot(self.gamma_post_half, R)
        return  m_post
        
#    def run(self, initial_state, n_iter):
#        states = [initial_state]

#        for i in range(0, int(n_iter)):
#            _, m_post=self.evaluation.random_samples()
#            current_state=State()
#            current_state.positions=m_post
#            current_state.log_prob=self.pdf.log_prob(m_post)
#            states.append(current_state)

#        return states
    
    def next(self):
        m_post=self.random_samples()
        next_state=State()
        next_state.positions=m_post
        next_state.log_prob=np.exp(-np.dot(m_post, np.dot(self.gamma_post, m_post)))
        self.state=next_state
        
    
    
    
        