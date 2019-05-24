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

from itreg.BIP.utils.BIP_utils import Monte_Carlo_evaluation
from itreg.BIP.utils.SVD_methods import Lanzcos_SVD
from itreg.BIP.utils.SVD_methods import randomized_SVD

from copy import deepcopy

class MetropolisHastings(object):
    log=classlogger
    """MetropolisHastings for *symmetric* proposal kernel
    """
    def __init__(self, pdf):
        assert isinstance(pdf, PDF), 'Instance of PDF expected'
        self.pdf = pdf
        
    def propose(self, current_state):
        raise NotImplementedError

    def accept(self, current_state, proposed_state):

        assert isinstance(current_state, State), 'State expected'
        assert isinstance(proposed_state, State), 'State expected'

        log_odds = proposed_state.log_prob - \
                   current_state.log_prob

        return np.log(np.random.random()) < log_odds

    def run(self, initial_state, n_iter):

        assert isinstance(initial_state, State), 'State expected'
        assert n_iter > 0, 'Positive number expected'

        states = [initial_state]

        current_state = initial_state

        for i in range(int(n_iter)):
            print(i)

            proposed_state = self.propose(current_state)

            do_accept = self.accept(current_state, proposed_state)

            if do_accept:
                current_state = proposed_state

            states.append(current_state)

        return states
    
class RandomWalk(MetropolisHastings):
    """Gaussian proposal kernel
    """
    def __init__(self, pdf, stepsize=1e-1):
        assert stepsize > 0., 'Positive number expected'
        super(RandomWalk, self).__init__(pdf)
        self.stepsize = float(stepsize)

    def propose(self, current_state):

        proposed_state = deepcopy(current_state)
        random_step = np.random.standard_normal(proposed_state.positions.shape)
        proposed_state.positions += self.stepsize * random_step
        #proposed_state.log_prob = self.pdf.log_prob(proposed_state)
        proposed_state.log_prob = self.pdf.log_prob(proposed_state.positions)

        return proposed_state

class AdaptiveRandomWalk(RandomWalk):
    """RandomWalk Metropolis-Hastings with an adaptive stepsize
    """
    stepsize_factor = 1.05

    def accept(self, current_state, proposed_state):
        do_accept = super(AdaptiveRandomWalk, self).accept(current_state, proposed_state)
        self.stepsize *= self.stepsize_factor if do_accept else \
                         1. / self.stepsize_factor
        return do_accept

class Leapfrog(object):
    """Leapfrog integrator
    """
    def __init__(self, pdf, stepsize=1e-1, n_steps=10):

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

    def __init__(self, pdf, stepsize=1e-1, n_steps=10):

        super(HamiltonianMonteCarlo, self).__init__(pdf, stepsize)

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
        #proposed_state.log_prob = -0.5 * np.sum(proposed_state.momenta**2) + \
        #                          self.pdf.log_prob(proposed_state)

        return proposed_state

    def run(self, initial_state, n_iter=1e3):

        current_state = HMCState()
        current_state.positions = initial_state.positions

        return super(HamiltonianMonteCarlo, self).run(current_state, n_iter)
    
    
    
class GaussianApproximation(object):    
    def __init__(self, pdf):
        
#find m_MAP by Maximum-Likelihood
        """TODO: Insert one of the implemented solvers instead of scipy.optimize.minimize
        Is done in mcmc_second_variant.
        Insert approximated code to compute gamma_prior_half^{1/2}
        """
        self.pdf=pdf
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
        self.evaluation=Monte_Carlo_evaluation(self.pdf.op, self.pdf.log_prob, self.gamma_post_half, self.pdf.initial_state.positions, self.pdf.initial_state.positions,  maxnum=self.pdf.n_iter, gamma_prior_half=self.gamma_prior_half)
#        self.m_prior, self.m_post=self.evaluation.random_samples()
        
    def run(self, initial_state, n_iter):
        states = [initial_state]

        for i in range(0, int(n_iter)):
            _, m_post=self.evaluation.random_samples()
            current_state=State()
            current_state.positions=m_post
            current_state.log_prob=self.pdf.log_prob(m_post)
            states.append(current_state)

        return states
    
        