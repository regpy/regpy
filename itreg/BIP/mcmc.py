import logging

import numpy as np
import scipy.optimize as scio

from . import PDF, State
from .MonteCarlo_basics import fixed_stepsize


class Settings(PDF):
    """Bayesian inverse problems with Tikhonov-like exponential
    """

    def __init__(
        self, setting, rhs, prior, likelihood, T, solver=None, stopping_rule=None,
        n_iter=None, stepsize_rule=None,
        n_steps=None, m_0=None, initial_stepsize=None, x_0=None
    ):
        self.setting = setting
        self.rhs = rhs
        self.prior = prior
        self.likelihood = likelihood
        self.T = T
        self.x_0 = x_0 or self.setting.op.domain.zeros()

        if solver is not None:
            self.solver = solver
        if stopping_rule is not None:
            self.stopping_rule = stopping_rule

        """The initial state is computed by the classical solver
        """

        self.initial_state = State()
        if 'solver' and 'stopping_rule' in dir(self):
            self.initial_state.positions, _ = self.solver.run(self.stopping_rule)
        else:
            res = scio.minimize(lambda x: -self.log_prob(x), self.x_0)
            self.initial_state.positions = res.x
        self.initial_state.log_prob = self.log_prob(self.initial_state.positions)
        self.first_state = self.initial_state.positions
        #        print(self.first_state)

        # parameters for Random Walk
        self.n_iter = n_iter or 2e4
        self.stepsize_rule = stepsize_rule or fixed_stepsize
        self.n_steps = n_steps or 10
        self.m_0 = m_0 or np.zeros(self.initial_state.positions.shape[0])
        self.stepsize = initial_stepsize or 1e-1

    def log_prob(self, x):
        return (self.prior.prior(x) + self.likelihood.likelihood(x)) / self.T

    def gradient(self, x):
        return (self.prior.gradient(x) + self.likelihood.gradient(x)) / self.T

    def run(self, sampler, statemanager):
        logging.info('Start MCMC')
        for i in range(int(self.n_iter)):
            accepted = sampler.next()
            #            print(sampler.stepsize)
            statemanager.statemanager(sampler.state, accepted)

        logging.info('MCMC finished')
        self.points = np.array([state.positions for state in statemanager.states])

        # accepted = [i for i in range(int(n_iter)) if statemanager.states[i]!=statemanager.states[i+1]]
        # print('acceptance_rate : {0:.1f} %'.format(100. * len(accepted) / n_iter))
        print('acceptance_rate : {0:.1f} %'.format(100. * statemanager.N / self.n_iter))
        if type(sampler.stepsize) == float:
            print('stepsize        : {0:.5f}'.format(sampler.stepsize))
        else:
            print('sampler.stepsize')

        self.reco = np.mean([s.positions for s in statemanager.states[-int(statemanager.N / 2):]], axis=0)
        self.std = np.std([s.positions for s in statemanager.states[-int(statemanager.N / 2):]], axis=0)
        self.reco_data = self.setting.op(self.reco)
