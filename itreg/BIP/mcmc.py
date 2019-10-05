import logging
from copy import copy

import numpy as np
import scipy.optimize as scio

from itreg.util import classlogger


class Settings:
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
        self.stepsize_rule = stepsize_rule
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


class State:
    __slots__ = 'pos', 'logprob'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def complete(self, logpdf):
        result = copy(self)
        result._complete(logpdf)
        return result

    def update(self, **kwargs):
        result = copy(self)
        for k, v in kwargs.items():
            setattr(result, k, v)
        return result

    def _complete(self, logpdf):
        if not hasattr(self, 'pos'):
            self.pos = logpdf.domain.zeros()
        if not hasattr(self, 'logprob'):
            self.logprob = logpdf.linearize(self.pos)


class MetropolisHastings:
    State = State
    log = classlogger

    def __init__(self, logpdf, state=None):
        self.logpdf = logpdf
        if state is not None:
            self.state = state.complete(logpdf)
        else:
            self.state = self.State().complete(logpdf)

    def next(self):
        proposed = self._propose(self.state).complete(self.logpdf)
        accepted = (np.log(np.random.rand()) < proposed.logprop - self.state.logprob)
        self._update(proposed, accepted)
        return proposed, accepted

    def _update(self, proposed, accepted):
        if accepted:
            self.state = proposed

    def _propose(self, state):
        raise NotImplementedError

    def __iter__(self):
        while True:
            yield self.next()


class RandomWalk(MetropolisHastings):
    def __init__(self, logpdf, stepsize, state=None, stepsize_rule=None):
        super().__init__(logpdf, state=state)
        self.stepsize = float(stepsize)
        self.stepsize_rule = stepsize_rule

    def _propose(self, state):
        return self.State(
            pos=state.pos + self.stepsize * self.logpdf.domain.randn()
        ).complete(self.logpdf)

    def _update(self, proposed, accepted):
        if self.stepsize_rule is not None:
            self.stepsize = self.stepsize_rule(self.stepsize, self.state, proposed, accepted)
        super()._update(proposed, accepted)


def fixed_stepsize(stepsize, state, proposed, accepted):
    return stepsize


def adaptive_stepsize(stepsize, state, proposed, accepted, stepsize_factor):
    stepsize *= stepsize_factor if accepted else 1 / stepsize_factor
    return stepsize


class HamiltonState(State):
    __slots__ = 'momenta', 'grad'

    def _complete(self, logpdf):
        if not hasattr(self, 'pos'):
            self.pos = logpdf.domain.zeros()
        if not hasattr(self, 'momenta'):
            self.momenta = logpdf.domain.zeros()
        if not hasattr(self, 'logprob'):
            # We just assume grad is not set either for simplicity.
            self.logprob, self.grad = logpdf.linearize(self.pos)
        elif not hasattr(self, 'grad'):
            self.grad = logpdf.gradient(self.pos)


class HamiltonianMonteCarlo(RandomWalk):
    State = HamiltonState

    def __init__(self, logpdf, stepsize, state=None, stepsize_rule=None, integrator=None):
        super().__init__(logpdf, stepsize, state=state, stepsize_rule=stepsize_rule)
        if integrator is not None:
            self.integrator = integrator
        else:
            self.integrator = leapfrog

    def _propose(self, state):
        return self.integrator(
            logpdf=self.logpdf,
            state=state.update(
                momenta=self.logpdf.domain.randn()
            ),
            stepsize=self.stepsize,
        )


def leapfrog(logpdf, state, stepsize, nsteps=10):
    state = state.complete(logpdf)
    pos = state.pos.copy()
    momenta = state.momenta.copy()

    momenta += 0.5 * stepsize * state.grad

    for _ in range(nsteps):
        pos += stepsize * momenta
        momenta += stepsize * logpdf.gradient(pos)

    pos += stepsize * momenta
    logprob, grad = logpdf.linearize(pos)
    momenta += 0.5 * stepsize * grad

    return HamiltonState(pos=pos, logprob=logprob, momenta=momenta, grad=grad)
