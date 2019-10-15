from collections import deque
from copy import copy
from itertools import islice

import numpy as np

from regpy.util import classlogger


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
            self.logprob = logpdf(self.pos)


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
        state = self._propose(self.state).complete(self.logpdf)
        accepted = (np.log(np.random.rand()) < state.logprob - self.state.logprob)
        self._update(state, accepted)
        return state, accepted

    def _update(self, state, accepted):
        if accepted:
            self.state = state

    def _propose(self, state):
        raise NotImplementedError

    def __iter__(self):
        while True:
            yield self.next()

    def run(self, niter, callback=None):
        # TODO some convenience logging
        for state, accepted in islice(self, int(niter)):
            if callback is not None:
                callback(state, accepted)


class RandomWalk(MetropolisHastings):
    def __init__(self, logpdf, stepsize, state=None, stepsize_rule=None):
        super().__init__(logpdf, state=state)
        self.stepsize = float(stepsize)
        self.stepsize_rule = stepsize_rule

    def _propose(self, state):
        return self.State(
            pos=state.pos + self.stepsize * self.logpdf.domain.randn()
        ).complete(self.logpdf)

    def _update(self, state, accepted):
        if self.stepsize_rule is not None:
            self.stepsize = self.stepsize_rule(self.stepsize, self.state, state, accepted)
        super()._update(state, accepted)


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


class StateHistory:
    def __init__(self, maxlen=None):
        self.states = deque(maxlen=int(maxlen))
        self.accepted = 0
        self.rejected = 0

    def add(self, state, accepted):
        if accepted:
            self.accepted += 1
            self.states.append(state)
        else:
            self.rejected += 1

    @property
    def total(self):
        return self.rejected + self.accepted

    @property
    def acceptance_rate(self):
        return self.accepted / self.total

    def samples(self):
        # TODO returning the entire array is a bad idea once we implement histroy managers
        #      that store states on disk
        return np.array([s.pos for s in self.states])

    def logprobs(self):
        return np.array([s.logprob for s in self.states])

# TODO fix this
#
# from regpy.util.svd import randomized_svd
#
# class GaussianApproximation(object):
#     def __init__(self, pdf):
#         # find m_MAP by Maximum-Likelihood
#         # TODO: Insert one of the implemented solvers instead of scipy.optimize.minimize
#         # Is done in mcmc_second_variant.
#         # Insert approximated code to compute gamma_prior_half^{1/2}
#
#         self.pdf = pdf
#         self.stepsize = 'randomly chosen'
#         self.y_MAP = self.pdf.setting.op(self.pdf.initial_state.positions)
#         N = self.pdf.initial_state.positions.shape[0]
#         # define the prior-preconditioned Hessian
#         self.Hessian_prior = np.zeros((N, N))
#         self.gamma_prior_inv = np.zeros((N, N))
#         for i in range(0, N):
#             self.gamma_prior_inv[:, i] = self.pdf.prior.hessian(self.pdf.m_0, np.eye(N)[:, i])
#         D, S = np.linalg.eig(-np.linalg.inv(self.gamma_prior_inv))
#         D = D.real
#         S = S.real
#
#         self.gamma_prior_half = np.dot(S.transpose(), np.dot(np.diag(np.sqrt(D)), S))
#         #
#         for i in range(0, N):
#             self.Hessian_prior[:, i] = np.dot(
#                 self.gamma_prior_half,
#                 self.pdf.likelihood.hessian(self.pdf.m_0, np.dot(self.gamma_prior_half, np.eye(N)[:, i]))
#             )
#         # construct randomized SVD of Hessian_prior
#         self.L, self.V = randomized_svd(self, self.Hessian_prior)
#         self.L = -self.L
#         # define gamma_post
#         self.gamma_post = np.dot(
#             self.gamma_prior_half,
#             np.dot(
#                 self.V,
#                 np.dot(np.diag(1 / (self.L + 1)), np.dot(self.V.transpose(), self.gamma_prior_half))
#             )
#         )
#         self.gamma_post_half = np.dot(
#             self.gamma_prior_half,
#             np.dot(self.V, np.dot(np.diag(1 / np.sqrt(self.L + 1) - 1), self.V.transpose())) +
#             np.eye(self.gamma_prior_half.shape[0])
#         )
#
#     def random_samples(self):
#         R = np.random.normal(0, 1, self.gamma_post_half.shape[0])
#         m_post = self.pdf.initial_state.positions + np.dot(self.gamma_post_half, R)
#         return m_post
#
#     def next(self):
#         m_post = self.random_samples()
#         next_state = State()
#         next_state.positions = m_post
#         next_state.log_prob = np.exp(-np.dot(m_post, np.dot(self.gamma_post, m_post)))
#         self.state = next_state
#         return True
