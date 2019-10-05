import setpath

import logging
from functools import partial

import numpy as np

from itreg.BIP.mcmc import RandomWalk, StateHistory, adaptive_stepsize
from itreg.functionals import tikhonov_functional
from itreg.operators import Volterra
from itreg.solvers import HilbertSpaceSetting
from itreg.spaces import L2, Sobolev, UniformGrid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

grid = UniformGrid((0, 2 * np.pi, 200))
op = Volterra(grid)

exact_solution = np.sin(grid.coords[0])
exact_data = op(exact_solution)
noise = 0.03 * op.domain.randn()
data = exact_data + noise

# werte für adaptive randomwalk, L2-regularization:
# n_iter=2e4
# stepsize=1e-2
# temperature=0.001
# regpar=0.1

temperature = 1e-3
prior, likelihood = tikhonov_functional(
    setting=HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=L2),
    data=data,
    regpar=1e-2
)
logpdf = (likelihood + prior) / temperature

init = op.domain.ones()

sampler = RandomWalk(
    logpdf=logpdf,
    stepsize=1.5,
    state=RandomWalk.State(pos=init),
    stepsize_rule=partial(adaptive_stepsize, stepsize_factor=1.05)
)

hist = StateHistory(maxlen=1e4)
sampler.run(niter=1e3, callback=hist)

samples = hist.samples()
logprobs = hist.logprobs()

mean = np.mean(samples, axis=0)
std = np.std(samples, axis=0)

# from itreg.BIP.plot_functions import plot_lastiter
# from itreg.BIP.plot_functions import plot_mean
# from itreg.BIP.plot_functions import plot_verlauf
# from itreg.BIP.plot_functions import plot_iter
#
# plot_lastiter(bip, exact_solution, exact_data, data)
# plot_mean(statemanager, exact_solution, n_iter=statemanager.N - 1, variance=np.asarray([1]))
# plot_verlauf(statemanager, pdf=bip, exact_solution=exact_solution, plot_real=True)
# plot_iter(bip, statemanager, 10)
