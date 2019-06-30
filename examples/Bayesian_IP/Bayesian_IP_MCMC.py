import setpath

import itreg

from itreg.operators import NonlinearVolterra
from itreg.spaces import L2, HilbertPullBack, UniformGrid
from itreg.solvers import Landweber, HilbertSpaceSetting
#from itreg.util import test_adjoint
import itreg.stoprules as rules

#from itreg.BIP.mcmc import tikhonov_like
from itreg.BIP.mcmc import Settings
from itreg.BIP.prior_distribution.prior_distribution import gaussian as gaussian_prior
from itreg.BIP.likelihood_distribution.likelihood_distribution import gaussian as gaussian_likelihood

from itreg.BIP.MonteCarlo_basics import fixed_stepsize
from itreg.BIP.MonteCarlo_basics import adaptive_stepsize
from itreg.BIP.MonteCarlo_basics import statemanager
from itreg.BIP.MonteCarlo_basics import RandomWalk
#from itreg.BIP.MonteCarlo_basics import AdaptiveRandomWalk
from itreg.BIP.MonteCarlo_basics import HamiltonianMonteCarlo
from itreg.BIP.MonteCarlo_basics import GaussianApproximation

from itreg.BIP.prior_distribution.prior_distribution import l1 as l1_prior
from itreg.BIP.likelihood_distribution.likelihood_distribution import l1 as l1_likelihood

import numpy as np
import logging
import matplotlib.pyplot as plt
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

spacing=2*np.pi/200
xs=np.linspace(0, 200, 200)/200*2*np.pi
grid=UniformGrid(xs)

op = NonlinearVolterra(grid, exponent=3)

exact_solution = np.sin(xs)
exact_data = op(exact_solution)
noise = 0.03 * op.domain.rand(np.random.randn)
data = exact_data + noise

#noiselevel = op.codomain.norm(noise)

init = op.domain.ones()

setting = HilbertSpaceSetting(op=op, domain=L2, codomain=L2)

solver = Landweber(setting, data, init, stepsize=0.01)
stopping_rule = (
    rules.CountIterations(10) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=setting.codomain.norm(noise), tau=1.1))

#prior=gaussian_prior(np.eye(200), op, exact_solution+0.1*np.ones(exact_solution.shape[0]))
#likelihood=gaussian_likelihood(op, np.eye(200), exact_data+0.1*np.ones(exact_data.shape[0]))
prior=gaussian_prior(0.1*np.eye(200), setting, np.zeros(200))
likelihood=gaussian_likelihood(setting, np.eye(200), exact_data)
#prior=l1_prior(1, op)
#likelihood=l1_likelihood(op, 1, exact_data)

## run random walk mcmc

n_iter   = 2e4
stepsize = [1e-2, 1e-1, 5e-1, 7e-1, 1e0, 1.2, 1.5, 2.5][0]



#sampler=['RandomWalk', 'AdaptiveRandomWalk', 'HamiltonianMonteCarlo', 'GaussianApproximation'][0]

    
stepsize_rule=partial(adaptive_stepsize, stepsize_factor=1.05)
#stepsize_rule=fixed_stepsize

bip=Settings(setting, data, prior, likelihood, solver, stopping_rule, 0.001, 
              n_iter=n_iter, stepsize_rule=stepsize_rule)

statemanager=statemanager(bip.initial_state)
#sampler=[RandomWalk(bip, stepsize=stepsize), AdaptiveRandomWalk(bip, stepsize=stepsize), \
#         HamiltonianMonteCarlo(bip, stepsize=stepsize), GaussianApproximation(bip)][0]
sampler=RandomWalk(bip, statemanager, stepsize_rule=stepsize_rule)
#sampler=HamiltonianMonteCarlo(bip, statemanager, stepsize_rule=stepsize_rule)
#sampler=GaussianApproximation(bip)

bip.run(sampler, statemanager, 2e4)

plt.plot(grid.coords.T, exact_solution.T, label='exact solution')
plt.plot(grid.coords.T, bip.reco, label='reco')
plt.plot(grid.coords.T, exact_data, label='exact data')
plt.plot(grid.coords.T, data, label='data')
plt.plot(grid.coords.T, bip.reco_data, label='reco data')
plt.legend()
plt.show()


a = np.array([s.positions for s in statemanager.states[-1000:]])
v = a.std(axis=0)*3
m = a.mean(axis=0)
plt.plot(np.array([m+v, m-v, exact_solution]).T)
plt.plot( exact_solution.T, label='exact solution')