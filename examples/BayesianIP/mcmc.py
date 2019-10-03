import setpath

#import itreg

from itreg.operators import NonlinearVolterra, Volterra
from itreg.spaces import L2, HilbertPullBack, UniformGrid
from itreg.spaces import Sobolev, HilbertPullBack, UniformGrid
from itreg.solvers import IrgnmCG, Landweber, HilbertSpaceSetting
#from itreg.util import test_adjoint
import itreg.stoprules as rules


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
from itreg.BIP.prior_distribution.prior_distribution import tikhonov as tikhonov_prior
from itreg.BIP.likelihood_distribution.likelihood_distribution import l1 as l1_likelihood
from itreg.BIP.likelihood_distribution.likelihood_distribution import tikhonov as tikhonov_likelihood
from itreg.BIP import HMCState
from itreg.BIP import State

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

#op = NonlinearVolterra(grid, exponent=3)
op=Volterra(grid)

exact_solution = np.sin(xs)
exact_data = op(exact_solution)
noise = 0.03 * op.domain.rand(np.random.randn)
data = exact_data + noise

#noiselevel = op.codomain.norm(noise)

init = op.domain.ones()

setting = HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=L2)

noiselevel=setting.Hcodomain.norm(noise)
#solver = IRGNM_CG(setting, data, init)
#stopping_rule = (
#    rules.CountIterations(100) +
#    rules.Discrepancy(setting.codomain.norm, data, noiselevel=setting.codomain.norm(noise), tau=1.1))



#prior=gaussian_prior(np.eye(200), op, exact_solution+0.1*np.ones(exact_solution.shape[0]))
#likelihood=gaussian_likelihood(op, np.eye(200), exact_data+0.1*np.ones(exact_data.shape[0]))

#prior=l1_prior(1, op)
#likelihood=l1_likelihood(op, 1, exact_data)

## run random walk mcmc

"""werte für adaptive randomwalk, L2-regularization:
n_iter=2e4
stepsize=1e-2
Temperature=0.001
reg_parameter=0.1
"""

"""exemplary parameters for adaptive randomwalk, H1-regularization
n_iter=2e7
stepsize=1.5
Temperature=0.001
reg_parameter=0.01
Averaging over last 2e4 parameters
"""

n_iter   = 1e4
stepsize = [1e-2, 1e-1, 5e-1, 7e-1, 1e0, 1.2, 1.5, 2.5, 10, 20][3]/1000
Temperature=1e-6
reg_parameter=1e-1


#prior=gaussian_prior(1/reg_parameter*np.eye(200), setting, np.zeros(200))
#likelihood=gaussian_likelihood(setting, np.eye(200), exact_data)
prior=tikhonov_prior(setting, reg_parameter)
likelihood=tikhonov_likelihood(setting, exact_data)




stepsize_rule=partial(adaptive_stepsize, stepsize_factor=1.05)
#stepsize_rule=fixed_stepsize

bip=Settings(setting, data, prior, likelihood, Temperature,
              n_iter=n_iter, stepsize_rule=stepsize_rule)

statemanager=statemanager(bip.initial_state)
#sampler=[RandomWalk(bip, stepsize=stepsize), AdaptiveRandomWalk(bip, stepsize=stepsize), \
#         HamiltonianMonteCarlo(bip, stepsize=stepsize), GaussianApproximation(bip)][0]
#sampler=RandomWalk(bip, statemanager, stepsize_rule=stepsize_rule)
#sampler=HamiltonianMonteCarlo(bip, statemanager, stepsize=1, stepsize_rule=stepsize_rule)
sampler=GaussianApproximation(bip)

bip.run(sampler, statemanager)

from itreg.BIP.plot_functions import plot_lastiter
from itreg.BIP.plot_functions import plot_mean
from itreg.BIP.plot_functions import plot_verlauf
from itreg.BIP.plot_functions import plot_iter

plot_lastiter(bip, exact_solution, exact_data, data)
plot_mean(statemanager, exact_solution, n_iter=statemanager.N-1, variance=np.asarray([1]))
plot_verlauf(statemanager, pdf=bip, exact_solution=exact_solution, plot_real=True)
plot_iter(bip, statemanager, 10)
