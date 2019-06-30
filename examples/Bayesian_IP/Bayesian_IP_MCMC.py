# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:18:06 2019

@author: Bjoern Mueller
"""

import setpath


from itreg.operators.Volterra.volterra import NonlinearVolterra
from itreg.spaces import L2
#from itreg.grids import Square_1D
from itreg.grids import User_Defined
#from itreg.BIP.mcmc import tikhonov_like
from itreg.BIP.mcmc import Settings
from itreg.BIP.prior_distribution.prior_distribution import gaussian as gaussian_prior
from itreg.BIP.likelihood_distribution.likelihood_distribution import gaussian as gaussian_likelihood
from itreg.solvers import Landweber
from itreg.util import test_adjoint
from itreg.BIP.MonteCarlo_basics import fixed_stepsize
from itreg.BIP.MonteCarlo_basics import adaptive_stepsize
from itreg.BIP.MonteCarlo_basics import statemanager
from itreg.BIP.MonteCarlo_basics import RandomWalk
#from itreg.BIP.MonteCarlo_basics import AdaptiveRandomWalk
from itreg.BIP.MonteCarlo_basics import HamiltonianMonteCarlo
from itreg.BIP.MonteCarlo_basics import GaussianApproximation
import itreg.stoprules as rules
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
grid=User_Defined(xs, 200)

op = NonlinearVolterra(L2(grid), exponent=3, spacing=spacing)

exact_solution = np.sin(grid.coords)
exact_data = op(exact_solution)
noise = 0.03 * op.domain.rand(np.random.randn)
data = exact_data + noise

noiselevel = op.range.norm(noise)

init=op.domain.one()

solver = Landweber(op, data, init, stepsize=0.01)
stopping_rule = (
    rules.CountIterations(10) +
    rules.Discrepancy(op.range.norm, data, noiselevel, tau=1.1))

#prior=gaussian_prior(np.eye(200), op, exact_solution+0.1*np.ones(exact_solution.shape[0]))
#likelihood=gaussian_likelihood(op, np.eye(200), exact_data+0.1*np.ones(exact_data.shape[0]))
prior=gaussian_prior(0.1*np.eye(200), op, np.zeros(200))
likelihood=gaussian_likelihood(op, np.eye(200), exact_data)
#prior=l1_prior(1, op)
#likelihood=l1_likelihood(op, 1, exact_data)

## run random walk mcmc

n_iter   = 2e4
stepsize = [1e-2, 1e-1, 5e-1, 7e-1, 1e0, 1.2, 1.5, 2.5][0]



#sampler=['RandomWalk', 'AdaptiveRandomWalk', 'HamiltonianMonteCarlo', 'GaussianApproximation'][0]

    
stepsize_rule=partial(adaptive_stepsize, stepsize_factor=1.05)
#stepsize_rule=fixed_stepsize

bip=Settings(op, data, prior, likelihood, solver, stopping_rule, 0.001, 
              n_iter=n_iter, stepsize_rule=stepsize_rule)

statemanager=statemanager(bip.initial_state)
#sampler=[RandomWalk(bip, stepsize=stepsize), AdaptiveRandomWalk(bip, stepsize=stepsize), \
#         HamiltonianMonteCarlo(bip, stepsize=stepsize), GaussianApproximation(bip)][0]
sampler=RandomWalk(bip, statemanager, stepsize_rule=stepsize_rule)
#sampler=HamiltonianMonteCarlo(bip, statemanager, stepsize_rule=stepsize_rule)
#sampler=GaussianApproximation(bip)

bip.run(sampler, statemanager, 2e4)

from itreg.BIP.plot_functions import plot_lastiter
from itreg.BIP.plot_functions import plot_mean
from itreg.BIP.plot_functions import plot_verlauf
from itreg.BIP.plot_functions import plot_iter

#plot_lastiter(bip, exact_solution, exact_data, data)
#plot_mean(statemanager, exact_solution, n_iter=1000)
#plot_verlauf(statemanager, pdf=bip, exact_solution=exact_solution, plot_real=True)
plot_iter(bip, statemanager, 10)



