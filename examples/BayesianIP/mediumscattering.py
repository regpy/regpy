from regpy.operators.mediumscattering import MediumScatteringFixed
from regpy.operators import CoordinateProjection
from regpy.hilbert import L2, Sobolev, HilbertPullBack
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber
#from regpy.util import test_adjoint
import regpy.stoprules as rules
import regpy.util as util


from regpy.BIP.mcmc import Settings, RandomWalk, adaptive_stepsize

from regpy.BIP.MonteCarlo_basics import statemanager
#from regpy.BIP.MonteCarlo_basics import AdaptiveRandomWalk

from regpy.BIP.prior_distribution.prior_distribution import tikhonov
from regpy.BIP.likelihood_distribution.likelihood_distribution import unity

import numpy as np
import logging
import matplotlib.pyplot as plt
from functools import partial



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s')

scattering = MediumScatteringFixed(
    gridshape=(65, 65),
    radius=1,
    wave_number=1,
    inc_directions=util.linspace_circle(16),
    farfield_directions=util.linspace_circle(16),
    # support=lambda grid, radius: np.max(np.abs(grid.coords), axis=0) <= radius,
    # coarseshape=(17, 17),
)

contrast = scattering.domain.zeros()
r = np.linalg.norm(scattering.domain.coords, axis=0)
contrast[r < 1] = np.exp(-1/(1 - r[r < 1]**2))

projection = CoordinateProjection(
    scattering.domain,
    scattering.support)
embedding = projection.adjoint

op = scattering * embedding

exact_solution = projection(contrast)
exact_data = op(exact_solution)
noise = 0.03 * op.codomain.randn()
data = exact_data + noise
init = 1.1 * op.domain.ones()

setting = HilbertSpaceSetting(
    op=op,
    Hdomain=HilbertPullBack(Sobolev(index=1), embedding, inverse='cholesky'),
    Hcodomain=L2)


solver = Landweber(setting, data, init, stepsize=0.01)
stopping_rule = (
    rules.CountIterations(100) +
    rules.Discrepancy(setting.Hcodomain.norm, data,
                      noiselevel=setting.Hcodomain.norm(noise),
                      tau=1))



n_iter   = 2e3
stepsize = [1e-2, 1e-1, 5e-1, 7e-1, 1e0, 1.2, 1.5, 2.5, 10, 20][-4]
Temperature=1e-5
reg_parameter=1e-4


n_codomain=np.prod(op.codomain.shape)

#prior=gaussian_prior(reg_parameter*np.eye(op.domain.shape[0]), setting, np.zeros(op.domain.shape[0]))
#likelihood=gaussian_likelihood(setting, np.eye(n_codomain), exact_data)
prior=tikhonov(setting, reg_parameter)
likelihood=unity(setting)


stepsize_rule=partial(adaptive_stepsize, stepsize_factor=1.05)

bip=Settings(setting, data, prior, likelihood, Temperature, solver, stopping_rule,
              n_iter=n_iter, stepsize_rule=stepsize_rule)

statemanager=statemanager(bip.initial_state)
#sampler=[RandomWalk(bip, stepsize=stepsize), AdaptiveRandomWalk(bip, stepsize=stepsize), \
#         HamiltonianMonteCarlo(bip, stepsize=stepsize), GaussianApproximation(bip)][0]
sampler=RandomWalk(bip, statemanager, stepsize_rule=stepsize_rule)
#sampler=HamiltonianMonteCarlo(bip, statemanager, stepsize_rule=stepsize_rule)
#sampler=GaussianApproximation(bip)

bip.run(sampler, statemanager)


a = np.array([s.positions for s in statemanager.states[-9000:]])
m = a.mean(axis=0)
solution=embedding(m)

plt.imshow(np.abs(solution))
plt.show()

solution=embedding(exact_solution)
plt.imshow(np.abs(solution))
plt.show()

solution=embedding(bip.first_state)
plt.imshow(np.abs(solution))
plt.show()

#from regpy.BIP.plot_functions import plot_lastiter
#from regpy.BIP.plot_functions import plot_mean
from regpy.BIP.plot_functions import plot_verlauf
#from regpy.BIP.plot_functions import plot_iter

#plot_lastiter(bip, exact_solution, exact_data, data)
#plot_mean(statemanager, exact_solution, n_iter=1e4)
plot_verlauf(statemanager, pdf=bip, exact_solution=exact_solution, plot_real=True)
#plot_iter(bip, statemanager, 10)
