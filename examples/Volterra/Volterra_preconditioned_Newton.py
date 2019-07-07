import setpath  # NOQA

from itreg.operators import Volterra
from itreg.spaces import L2, UniformGrid
from itreg.solvers import HilbertSpaceSetting
from itreg.solvers.preconditioned_newton_cg import Newton_CG_Preconditioned
import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

grid=UniformGrid(np.linspace(1, 200, 200))

op = Volterra(grid)

exact_solution = np.cos(grid.coords[0])
exact_data = op(exact_solution)
noise = 0.1 * np.random.normal(size=grid.shape)
data = exact_data + noise


setting=HilbertSpaceSetting(op=op, domain=L2, codomain=L2)

newton_cg = Newton_CG_Preconditioned(setting, data, np.zeros(grid.shape), cgmaxit = 100, rho = 0.98)
stoprule = (
    rules.CountIterations(1) +
    rules.Discrepancy(setting.codomain.norm, data,
                      noiselevel=setting.codomain.norm(noise), tau=1.1))

reco, reco_data = newton_cg.run(stoprule)

plt.plot(grid.coords[0], exact_solution.T, label='exact solution')
plt.plot(grid.coords[0], reco, label='reco')
plt.plot(grid.coords[0], exact_data, label='exact data')
plt.plot(grid.coords[0], data, label='data')
plt.plot(grid.coords[0], reco_data, label='reco data')
plt.legend()
plt.show()