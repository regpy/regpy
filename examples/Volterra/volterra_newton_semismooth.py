from itreg.operators.volterra import Volterra
from itreg.hilbert import L2
from itreg.discrs import UniformGrid
from itreg.solvers import HilbertSpaceSetting
from itreg.solvers.newton_semismooth import NewtonSemiSmooth
import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s')

grid = UniformGrid(np.linspace(0, 2*np.pi, 200))
op = Volterra(grid)

exact_solution = np.sin(grid.coords[0])
exact_data = op(exact_solution)
noise = 0.3 * op.domain.randn()
data = exact_data + noise
init = 0.9*np.sin(grid.coords[0])
init_sol = init.copy()
init_data = op(init)

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)

newton = NewtonSemiSmooth(setting, data, init, alpha=1, psi_minus=-1, psi_plus=1)
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(setting.Hcodomain.norm, data,
                      noiselevel=0,
                      tau=1.1))

reco, reco_data = newton.run(stoprule)

plt.plot(grid.coords[0], exact_solution.T, label='exact solution')
plt.plot(grid.coords[0], reco, label='reco')
plt.plot(grid.coords[0], exact_data, label='exact data')
plt.plot(grid.coords[0], data, label='data')
plt.plot(grid.coords[0], reco_data, label='reco data')
plt.legend()
plt.show()
