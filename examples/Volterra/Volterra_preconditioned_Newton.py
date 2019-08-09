import setpath  # NOQA

from itreg.operators import Volterra
from itreg.spaces import L2, UniformGrid
from itreg.solvers import HilbertSpaceSetting
from itreg.solvers.irgnm_cg_lanczos import IRGNM_CG_Lanczos
from itreg.solvers.irgnm_cg import IRGNM_CG
import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

grid=UniformGrid(np.linspace(0, 1, 200))

op = Volterra(grid)

exact_solution = np.cos(grid.coords[0])
exact_data = op(exact_solution)
noise = 0.1 * np.random.normal(size=grid.shape)
data = exact_data + noise


setting=HilbertSpaceSetting(op=op, domain=L2, codomain=L2)

irgnm_cg = IRGNM_CG_Lanczos(setting, data, np.zeros(grid.shape), cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
    rules.CountIterations(1) +
    rules.Discrepancy(setting.codomain.norm, data,
                      noiselevel=setting.codomain.norm(noise), tau=0.2))

reco, reco_data = irgnm_cg.run(stoprule)

plt.plot(grid.coords[0], exact_solution.T, label='exact solution')
plt.plot(grid.coords[0], reco, label='reco')
plt.legend()
plt.show()

plt.plot(grid.coords[0], exact_data, label='exact data')
plt.plot(grid.coords[0], data, label='data')
plt.plot(grid.coords[0], reco_data, label='reco data')
plt.legend()
plt.show()