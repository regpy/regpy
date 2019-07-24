"""Example:
    Solver: IRGNM_CG
    Operator: Volterra
"""

import setpath  # NOQA

from itreg.operators.volterra import Volterra
from itreg.spaces import L2
from itreg.grids import Square_1D
from itreg.solvers import IRGNM_CG
import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

#xs = np.linspace(0, 2 * np.pi, 200)
#spacing = xs[1] - xs[0]

spacing=2*np.pi/200
grid=Square_1D((200,), np.pi, spacing)
op = Volterra(L2(grid), spacing=spacing)

exact_solution = np.sin(grid.coords)
exact_data = op(exact_solution)
noise = 0.1 * np.random.normal(size=grid.shape)
data = exact_data + noise

noiselevel = op.codomain.norm(noise)

irgnm_cg = IRGNM_CG(op, data, np.zeros(grid.shape), cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
#stoprule = rules.CombineRules(
#    [rules.CountIterations(100),
#     rules.Discrepancy(op, data, noiselevel, tau=1.1)],
#    op=op)


stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(op.codomain.norm, data, noiselevel, tau=1.1))

reco, reco_data = irgnm_cg.run(stoprule)
plt.plot(grid.coords.T, exact_solution.T)
plt.plot(grid.coords.T, reco)

plt.plot(grid.coords.T, exact_data)
plt.plot(grid.coords.T, reco_data)
plt.plot(grid.coords.T, data)
plt.show()

