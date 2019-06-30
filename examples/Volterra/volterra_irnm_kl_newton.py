"""Example:
    Solver: IRNM_KL_Newton
    Operator: Volterra
"""

import setpath  # NOQA

from itreg.operators.Volterra.volterra import Volterra
from itreg.spaces import L2
from itreg.grids import Square_1D
from itreg.solvers import IRNM_KL_Newton
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

irnm_kl_newton = IRNM_KL_Newton(op, data, np.ones(grid.shape), \
        alpha0 = 1e-0, alpha_step = 0.3, intensity = 1,\
        scaling = 1, offset = 1e-4, offset_step = 1, inner_res = 1e-10, \
        inner_it = 100, cgmaxit = 100)

stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(op.codomain.norm, data, noiselevel, tau=1.1))

reco, reco_data = irnm_kl_newton.run(stoprule)
plt.plot(grid.coords.T, exact_solution.T, label='exact solution')
plt.plot(grid.coords.T, reco, label='reco')
plt.plot(grid.coords.T, exact_data, label='exact data')
plt.plot(grid.coords.T, data, label='data')
plt.plot(grid.coords.T, reco_data, label='reco data')
plt.legend()
plt.show()

