# -*- coding: utf-8 -*-


"""Example:
    Solver: Newton_CG
    Operator: Volterra
"""

import setpath  # NOQA

#from itreg.operators.Volterra.volterra import Volterra
from itreg.operators import Volterra


from itreg.spaces import L2, UniformGrid
from itreg.solvers import Newton_CG_Frozen, HilbertSpaceSetting
import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

#xs = np.linspace(0, 2 * np.pi, 200)
#spacing = xs[1] - xs[0]
grid = UniformGrid(np.linspace(0, 2*np.pi, 200))
op = Volterra(grid)

exact_solution = np.sin(grid.coords[0])
exact_data = op(exact_solution)
noise = 0.03 * op.domain.randn()
data = exact_data + noise
init = op.domain.ones()

setting = HilbertSpaceSetting(op=op, domain=L2, codomain=L2)

newton_cg = Newton_CG_Frozen(setting, data, init, cgmaxit = 100, rho = 0.98)
stoprule = (
    rules.CountIterations(1000) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0.03, tau=1.1))

reco, reco_data = newton_cg.run(stoprule)
plt.plot(grid.coords.T, exact_solution.T, label='exact solution')
plt.plot(grid.coords.T, reco, label='reco')
plt.plot(grid.coords.T, exact_data, label='exact data')
plt.plot(grid.coords.T, data, label='data')
plt.plot(grid.coords.T, reco_data, label='reco data')
plt.legend()
plt.show()