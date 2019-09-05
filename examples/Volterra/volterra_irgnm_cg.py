import setpath

from itreg.operators import NonlinearVolterra
from itreg.spaces import L2, Sobolev, UniformGrid
from itreg.solvers import IrgnmCG, HilbertSpaceSetting
import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s')

grid = UniformGrid(np.linspace(0, 2*np.pi, 200))
op = NonlinearVolterra(grid, exponent=3)

exact_solution = np.sin(grid.coords[0])
exact_data = op(exact_solution)
noise = 0.03 * op.domain.randn()
data = exact_data + noise
init = op.domain.ones()

setting = HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=L2)

solver = IrgnmCG(setting, data, regpar=100, regpar_step=0.9, init=init)
stoprule = (
    rules.CountIterations(1000) +
    rules.Discrepancy(
        setting.Hcodomain.norm, data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=2
    )
)

reco, reco_data = solver.run(stoprule)

plt.plot(grid.coords[0], exact_solution.T, label='exact solution')
plt.plot(grid.coords[0], reco, label='reco')
plt.plot(grid.coords[0], exact_data, label='exact data')
plt.plot(grid.coords[0], data, label='data')
plt.plot(grid.coords[0], reco_data, label='reco data')
plt.legend()
plt.show()
