from regpy.operators.volterra import Volterra
from regpy.hilbert import L2
from regpy.discrs import UniformGrid
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.newton import NewtonSemiSmooth
import regpy.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

grid = UniformGrid(np.linspace(0, 2*np.pi, 200))
op = Volterra(grid)

exact_solution = np.sin(grid.coords[0])
exact_data = op(exact_solution)
noise = 0.03 * op.domain.randn()
data = exact_data + noise
init = grid.zeros()

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)

# Run the solver with a `-1 <= reco <= 1` constraint. For illustration, you can also try a
# constraint which is violated by the exact solution.
newton = NewtonSemiSmooth(setting, data, init, alpha=0.1, psi_minus=-1, psi_plus=1)
stoprule = (
    rules.CountIterations(1000) +
    rules.Discrepancy(
        setting.Hcodomain.norm, data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=1.1
    )
)

reco, reco_data = newton.run(stoprule)

plt.plot(grid.coords[0], exact_solution.T, label='exact solution')
plt.plot(grid.coords[0], reco, label='reco')
plt.plot(grid.coords[0], exact_data, label='exact data')
plt.plot(grid.coords[0], data, label='data')
plt.plot(grid.coords[0], reco_data, label='reco data')
plt.legend()
plt.show()
