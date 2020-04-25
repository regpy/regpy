from regpy.operators.volterra import Volterra
from regpy.hilbert import L2, Sobolev
from regpy.discrs import UniformGrid
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.irgnm import IrgnmCGLanczos
import regpy.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

grid = UniformGrid(np.linspace(0, 2*np.pi, 200))
op = Volterra(grid, exponent=1)

exact_solution = np.sin(grid.coords[0])
exact_data = op(exact_solution)
noise = 0.03 * op.domain.randn()
data = exact_data + noise
init = op.domain.ones()

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)

solver = IrgnmCGLanczos(setting, data, regpar=1, regpar_step=0.9, init=init)
stoprule = (
    rules.CountIterations(max_iterations=10) +
    rules.Discrepancy(
        setting.Hcodomain.norm, data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=1.1
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

preconditioned = solver.M @ (solver.setting.Hdomain.gram_inv(solver.deriv.adjoint(solver.setting.Hcodomain.gram(solver.deriv(solver.M @ exact_solution))))+solver.regpar*solver.M @ exact_solution)
    
unpreconditioned = solver.setting.Hdomain.gram_inv(solver.deriv.adjoint(solver.setting.Hcodomain.gram(solver.deriv(exact_solution))))+solver.regpar*exact_solution

plt.plot(preconditioned)
#plt.plot(unpreconditioned)
plt.plot(exact_solution)
plt.show()
