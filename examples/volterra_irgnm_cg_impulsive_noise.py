from regpy.operators.volterra import Volterra
from regpy.hilbert import L2, Sobolev
from regpy.vecsps import UniformGridFcts
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
import regpy.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

grid = UniformGridFcts(np.linspace(0, 2*np.pi, 200))
op = Volterra(grid, exponent=3)

"""Impulsive Noise"""
sigma = 0.01*np.ones(grid.coords.shape[1])
sigma[100:110] = 0.5

exact_solution = np.sin(grid.coords[0])
exact_data = op(exact_solution)
noise = sigma * op.domain.randn()
data = exact_data + noise
init = op.domain.ones()

setting = HilbertSpaceSetting(op=op, h_domain=Sobolev(index=2), h_codomain=L2)

solver = IrgnmCG(setting, data, regpar=1, regpar_step=0.9, init=init)
stoprule = (
    rules.CountIterations(max_iterations=100) +
    rules.Discrepancy(
        setting.h_codomain.norm, data,
        noiselevel=setting.h_codomain.norm(noise),
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

