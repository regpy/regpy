#!/usr/bin/env python

import setpath

from itreg.operators.Volterra.volterra import NonlinearVolterra
from itreg.spaces import L2
from itreg.grids import Square_1D
from itreg.solvers import Landweber
from itreg.util import test_adjoint
import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

spacing=2*np.pi/200
grid=Square_1D((200,), np.pi, spacing)

op = NonlinearVolterra(L2(grid), exponent=3, spacing=spacing)

exact_solution = np.sin(grid.coords)
exact_data = op(exact_solution)
noise = 0.03 * op.domain.rand(np.random.randn)
data = exact_data + noise

noiselevel = op.codomain.norm(noise)

init = op.domain.one()

_, deriv = op.linearize(init)
test_adjoint(deriv)

landweber = Landweber(op, data, init, stepsize=0.01)
stoprule = (
    rules.CountIterations(1000) +
    rules.Discrepancy(op.codomain.norm, data, noiselevel, tau=1.1))

reco, reco_data = landweber.run(stoprule)

plt.plot(grid.coords.T, exact_solution.T, label='exact solution')
plt.plot(grid.coords.T, reco, label='reco')
plt.plot(grid.coords.T, exact_data, label='exact data')
plt.plot(grid.coords.T, data, label='data')
plt.plot(grid.coords.T, reco_data, label='reco data')
plt.legend()
plt.show()
