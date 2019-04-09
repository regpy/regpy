# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:30:00 2019

@author: Hendrik Müller
"""

import setpath

from itreg.operators.DiffusionCoefficient_1D import DiffusionCoefficient
from itreg.spaces import L2
from itreg.solvers import Landweber
from itreg.util import test_adjoint
import itreg.stoprules as rule
from itreg.grids import User_Defined

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

xs = np.linspace(0, 1, 200)
spacing = xs[1] - xs[0]

grid=User_Defined(xs, xs.shape)

op = DiffusionCoefficient(L2(grid), rhs=np.ones(200), spacing=spacing)

exact_solution = np.ones(200)
exact_data = op(exact_solution)
#noise = 0.03 * op.domain.rand(np.random.randn)
data = exact_data

#noiselevel = op.range.norm(noise)

init = op.domain.one()

#_, deriv = op.linearize(init)
#test_adjoint(deriv)



landweber = Landweber(op, data, init, stepsize=0.1)
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(op.range.norm, data, noiselevel=0.1, tau=1.1))

reco, reco_data = landweber.run(stoprule)

plt.plot(xs, exact_solution, label='exact solution')
plt.plot(xs, reco, label='reco')
plt.plot(xs, exact_data, label='exact data')
plt.plot(xs, data, label='data')
plt.plot(xs, reco_data, label='reco data')
plt.legend()
plt.show()