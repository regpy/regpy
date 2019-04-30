# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:43:53 2019

@author: Hendrik Müller
"""

import setpath

from itreg.operators.Reaction.ReactionCoefficient_2D import ReactionCoefficient
from itreg.spaces import L2
from itreg.solvers import Landweber
from itreg.util import test_adjoint
import itreg.stoprules as rules
from itreg.grids import User_Defined

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

N=10
xcoo=np.linspace(0, 1, 10)
ycoo=np.linspace(0, 1, 10)
spacing = xcoo[1] - xcoo[0]

#rhs=np.dot(np.sin(xcoo).reshape((N, 1)), np.cos(ycoo).reshape((1, N)))
rhs=np.ones((N, N))

coords=np.asarray([xcoo, ycoo])
grid=User_Defined(coords, (10, 10))
op = ReactionCoefficient(L2(grid), rhs, spacing=spacing)

#exact_solution = np.dot(np.sin(xcoo).reshape((N, 1)), np.cos(ycoo).reshape((1, N)))
exact_solution=np.ones((N, N))
exact_data = op(exact_solution)
noise = 0.03 * op.domain.rand(np.random.randn).reshape((10, 10))
data = exact_data + noise

#noiselevel = op.range.norm(noise)

#init = op.domain.one()
init=1.1*np.ones((N, N))

#_, deriv = op.linearize(init)
#test_adjoint(deriv)
#deriv(init)


landweber = Landweber(op, exact_data, init, stepsize=0.1)
stoprule = (
    rules.CountIterations(1000) +
    rules.Discrepancy(op.range.norm, exact_data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)

xs=np.linspace(1, 100, 100)

plt.plot(xs, exact_solution.reshape((100, 1)), label='exact solution')
plt.plot(xs, reco.reshape((100, 1)), label='reco')
plt.plot(xs, exact_data.reshape((100, 1)), label='exact data')
plt.plot(xs, data.reshape((100,1)), label='data')
plt.plot(xs, reco_data.reshape((100,1)), label='reco data')
plt.legend()
plt.show()