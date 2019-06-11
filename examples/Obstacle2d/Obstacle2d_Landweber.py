# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:07:16 2019

@author: Bjoern Mueller
"""

import setpath

#import itreg.operators as iop
from itreg.operators.Obstacle2d.PotentialOp import PotentialOp as iop
#from itreg.operators.Obstacle2d.PotentialOp import PotentialOp as iop
#import itreg.operators.Obstacle2d.PotentialOp as iop
from itreg.operators.Obstacle2d.PotentialOp import PotentialOp 
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

xs = np.linspace(0, 2 * np.pi, 200)
spacing = xs[1] - xs[0]

grid=User_Defined(xs, (200,))

op=iop(L2(grid))
#op = PotentialOp(L2(grid))

exact_solution = np.ones(200)
exact_data = op(exact_solution)
#noise = 0.03 * op.domain.rand(np.random.randn)
data = exact_data 

#noiselevel = op.range.norm(noise)

init = op.domain.one()

_, deriv = op.linearize(init)
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