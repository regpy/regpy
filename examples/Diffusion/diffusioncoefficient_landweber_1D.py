# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:30:00 2019

@author: Hendrik Müller
"""

import setpath

from itreg.operators.Diffusion.DiffusionCoefficient_1D import DiffusionCoefficient
from itreg.spaces import L2
from itreg.solvers import Landweber
from itreg.solvers import IRGNM_CG
from itreg.util import test_adjoint
import itreg.stoprules as rules
from itreg.grids import User_Defined
from itreg.spaces import H1_NGSolve
from ngsolve.meshes import Make1DMesh


import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

xs = np.linspace(0, 1, 200)
spacing = xs[1] - xs[0]

grid=User_Defined(xs, xs.shape)

op = DiffusionCoefficient(L2(grid), rhs=np.ones(200), bc_left=1, bc_right=1, spacing=spacing)

exact_solution = np.ones(200)
exact_data = op(exact_solution)
noise = 0.003 * op.domain.rand(np.random.randn)
data = exact_data+noise

#noiselevel = op.range.norm(noise)

init = 1.1*op.domain.one()
init_data=op(init)

#_, deriv = op.linearize(init)
#test_adjoint(deriv)



landweber = Landweber(op, data, init, stepsize=5)
#irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
    rules.CountIterations(1000) +
    rules.Discrepancy(op.range.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)

plt.plot(xs, exact_solution, label='exact solution')
plt.plot(xs, reco, label='reco')
plt.plot(xs, exact_data, label='exact data')
plt.plot(xs, data, label='data')
plt.plot(xs, reco_data, label='reco data')
#plt.plot(xs, init_data, label='init data')
plt.legend()
plt.show()