import logging

import matplotlib.pyplot as plt
import numpy as np

from regpy.solvers.nonlinear.irgnm import IrgnmCG
from regpy.solvers.nonlinear.newton import NewtonCG

import regpy.stoprules as rules
from regpy.hilbert import L2, Sobolev
from regpy.vecsps.curve import StarTrigDiscr
from regpy.solvers import HilbertSpaceSetting
from potential_op import Potential

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

#Forward operator
op = Potential(
    domain=StarTrigDiscr(200),
    radius=1.5,
    nmeas=128,
)

setting = HilbertSpaceSetting(op=op, h_domain=Sobolev, h_codomain=L2)

#Exact data and Poission data
exact_solution = op.domain.sample(lambda t: np.sqrt(3*np.cos(t)**2+1)/2)
exact_data = op(exact_solution)
noise = op.codomain.randn()
noise = 0.01*setting.h_codomain.norm(exact_data)/setting.h_codomain.norm(noise) * noise
data = exact_data + noise

#Initial guess
init = op.domain.sample(lambda t: 1)

#Solver: NewtonCG or IrgnmCG
solver = NewtonCG(
    setting, data, init = init,
        cgmaxit=50, rho=0.3
)

"""
solver = IrgnmCG(
    setting, data,
    regpar = 10,
    regpar_step = 0.8,
    init = init,
    cg_pars = dict(
        tol = 1e-4
    )
)
"""
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(
        setting.h_codomain.norm, data,
        noiselevel = setting.h_codomain.norm(noise),
        tau=1.2
    )
)

#Plot function
plt.ion()
fig, axs = plt.subplots(1, 2)
axs[0].set_title('Obstacle')
axs[1].set_title('Heat flux')

for n, (reco, reco_data) in enumerate(solver.until(stoprule)):
    if n % 1 == 0:
        axs[0].clear()
        axs[0].plot(*op.domain.eval_curve(exact_solution).curve[0])
        axs[0].plot(*op.domain.eval_curve(reco).curve[0])

        axs[1].clear()
        axs[1].plot(exact_data, label='exact')
        axs[1].plot(reco_data, label='reco')
        axs[1].plot(data, label='measured')
        axs[1].legend()
        axs[1].set_ylim(ymin=0)
        plt.pause(0.5)

plt.ioff()
plt.show()
