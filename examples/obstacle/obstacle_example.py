import logging

import matplotlib.pyplot as plt
import numpy as np

from regpy.solvers.nonlinear.irgnm import IrgnmCG
from regpy.solvers.nonlinear.newton import NewtonCG
import regpy.stoprules as rules
from regpy.hilbert import L2, Sobolev
from regpy.solvers import RegularizationSetting
from dirichlet_op import DirichletOp
from dirichlet_op import create_synthetic_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

#Forward operator
op = DirichletOp(
    kappa = 3,
    N_inc = 4
)

setting = RegularizationSetting(op=op, penalty=Sobolev, data_fid=L2)

#Exact data
farfield, exact_solution = create_synthetic_data(op, true_curve='apple')

# Gaussian data 
noiselevel=0.01
noise = op.codomain.randn()
noise = noiselevel*setting.h_codomain.norm(farfield)/setting.h_codomain.norm(noise)*noise
data = farfield+noise

#Initial guess
t = 2*np.pi*np.arange(0, op.N_FK)/op.N_FK
init = 0.45*np.append(np.cos(t), np.sin(t)).reshape((2, op.N_FK))
init = init.flatten()

#Solver: NewtonCG or IrgnmCG
solver = NewtonCG(
    setting, data, init = init,
        cgmaxit=50, rho=0.6
)

"""
solver = IrgnmCG(
    setting, data,
    regpar=1.,
    regpar_step=0.5,
    init=init,
    cg_pars=dict(
        tol=1e-4
    )
)
"""
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(
        setting.h_codomain.norm, data,
        noiselevel=noiselevel,
        tau=2.1
    )
)

#Plot function
plt.ion()
fig, axs = plt.subplots(1, 2)
axs[0].set_title('Obstacle')
axs[1].set_title('Farfield (real part)')

for n, (reco, reco_data) in enumerate(solver.until(stoprule)):
    if n % 1 == 0:
        axs[0].clear()
        axs[0].plot(*exact_solution.z)
        axs[0].plot(*op.domain.bd_eval(reco, nvals=op.N_ieq, nderivs=3).z)
        
        axs[1].clear()
        axs[1].plot(op.codomain.coords[0][:,0], farfield.real[:,0], label='exact')
        axs[1].plot(op.codomain.coords[0][:,0], reco_data.real[:,0], label='reco')
        axs[1].plot(op.codomain.coords[0][:,0], data.real[:,0], label='measured')
        axs[1].legend()
        plt.pause(0.5)

plt.ioff()
plt.show()


