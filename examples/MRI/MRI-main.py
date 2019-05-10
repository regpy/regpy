from itreg.spaces.l2 import L2
from itreg.solvers.landweber import Landweber
# from itreg.util import test_adjoint
from itreg.operators.MRI.MRI import parallel_MRI
import itreg.stoprules as rules
from itreg.grids import Square_2D, User_Defined

import numpy as np
import scipy as scp
import logging
import matplotlib.pyplot as plt
from itreg.solvers import IRGNM_CG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')
Nx=200
Ny=20
nr_coils=20
center=10

#coords=np.zeros((Nx, Ny, nr_coils+1))
#grid_x=np.arange(0, Nx, 1)/Nx
#grid_y=np.arange(0, Ny, 1)/Ny
#grid_xy, grid_xy=np.meshgrid(grid_x, grid_y)

coords=np.ones(Nx*Ny*(nr_coils+1))
grid=User_Defined(coords, coords.shape)
domain=L2(grid)
op=parallel_MRI(domain)


exact_solution=np.ones(Nx*Ny*(nr_coils+1))
exact_data=op(exact_solution)
#yscale=100/np.linalg.norm(exact_data)
yscale=1
data=yscale*exact_data
        

#init=op.domain.one()+1j*op.domain.zero()
init=2*exact_solution+1j*op.domain.zero()
init_sol=init.copy()
init_data, deriv = op.linearize(init)
#test_adjoint(deriv, tolerance=10**(-8))


#irgnm_cg = IRGNM_CG(op, data, np.zeros(grid.shape), cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])

#stoprule = (
#    rules.CountIterations(100) +
#    rules.Discrepancy(op.range.norm, data, noiselevel=0.1, tau=1.1))

#reco, reco_data = irgnm_cg.run(stoprule)

landweber= Landweber(op, data, init, stepsize=1/(Nx*Ny*(nr_coils+1)))
stoprule=(
    rules.CountIterations(30)+
    rules.Discrepancy(op.range.norm, data, noiselevel=0, tau=2))

reco, reco_data=landweber.run(stoprule)
