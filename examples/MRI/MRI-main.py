
import setpath


from itreg.spaces import L2
from itreg.solvers import Landweber
from itreg.util import test_adjoint
from itreg.operators.MRI.MRI import parallel_MRI
import itreg.stoprules as rules
from itreg.grids import Square_2D, User_Defined

import numpy as np
import scipy as scp
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')
Nx=128
Ny=96
nr_coils=12
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
yscale=100/np.linalg.norm(exact_data)
data=yscale*exact_data
        

#init=op.domain.one()+1j*op.domain.zero()
init=exact_solution+1j*op.domain.zero()
_, deriv = op.linearize(init)
#test_adjoint(deriv)

landweber= Landweber(op, data, init, stepsize=0.1)
stoprule=(
    rules.CountIterations(10)+
    rules.Discrepancy(op.range.norm, data, noiselevel=1, tau=2))

reco, reco_data=landweber.run(stoprule)
