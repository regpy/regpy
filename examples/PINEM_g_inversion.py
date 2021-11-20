from numpy.core.numeric import ones_like
from regpy.solvers.irgnm import IrgnmCG

from regpy.operators import Exponential, Matrix_of_operators 
from regpy.operators.PINEM import Nemitzky_op_for_g, PINEM_g_to_data
from regpy.hilbert import L2
from regpy.discrs import UniformGrid, DirectSum
from regpy.solvers import HilbertSpaceSetting
import regpy.stoprules as rules

import numpy as np
from scipy.misc import ascent
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

# Example parameters
fresnelNumber = 5e3    # Fresnel-number of the simulated imaging system, associated with the unit-lengthscale
                        # in grid (i.e. with the size of one pixel for the above choice of grid)
noise_level = 0.0001      # Noise level in the simulated data

# Uniform grid
Xdim= 256; Ydim= 256
grid = UniformGrid(np.linspace(0,1,Xdim,endpoint=False), np.linspace(0,1,Ydim,endpoint=False))
sum_of_grids = DirectSum(grid,grid)
[Xco,Yco] = np.meshgrid(np.arange(-1,1,2/Xdim),np.arange(-1,1,2/Ydim))
mask = (abs(Xco+0.2)<=0.2) & (abs(Yco)<=0.4)
mask = mask | (abs((Xco-0.35)*(Xco-0.35)+(Yco-0.35)*(Yco-0.35))<=0.01)
#mask = mask.astype(float)
#mask_abs = mask # mask for abs_g
#mask_arg = mask # mask for arg_g 
#masks = sum_of_grids.join(mask_abs,mask_arg)
A_Psi0_Multiplier = np.ones(grid.shape,complex)

# Forward operator
op = PINEM_g_to_data(grid,fresnelNumber,mask,A_Psi0_Multiplier,N=2)

# Create phantom phase-image (= padded example-image)
picture = ascent()
exact_solution = picture[-Xdim//2:,-Ydim//2:].astype(np.float64)/255 \
    * np.exp(1j*2*np.pi*picture[:Xdim//2,:Ydim//2].astype(np.float64)/255)
exact_solution /= 10*abs(exact_solution).max()
exact_solution += ones_like(exact_solution)
pad_amount = tuple([(grid.shape[0] - exact_solution.shape[0])//2, (grid.shape[1] - exact_solution.shape[1])//2])
exact_solution = np.pad(exact_solution, pad_amount, 'constant', constant_values=1)
exact_solution = exact_solution.astype(complex)*mask

# Create exact and noisy data
sexact_solution = sum_of_grids.join(np.abs(exact_solution), np.angle(exact_solution))
exact_data = op(sexact_solution)
#exact_data = op(exact_solution)
noise = noise_level * op.codomain.randn()
data = exact_data + noise

# Image-reconstruction using the IRGNM method
setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)
init_vec = sum_of_grids.join(grid.ones(), grid.zeros())

solver = IrgnmCG(
    setting, data, init = init_vec,
    regpar=1e-4, regpar_step = 2/3,
    inner_it_logging_level=logging.DEBUG)
stoprule = (
    rules.CountIterations(max_iterations=5) +
    rules.Discrepancy(
        setting.Hcodomain.norm,
        data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=1.1
    )
)

#reco, reco_data = solver.run(stoprule)
reco, reco_data = solver.run(stoprule)
reco1,reco2=sum_of_grids.split(reco)
reco_data1,reco_data2 = sum_of_grids.split(reco_data)
data1,data2 = sum_of_grids.split(data)
#reco = mask

# Plot reults
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs[0,0].set_title('Exact solution (amplitude)')
axs[0,0].imshow(np.abs(exact_solution))
axs[0,1].set_title('Exact solution (phase)')
axs[0,1].imshow(np.angle(exact_solution))
axs[1,0].set_title('Reconstruction (amplitude)')
axs[1,0].imshow(reco1)
#axs[1,0].imshow(reco.real)
axs[1,1].set_title('Reconstruction (phase)')
axs[1,1].imshow(reco2)
#axs[1,1].imshow(reco.imag)

fig2, axs2 = plt.subplots(2, 2, sharex=True, sharey=True)
axs2[0,0].imshow(data1)
axs2[0,0].set_title('Simulated data 1')
axs2[1,0].imshow(reco_data1)
axs2[1,0].set_title('reconstructed data 1')
axs2[0,1].imshow(data2)
axs2[0,1].set_title('Simulated data 2')
axs2[1,1].imshow(reco_data2)
axs2[1,1].set_title('reconstructed data 2')
plt.show()