from regpy.solvers.irgnm import IrgnmCG

from regpy.operators.PINEM import wave_field_reco_PINEM
from regpy.hilbert import L2
from regpy.discrs import UniformGrid
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
fresnelNumber = 1e-3    # Fresnel-number of the simulated imaging system, associated with the unit-lengthscale
                        # in grid (i.e. with the size of one pixel for the above choice of grid)
noise_level = 0.01      # Noise level in the simulated data

# Uniform grid
Xdim= 256; Ydim= 256
grid = UniformGrid(np.arange(Xdim), np.arange(Ydim)).complex_space()
[Xco,Yco] = np.meshgrid(np.arange(-1,1,2/Xdim),np.arange(-1,1,2/Ydim))
mask = (abs(Xco+0.2)<=0.2) & (abs(Yco)<=0.4)
mask = mask | (abs((Xco-0.35)*(Xco-0.35)+(Yco-0.35)*(Yco-0.35))<=0.01)

# Forward operator
op = wave_field_reco_PINEM(grid, fresnelNumber,mask.astype(complex))

# Create phantom phase-image (= padded example-image)
picture = ascent()
exact_solution = picture[-Xdim//2:,-Ydim//2:].astype(np.float64) \
    + 1j*picture[:Xdim//2,:Ydim//2].astype(np.float64)
exact_solution /= abs(exact_solution).max()
pad_amount = tuple([(grid.shape[0] - exact_solution.shape[0])//2, (grid.shape[1] - exact_solution.shape[1])//2])
exact_solution = np.pad(exact_solution, pad_amount, 'constant', constant_values=0)
exact_solution = exact_solution.astype(complex);

# Create exact and noisy data
exact_data = op(exact_solution)
noise = noise_level * op.codomain.randn()
data = exact_data + noise

# Image-reconstruction using the IRGNM method
setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)
init_vec = np.ones_like(exact_solution)

solver = IrgnmCG(setting, data, regpar=10, regpar_step = 2/3, init = init_vec)
stoprule = (
    rules.CountIterations(max_iterations=30) +
    rules.Discrepancy(
        setting.Hcodomain.norm,
        data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=1.1
    )
)

#reco, reco_data = solver.run(stoprule)
reco,reco_data = solver.run(stoprule)
#reco = mask

# Plot reults
fig, axs = plt.subplots(2, 2)
axs[0,0].set_title('Exact solution (real part)')
axs[0,0].imshow(mask*exact_solution.real)
axs[0,1].set_title('Exact solution (imag. part)')
axs[0,1].imshow(mask*exact_solution.imag)
axs[1,0].set_title('Reconstruction (real part)')
axs[1,0].imshow(reco.real)
axs[1,1].set_title('Reconstruction (imag. part)')
axs[1,1].imshow(reco.imag)

data_comp = op.codomain.split(data)
reco_data_comp = op.codomain.split(reco_data)
fig2, axs2 = plt.subplots(2, 3)
for j in range(3):
    axs2[0,j].imshow(data_comp[j])
    axs2[0,j].set_title('Simulated data')
    axs2[1,j].imshow(reco_data_comp[j])
    axs2[1,j].set_title('reconstructed data')
plt.show()