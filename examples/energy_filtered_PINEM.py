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
fresnelNumber = 5e3    # Fresnel-number of the simulated imaging system, associated with the unit-lengthscale
                        # in grid (i.e. with the size of one pixel for the above choice of grid)
noise_level = 0.01      # Noise level in the simulated data

# Uniform grid
Xdim= 256; Ydim= 256
grid = UniformGrid(np.linspace(0,1,Xdim,endpoint=False), np.linspace(0,1,Ydim,endpoint=False)).complex_space()
[Xco,Yco] = np.meshgrid(np.arange(-1,1,2/Xdim),np.arange(-1,1,2/Ydim))
mask = (abs(Xco+0.2)<=0.2) & (abs(Yco)<=0.4)
mask = mask | (abs((Xco-0.35)*(Xco-0.35)+(Yco-0.35)*(Yco-0.35))<=0.01)

# Forward operator
op = wave_field_reco_PINEM(grid, fresnelNumber,mask.astype(complex))

# Create phantom phase-image (= padded example-image)
picture = ascent()
exact_solution = picture[-Xdim//2:,-Ydim//2:].astype(np.float64)/255 \
    * np.exp(1j*2*np.pi*picture[:Xdim//2,:Ydim//2].astype(np.float64)/255)
exact_solution /= abs(exact_solution).max()
pad_amount = tuple([(grid.shape[0] - exact_solution.shape[0])//2, (grid.shape[1] - exact_solution.shape[1])//2])
exact_solution = np.pad(exact_solution, pad_amount, 'constant', constant_values=0)
exact_solution = exact_solution.astype(complex)

# Create exact and noisy data
exact_data = op(exact_solution)
noise = noise_level * op.codomain.randn()
data = exact_data + noise

# Image-reconstruction using the IRGNM method
setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)
init_vec = np.ones_like(exact_solution)

solver = IrgnmCG(
    setting, data, regpar=10, regpar_step = 2/3, init = init_vec, 
    inner_it_logging_level=logging.DEBUG
    )
stoprule = (
    rules.CountIterations(max_iterations=30) +
    rules.Discrepancy(
        setting.Hcodomain.norm,
        data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=1.1
    )
)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs[0,0].set_title('Exact solution (abs)')
im = axs[0,0].imshow(mask*np.abs(exact_solution))
fig.colorbar(im,ax=axs[0,0])
axs[0,1].set_title('Exact solution (phase)')
im = axs[0,1].imshow(mask*np.angle(exact_solution))
fig.colorbar(im,ax=axs[0,1])

data_comp = op.codomain.split(data)
fig2, axs2 = plt.subplots(2, len(data_comp), sharex=True, sharey=True)
for j in range(len(data_comp)):
    im = axs2[0,j].imshow(data_comp[j])
    fig2.colorbar(im,ax=axs2[0,j])
    axs2[0,j].set_title('Simulated data')

#reco, reco_data = solver.run(stoprule)
for reco, reco_data in solver.until(stoprule):    
    Newton_step = solver.iteration_step_nr  
    # Plot reults
    if Newton_step%2 == 0:
        axs[1,0].set_title('Reco abs, step {}'.format(Newton_step))
        im = axs[1,0].imshow(mask*np.abs(reco))
        fig.colorbar(im,ax=axs[1,0])
        axs[1,1].set_title('Reco phase, step {}'.format(Newton_step))
        im = axs[1,1].imshow(mask*np.angle(reco))
        fig.colorbar(im,ax=axs[1,1])

        reco_data_comp = op.codomain.split(reco_data)
        for j in range(len(data_comp)):
            im = axs2[1,j].imshow(reco_data_comp[j])
            fig2.colorbar(im,ax=axs2[1,j])
            axs2[1,j].set_title('reconstructed data step {}'.format(Newton_step))
    plt.show(block=False)
    plt.pause(0.1)