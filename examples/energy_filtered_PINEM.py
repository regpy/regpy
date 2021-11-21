from scipy.sparse import linalg
from regpy.solvers.irgnm import IrgnmCG

from regpy.operators.PINEM import wave_field_reco_PINEM
from regpy.hilbert import L2, Sobolev
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
fresnelNumber = 5e-3    # Fresnel-number of the simulated imaging system, associated with the unit-lengthscale
                        # in grid (i.e. with the size of one pixel for the above choice of grid)
noise_level = 0.001       # Noise level in the simulated data
intensity = 1e1

# Uniform grid
Xdim= 256; Ydim= 256
grid = UniformGrid(np.linspace(0,1,Xdim,endpoint=False), np.linspace(0,1,Ydim,endpoint=False)).real_space()
cgrid = grid.complex_space()
[Xco,Yco] = np.meshgrid(np.arange(-1,1,2/Xdim),np.arange(-1,1,2/Ydim))
mask = (abs(Xco+0.2)<=0.2) & (abs(Yco)<=0.4)
mask = mask | (abs((Xco-0.35)*(Xco-0.35)+(Yco-0.35)*(Yco-0.35))<=0.01)

# Forward operator
op = wave_field_reco_PINEM(cgrid, fresnelNumber,mask.astype(complex))

# Create phantom phase-image (= padded example-image)
picture = ascent()
exact_solution = picture[-Xdim//2:,-Ydim//2:].astype(np.float64)/255 \
    + 0.3j*2*np.pi*picture[:Xdim//2,:Ydim//2].astype(np.float64)/255 
pad_amount = tuple([(grid.shape[0] - exact_solution.shape[0])//2, (grid.shape[1] - exact_solution.shape[1])//2])
exact_solution = np.pad(exact_solution, pad_amount, 'constant', constant_values=0)
exact_solution = exact_solution.astype(complex) * mask - 4*(1-mask)

# Create exact and noisy data
exact_data = op(exact_solution)
#noise = noise_level * op.codomain.randn()
#data = exact_data + noise
data = np.random.poisson(intensity * exact_data)/intensity
data_comp = op.codomain.split(data)
noise = data-exact_data

# Image-reconstruction using the IRGNM method
Hdomain = Sobolev(cgrid, index=0.5)
Hcodomain0 = L2(grid, weights=(1+intensity*data[0])/intensity)
Hcodomain1 = L2(grid, weights=(1+intensity*data[1])/intensity)
Hcodomain2 = L2(grid, weights=(1+intensity*data[2])/intensity)
Hcodomain=Hcodomain0+Hcodomain1+Hcodomain2
setting = HilbertSpaceSetting(
    op=op, Hdomain=Hdomain, 
    Hcodomain=Hcodomain)
init_vec = np.zeros_like(exact_solution)
#init_vec = np.zeros_like(exact_solution.real)

solver = IrgnmCG(
    setting, data, regpar=10, regpar_step = 2/3, init = init_vec, 
    inner_it_logging_level=logging.INFO
    )
stoprule = (
    rules.CountIterations(max_iterations=100) +
    rules.Discrepancy(
        setting.Hcodomain.norm,
        data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau= 1
    )
)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs[0,0].set_title('Exact solution (abs)')
im = axs[0,0].imshow(np.real(exact_solution))
fig.colorbar(im,ax=axs[0,0])
axs[0,1].set_title('Exact solution (phase)')
im = axs[0,1].imshow(np.imag(exact_solution))
fig.colorbar(im,ax=axs[0,1])

fig2, axs2 = plt.subplots(2, len(data_comp), sharex=True, sharey=True)
for j in range(len(data_comp)):
    im = axs2[0,j].imshow(data_comp[j])
    fig2.colorbar(im,ax=axs2[0,j])
    axs2[0,j].set_title('Simulated data')

#reco, reco_data = solver.run(stoprule)
for reco, reco_data in solver.until(stoprule):    
    Newton_step = solver.iteration_step_nr  
    reco_error = reco-exact_solution
    print('rel. reconstruction errors step {}: modulus: {:1.4f}, phase: {:1.4f}'.format(
        Newton_step, 
        np.linalg.norm(reco_error.real)/np.linalg.norm(exact_solution.real),
        np.linalg.norm(reco_error.imag)/np.linalg.norm(exact_solution.imag)))
    # Plot reults
    if Newton_step%5 == 0:
        axs[1,0].set_title('Reco abs, step {}'.format(Newton_step))
        im = axs[1,0].imshow(np.real(reco))
        if Newton_step==5:
            fig.colorbar(im,ax=axs[1,0])
        axs[1,1].set_title('Reco phase, step {}'.format(Newton_step))
        im = axs[1,1].imshow(np.imag(reco))
        if Newton_step==5:
            fig.colorbar(im,ax=axs[1,1])

        reco_data_comp = op.codomain.split(reco_data)
        for j in range(len(data_comp)):
            im = axs2[1,j].imshow(reco_data_comp[j])
            if Newton_step==5:
                fig2.colorbar(im,ax=axs2[1,j])
            axs2[1,j].set_title('reconstructed data step {}'.format(Newton_step))
    plt.show(block=False)
    plt.pause(0.1)
plt.show(block=True)