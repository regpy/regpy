from numpy.core.numeric import ones_like
from regpy.solvers.irgnm import IrgnmCG

from regpy.operators import Exponential, Matrix_of_operators 
from regpy.operators.PINEM import Nemitzky_op_for_g, PINEM_g_to_data
from regpy.hilbert import L2, Sobolev
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
intensity = 1e5
# noise_level = 0.0001      # Noise level in the simulated data

# define grid 
Xdim= 256; Ydim= 256
grid = UniformGrid(np.linspace(0,1,Xdim,endpoint=False), np.linspace(0,1,Ydim,endpoint=False))

# define forward operator and its domain
[Xco,Yco] = np.meshgrid(np.arange(-1,1,2/Xdim),np.arange(-1,1,2/Ydim))
mask = (abs(Xco+0.2)<=0.2) & (abs(Yco)<=0.4)
mask = mask | (abs((Xco-0.35)*(Xco-0.35)+(Yco-0.35)*(Yco-0.35))<=0.01)
A_Psi0_Multiplier = np.ones(grid.shape,complex)
# Here we are weighting the penalty for the modulus a bit less
Hdomain = 0.5*Sobolev(grid, index=0.5) + Sobolev(grid, index=0.5)
op = PINEM_g_to_data(grid,fresnelNumber,mask,A_Psi0_Multiplier,N=2)

# Create phantom image (= padded example-image)
picture = ascent()
log_g = picture[-Xdim//2:,-Ydim//2:].astype(np.float64)/255 \
    * np.exp(1j*2*np.pi*picture[:Xdim//2,:Ydim//2].astype(np.float64)/255)
log_g /= 10*abs(log_g).max()
log_g += ones_like(log_g)
pad_amount = tuple([(grid.shape[0] - log_g.shape[0])//2, (grid.shape[1] - log_g.shape[1])//2])
log_g = np.pad(log_g, pad_amount, 'constant', constant_values=1)
log_g = log_g.astype(complex)*mask

# Create exact and noisy data
exact_solution = op.domain.join(np.exp(log_g.real), log_g.imag)
exact_data = op(exact_solution)
data = np.random.poisson(intensity * exact_data)/intensity

# define codomain Gram matrix based on observed data to approximate log-likelihood
Hcodomain0 = L2(grid, weights=(1+intensity*data[0])/intensity)
Hcodomain1 = L2(grid, weights=(1+intensity*data[1])/intensity)
Hcodomain=Hcodomain0+Hcodomain1

# Image-reconstruction using the IRGNM method
setting = HilbertSpaceSetting(op=op, Hdomain=Hdomain, Hcodomain=Hcodomain)
init_vec = op.domain.join(grid.ones(), grid.zeros())

solver = IrgnmCG(
    setting, data, init = init_vec,
    regpar=5e-4, regpar_step = 2/3,
    inner_it_logging_level=logging.INFO)
stoprule = (
    rules.CountIterations(max_iterations=100) +
    rules.Discrepancy(
        setting.Hcodomain.norm,
        data,
        noiselevel=setting.Hcodomain.norm(np.sqrt(data/intensity)),
        tau=1.0
    )
)

# plot exact solution and data
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs[0,0].set_title('Exact |g|')
im = axs[0,0].imshow(np.exp(log_g.real))
fig.colorbar(im,ax=axs[0,0])
axs[0,1].set_title('Exact  arg(g)')
im = axs[0,1].imshow(log_g.imag)
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
    # Print reconstruction error 
    reco_error1, reco_error2 = op.domain.split(reco-exact_solution)
    print('rel. reconstruction errors step {}: modulus: {:1.4f}, phase: {:1.4f}'.format(
        Newton_step, 
        np.linalg.norm(reco_error1)/np.linalg.norm(np.exp(log_g.real)),
        np.linalg.norm(reco_error2)/np.linalg.norm(log_g.imag)))
    # Plot reults
    if Newton_step%2 == 0 or stoprule.triggered:
        reco1,reco2=op.domain.split(reco)
        reco_data_comp = op.codomain.split(reco_data)

        axs[1,0].set_title('Reco |g|, step {}'.format(Newton_step))
        im = axs[1,0].imshow(reco1)
        if Newton_step==2:
            fig.colorbar(im,ax=axs[1,0])
        axs[1,1].set_title('Reco arg(g), step {}'.format(Newton_step))
        im = axs[1,1].imshow(reco2)
        if Newton_step==2:
            fig.colorbar(im,ax=axs[1,1])

        for j in range(len(reco_data_comp)):
            im = axs2[1,j].imshow(reco_data_comp[j])
            if Newton_step==2:
                fig2.colorbar(im,ax=axs2[1,j])
            axs2[1,j].set_title('recon. data step {}'.format(Newton_step))
        plt.show(block=False)
        plt.pause(0.1)
plt.show(block=True)