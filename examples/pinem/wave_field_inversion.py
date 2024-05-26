from regpy.solvers.nonlinear.irgnm import IrgnmCG

from regpy.operators import CoordinateProjection
from operators import get_wave_field_reco
from regpy.hilbert import L2, HmDomain
from regpy.vecsps import UniformGridFcts
from regpy.solvers import RegularizationSetting
import regpy.stoprules as rules

import numpy as np
from scipy.datasets import ascent
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

r"""Shows inversion of imaging system with modeled by Fresnel propagator. The (complex) image \(x\) is reconstructed from
 \(|\mathcal{D}_{+N}(x)|^{2}) and \(|\mathcal{D}_{-N}(x)|^{2}) where \(\mathcal{D}\) is the Fresnel propagator.
The reconstruction is done using the iteratively regularized Gauss-Newton method.
"""


# Example parameters
fresnel_number = 5e2   # Fresnel-number of the simulated imaging system, associated with the unit-lengthscale

# in grid (i.e. with the size of one pixel for the above choice of grid)
noise_level = 0.001       # Noise level in the simulated data
intensity = 1e3
# sol_type = 'phase'
# sol_type = 'modulus'
sol_type = None

# define grid
Xdim = 256
Y_dim = 256
grid = UniformGridFcts(np.linspace(0, 1, Xdim, endpoint=False),
                    np.linspace(0, 1, Y_dim, endpoint=False)).real_space()
cgrid = grid.complex_space()
[Xco, Yco] = np.meshgrid(np.arange(-1, 1, 2/Xdim), np.arange(-1, 1, 2/Y_dim))
mask = (abs(Xco+0.2) <= 0.2) & (abs(Yco) <= 0.4)
mask = mask | (abs((Xco-0.35)*(Xco-0.35)+(Yco-0.35)*(Yco-0.35)) <= 0.01)

# Forward operator and its domain
op = get_wave_field_reco(cgrid, fresnel_number, mask.astype(float), sol_type,parallel=False)  

if sol_type == None:
    projection = CoordinateProjection(cgrid,mask)
    h_domain =  HmDomain(cgrid,mask,dtype=complex,index=1)
else:
    projection = CoordinateProjection(grid,mask)
    h_domain = HmDomain(grid,mask,index=1)
embedding = projection.adjoint
op = op*embedding

# Create phantom image (= padded example-image)
picture = ascent()
exact_solution = picture[-Xdim//2:, -Y_dim//2:].astype(np.float64)/255
if sol_type is None:
    exact_solution = exact_solution + 0.3j*2*np.pi * \
        picture[:Xdim//2, :Y_dim//2].astype(np.float64)/255
pad_amount = tuple([(grid.shape[0] - exact_solution.shape[0])//2,
                (grid.shape[1] - exact_solution.shape[1])//2])
exact_solution = np.pad(exact_solution, pad_amount, 'constant', constant_values=0)
exact_solution = exact_solution * mask  # - 4*(1-mask)

# Create exact data and Poisson data
exact_data = op(projection(exact_solution))
data = np.random.poisson(intensity * exact_data)/intensity

# define codomain Gram matrix based on observed data to approximate log-likelihood
h_codomain0 = L2(grid, weights=(1+intensity*data[0])/intensity)
h_codomain1 = L2(grid, weights=(1+intensity*data[1])/intensity)  
h_codomain2 = L2(grid, weights=(1+intensity*data[2])/intensity)
h_codomain = h_codomain0+h_codomain1+h_codomain2

# Image reconstruction using the IRGNM method
setting = RegularizationSetting(op=op,penalty=h_domain,data_fid=h_codomain)

init_vec = np.zeros_like(projection(exact_solution))

solver = IrgnmCG(
    setting, data, regpar=0.1, regpar_step=2/3, init=init_vec,
    inner_it_logging_level=logging.INFO
)
stoprule = (
    rules.CountIterations(max_iterations=100) +
    rules.Discrepancy(
        setting.h_codomain.norm,
        data,
        noiselevel=setting.h_codomain.norm(np.sqrt(data/intensity)),
        tau=1
    )
)

# plot exact solution
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs[0, 0].set_title('Exact solution (abs)')
im = axs[0, 0].imshow(np.abs(exact_solution), interpolation='nearest')
fig.colorbar(im, ax=axs[0, 0])
axs[0, 1].set_title('Exact solution (phase)')
im = axs[0, 1].imshow(np.angle(exact_solution), cmap='twilight', interpolation='nearest')
fig.colorbar(im, ax=axs[0, 1])

# plot data
data_comp = op.codomain.split(data)
fig2, axs2 = plt.subplots(2, len(data_comp), sharex=True, sharey=True)
for j in range(len(data_comp)):
    im = axs2[0, j].imshow(data_comp[j], interpolation='nearest')
    fig2.colorbar(im, ax=axs2[0, j])
    axs2[0, j].set_title('Simulated data')

# perform reconstruction    
for reco, reco_data in solver.until(stoprule):
    newton_step = solver.iteration_step_nr
    ereco = embedding(reco)
    reco_error = ereco-exact_solution
    print('rel. reconstruction errors step {}: modulus: {:1.4f}, phase: {:1.4f}'.format(
        newton_step,
        np.linalg.norm(reco_error.real)/np.linalg.norm(exact_solution.real),
        np.linalg.norm(reco_error.imag)/np.linalg.norm(exact_solution.imag)))
    # Plot results
    if newton_step % 5 == 0 or stoprule.triggered:
        axs[1, 0].set_title('Reco abs, step {}'.format(newton_step))
        im = axs[1, 0].imshow(np.abs(ereco), interpolation='nearest')
        if newton_step == 5:
            fig.colorbar(im, ax=axs[1, 0])
        axs[1, 1].set_title('Reco phase, step {}'.format(newton_step))
        im = axs[1, 1].imshow(np.angle(ereco), cmap='twilight', interpolation='nearest')
        if newton_step == 5:
            fig.colorbar(im, ax=axs[1, 1])

        reco_data_comp = op.codomain.split(reco_data)
        for j in range(len(data_comp)):
            im = axs2[1, j].imshow(reco_data_comp[j], interpolation='nearest')
            if newton_step == 5:
                fig2.colorbar(im, ax=axs2[1, j])
            axs2[1, j].set_title('reconstructed data step {}'.format(newton_step))
        plt.show(block=False)
        plt.pause(0.1)
plt.show(block=True)
