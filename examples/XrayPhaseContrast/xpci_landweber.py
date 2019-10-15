from regpy.operators.fresnel import xray_phase_contrast
from regpy.hilbert import L2
from regpy.discrs import UniformGrid
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber
import regpy.stoprules as rules

import numpy as np
from scipy.misc import ascent
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s')


# Example parameters
fresnelNumber = 5e-4    # Fresnel-number of the simulated imaging system, associated with the unit-lengthscale
                        # in grid (i.e. with the size of one pixel for the above choice of grid)
noise_level = 0.01      # Noise level in the simulated data
max_iterations = 100    # Maximum number of Landweber iterations performed to reconstruct the image


# Uniform grid of unit-spacing
grid = UniformGrid(np.arange(1024), np.arange(1024))

# Forward operator
op = xray_phase_contrast(grid, fresnelNumber)

# Create phantom phase-image (= padded example-image)
exact_solution = ascent().astype(np.float64)
exact_solution /= exact_solution.max()
pad_amount = tuple([(grid.shape[0] - exact_solution.shape[0])//2, (grid.shape[1] - exact_solution.shape[1])//2])
exact_solution = np.pad(exact_solution, pad_amount, 'constant', constant_values=0)

# Create exact and noisy data
exact_data = op(exact_solution)
noise = noise_level * op.codomain.randn()
data = exact_data + noise


# Image-reconstruction using Landweber's method
setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)
initial_guess = grid.zeros()

landweber = Landweber(setting, data, initial_guess, stepsize=0.25)
stoprule = (rules.CountIterations(max_iterations) +
            rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.1))

reco, reco_data = landweber.run(stoprule)


# Plot reults
plt.figure()
plt.title('Exact solution (phase-image)')
plt.imshow(exact_solution)
plt.colorbar()

plt.figure()
plt.title('Simulated data (hologram)')
plt.imshow(data)
plt.colorbar()

plt.figure()
plt.title('Reconstruction (phase-image)')
plt.imshow(reco)
plt.colorbar()

plt.show()
