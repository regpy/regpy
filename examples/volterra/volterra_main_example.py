### Volterra main example file you may also checkout the Jupyter notebook volterra_main_example.ipynb 

# general import
import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)


from regpy.hilbert import L2
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.irgnm_semismooth import IrgnmSemiSmooth
import regpy.stoprules as rules

# Import operator Volterra from the definition in volterra.py and choose a grid and take the exponent to be 2 so that it is a non-linear operator. 

from volterra import Volterra
from regpy.vecsps import UniformGridFcts

grid = UniformGridFcts(np.linspace(0, 2*np.pi, 200))
op = Volterra(grid, exponent=2)

# define an exact solution and take the explicitly computed exact data. 
exact_solution = (1-np.cos(grid.coords[0]))**2/4 
exact_data = (3*grid.coords[0] - 4*np.sin(grid.coords[0]) + np.cos(grid.coords[0])*np.sin(grid.coords[0]))/8

# add noise to the exact data
noise = 0.3 * op.domain.randn()
data = exact_data + noise

# Define a regularization setting be choosing the hilbert spaces on the domain and codomain.
setting = RegularizationSetting(op, L2, L2)

# Choose an initial guess and define the solver with the appropriate parameters
init = op.domain.ones()*0.5
solver = IrgnmSemiSmooth(
    setting, # Regularization setting
    data, # noisy data constructed above
    psi_minus=0, # lower bound
    psi_plus=1.5, # upper bound
    regpar=1, # initial regularization parameter
    regpar_step=0.7, # regularization parameter step
    init=init # initial guess
)

# Define a stoprule that will stop the regularization. here it is a combination of the iteration count and a discrepancy rule. 
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(
        setting.h_codomain.norm, data,
        noiselevel=setting.h_codomain.norm(noise),
        tau=1.1
    )
)

# Run the regularization
reco, reco_data = solver.run(stoprule)

# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=1)
axs.plot(grid.coords[0],reco,color="lightblue",label="reconstruction")
axs.plot(grid.coords[0],reco_data,color="orange",label="reconstruction data")
axs.plot(grid.coords[0],exact_solution,color="blue",label="exact solution")
axs.plot(grid.coords[0], exact_data, color="green",label="exact data")
axs.plot(grid.coords[0], data, color="lightgreen",label="noisy data")
plt.show()