import logging

import matplotlib.pyplot as plt
import matplotlib as mplib
from matplotlib.colors import hsv_to_rgb
import numpy as np
from scipy.io import loadmat

import regpy.stoprules as rules
import regpy.util as uti
from examples.mri.mri import cartesian_sampling, normalize, parallel_mri, sobolev_smoother, estimate_sampling_pattern
from regpy.operators import PtwMultiplication
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
from regpy.vecsps import UniformGridFcts
from regpy.hilbert import L2
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

# ### Complex to rgb conversion
# 
# Converts array of complex numbers into array of RGB color values for plotting. The hue corresponds to the argument.
# The brighntess corresponds to the absolute value.  

def complex_to_rgb(z):
    HSV = np.dstack( (np.mod(np.angle(z)/(2.*np.pi),1), 1.0*np.ones(z.shape), np.abs(z)/np.max((np.abs(z[:]))), ))
    return hsv_to_rgb(HSV)

# ### Load data from file and estimate sampling pattern

data = loadmat('data/ksp3x2.mat')['Y']
data = np.transpose(data,(2,0,1))*(100/np.linalg.norm(data))
# normalize and transpose data 
nrcoils,n1,n2 = data.shape
grid = UniformGridFcts((-1, 1, n1), (-1, 1, n2), dtype=complex)
mask = estimate_sampling_pattern(data)
plt.imshow(mask.T); plt.title('Undersampling pattern of data')

# ### Set up forward operator

sobolev_index = 32

full_mri_op = parallel_mri(grid=grid, ncoils=nrcoils,centered=True)
sampling = PtwMultiplication(full_mri_op.codomain,(1.+0j)* mask)
smoother = sobolev_smoother(full_mri_op.domain, sobolev_index, factor=220.)

parallel_mri_op = sampling * full_mri_op * smoother

# ### Set up initial guess
# We use constant density and zero coil profiles as initial guess.

init = parallel_mri_op.domain.zeros()
init_density, _ = parallel_mri_op.domain.split(init)
init_density[...] = 1

# ### Set up regularization method

setting = RegularizationSetting(op=parallel_mri_op, penalty=L2, data_fid=L2)

solver = IrgnmCG(
    setting=setting,
    data=data,
    regpar=1,
    regpar_step=1/3.,
    init=init
)

stoprule = rules.CountIterations(max_iterations=5) 

# ### Run solver by hand and plot iterates
# Get an iterator from the solver

it = iter(solver)

for reco, reco_data in solver.while_(stoprule):
    rho, coils = smoother.codomain.split(smoother(reco))
    #rho, coils = normalize(rho,coils)

    fig = plt.figure(figsize = (15,9))

    gs = fig.add_gridspec(3,7)
    axs = [fig.add_subplot(gs[0:3, 0:3])]
    axs[0].imshow(np.abs(rho),cmap=mplib.colormaps['Greys_r'],origin='lower')
    axs[0].xaxis.set_ticklabels([])
    axs[0].yaxis.set_ticklabels([])
    for j in range(3):
        for k in range(3,7):
            axs.append(fig.add_subplot(gs[j,k]))
            axs[-1].xaxis.set_ticklabels([])
            axs[-1].yaxis.set_ticklabels([])
    for j in range(nrcoils):
        axs[1+j].imshow(complex_to_rgb(coils[j,:,:]),origin='lower')
    plt.show()


