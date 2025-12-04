import logging
import sys

import numpy as np
from scipy.io import loadmat

import regpy.stoprules as rules

from regpy.operators import PtwMultiplication
from regpy.solvers import Setting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
from regpy.vecsps import UniformGridFcts
from regpy.hilbert import L2
import regpy.util as util

util.set_rng_seed(15873098306879350073259142812684978477)


from . import import_example_package

import_example_package("./examples/mri/")

from mri import parallel_mri, sobolev_smoother, estimate_sampling_pattern


def test_mri():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
    )

    # ### Complex to rgb conversion
    # 
    # Converts array of complex numbers into array of RGB color values for plotting. The hue corresponds to the argument.
    # The brighntess corresponds to the absolute value.  


    # ### Load data from file and estimate sampling pattern

    data = loadmat('examples/mri/data/ksp3x2.mat')['Y']
    data = np.transpose(data,(2,0,1))*(100/np.linalg.norm(data))
    # normalize and transpose data 
    nrcoils,n1,n2 = data.shape
    grid = UniformGridFcts((-1, 1, n1), (-1, 1, n2), dtype=complex)
    mask = estimate_sampling_pattern(data)

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

    setting = Setting(op=parallel_mri_op, penalty=L2, data_fid=L2)

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

    
    for reco, reco_data in solver.while_(stoprule):
        rho, coils = smoother.codomain.split(smoother(reco))
        #rho, coils = normalize(rho,coils)

sys.path.pop(0)
