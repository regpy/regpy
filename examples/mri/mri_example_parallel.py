import logging

import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import numpy as np

import regpy.stoprules as rules
import regpy.util as util
from examples.mri.mri import full_parallel_mri_parallelized,normalize
from regpy.operators.parallel_operators import ParallelExecutionManager
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
from regpy.vecsps import UniformGridFcts
from regpy.hilbert import L2,Sobolev
from regpy.util.operator_tests import test_adjoint,test_derivative

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

grid = UniformGridFcts((-1, 1, 100), (-1, 1, 100), dtype=complex)

sobolev_index = 32
noiselevel = 0.05

# In real applications with data known before constructing the operator, estimate_sampling_pattern
# can be used to determine the mask.
mask = grid.zeros(dtype=bool)
mask[::2] = True
mask[:10] = True
mask[-10:] = True

with ParallelExecutionManager():
    mri_op,op=full_parallel_mri_parallelized(grid,ncoils=10,mask=mask,sobolev_index=sobolev_index,smoothing_factor=220)
    val,deriv=mri_op.linearize(mri_op.domain.ones())
    #print(test_adjoint(deriv))
    #print(smoothed_op(smoothed_op.domain.zeros()))
    # full_mri_op = parallel_mri(grid=grid, ncoils=10)
    # sampling = cartesian_sampling(full_mri_op.codomain, mask=mask)
    # mri_op = sampling * full_mri_op

    # # Substitute Sobolev weights into coil profiles
    # smoother = sobolev_smoother(mri_op.domain, sobolev_index, factor=220.)
    # smoothed_op = mri_op * smoother
    #print(test_derivative(mri_op))


    exact_solution = mri_op.domain.zeros()
    exact_solution_elms=mri_op.domain.split(exact_solution) # returns views into exact_solution in this case
    # Exact density is just a square shape
    exact_solution_elms[0][...]=(np.max(np.abs(grid.coords), axis=0) < 0.4)
    # Exact coils are Gaussians centered on points on a circle
    centers = util.linspace_circle(len(exact_solution_elms)-1) / np.sqrt(2)
    for coil, center in zip(exact_solution_elms[1:], centers):
        r = np.linalg.norm(grid.coords - center[:, np.newaxis, np.newaxis], axis=0)
        coil[...] = np.exp(-r**2 / 2)


    # Construct data (criminally), add noise
    exact_data = mri_op(exact_solution)
    data = exact_data + noiselevel * mri_op.codomain.randn()

    # Initial guess: constant density, zero coils
    init = mri_op.domain.zeros()
    init_elms=mri_op.domain.split(init)
    init_elms[0][...]=1

    setting = RegularizationSetting(op=mri_op, penalty=Sobolev(index=2), data_fid=L2)

    solver = IrgnmCG(
        setting=setting,
        data=data,
        regpar=10,
        regpar_step=0.8,
        init=init
    )

    # stoprule = (
    #     rules.CountIterations(max_iterations=100) +
    #     rules.Discrepancy(
    #         setting.h_codomain.norm, data,
    #         noiselevel=setting.h_codomain.norm(exact_data - data),
    #         tau=1.1
    #     )
    # )
    stoprule=rules.CountIterations(max_iterations=100)

    # Plotting setup
    plt.ion()
    fig, axes = plt.subplots(ncols=2, constrained_layout=True)
    bars = [cbar.make_axes(ax)[0] for ax in axes]

    axes[0].set_title('exact solution')
    axes[1].set_title('reconstruction')

    # Plot exact solution
    # rho_ex,coils_ex = normalize(*mri_op.domain.split(exact_solution))

    rho_ex=exact_solution_elms[0]

    im = axes[0].imshow(np.abs(rho_ex))
    fig.colorbar(im, cax=bars[0])

    # Run the solver, plot iterates
    for reco, reco_data in solver.until(stoprule):
        print(np.max(np.abs(exact_data-reco_data)))
        reco_elms=mri_op.domain.split(reco)
        rho=reco_elms[0]
        im = axes[1].imshow(np.abs(rho))
        bars[1].clear()
        fig.colorbar(im, cax=bars[1])
        plt.pause(0.5)

    plt.ioff()
    plt.show()
