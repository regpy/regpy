import logging

import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import numpy as np

import regpy.stoprules as rules
import regpy.util as util
from examples.mri.mri import normalize, sobolev_smoother,full_parallel_mri_parallelized,DomainConversion
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
from regpy.vecsps import UniformGridFcts
from regpy.hilbert import L2
from regpy.operators.parallel_operators import ParallelExecutionManager

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
if __name__ == '__main__':
    with ParallelExecutionManager():
        full_mri_op=full_parallel_mri_parallelized(grid,ncoils=10,mask=mask)
        converter=DomainConversion(full_mri_op.domain)
        mri_op=full_mri_op*converter

        # Substitute Sobolev weights into coil profiles
        smoother = sobolev_smoother(mri_op.domain, sobolev_index, factor=220.)
        smoothed_op = mri_op* smoother

        exact_solution = mri_op.domain.zeros()
        exact_density, exact_coils = mri_op.domain.split(exact_solution)  # returns views into exact_solution in this case

        # Exact density is just a square shape
        exact_density[...] = (np.max(np.abs(grid.coords), axis=0) < 0.4)

        # Exact coils are Gaussians centered on points on a circle
        centers = util.linspace_circle(exact_coils.shape[0]) / np.sqrt(2)
        for coil, center in zip(exact_coils, centers):
            r = np.linalg.norm(grid.coords - center[:, np.newaxis, np.newaxis], axis=0)
            coil[...] = np.exp(-r**2 / 2)

        # Construct data (criminally), add noise
        exact_data = mri_op(exact_solution)
        data = exact_data + noiselevel * mri_op.codomain.randn()

        # Initial guess: constant density, zero coils
        init = smoothed_op.domain.zeros()
        init_density, _ = smoothed_op.domain.split(init)
        init_density[...] = 1

        setting = RegularizationSetting(op=smoothed_op, penalty=L2, data_fid=L2)

        solver = IrgnmCG(
            setting=setting,
            data=data,
            regpar=10,
            regpar_step=0.8,
            init=init
        )

        stoprule = (
            rules.CountIterations(max_iterations=100) +
            rules.Discrepancy(
                setting.h_codomain.norm, data,
                noiselevel=setting.h_codomain.norm(exact_data - data),
                tau=1.1
            )
        )

        # Plotting setup
        plt.ion()
        fig, axes = plt.subplots(ncols=2, constrained_layout=True)
        bars = [cbar.make_axes(ax)[0] for ax in axes]

        axes[0].set_title('exact solution')
        axes[1].set_title('reconstruction')

        # Plot exact solution
        rho_ex,coils_ex = normalize(*mri_op.domain.split(exact_solution))
        im = axes[0].imshow(np.abs(rho_ex))
        fig.colorbar(im, cax=bars[0])

        # Run the solver, plot iterates
        for reco, reco_data in solver.until(stoprule):
            rho,coils = normalize(*mri_op.domain.split(smoother(reco)))
            im = axes[1].imshow(np.abs(rho))
            bars[1].clear()
            fig.colorbar(im, cax=bars[1])
            plt.pause(0.5)

        plt.ioff()
        plt.show()
