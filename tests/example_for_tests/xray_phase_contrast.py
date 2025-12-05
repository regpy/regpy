import logging
import sys

import numpy as np
from scipy.datasets import ascent

from regpy.vecsps import UniformGridFcts
from regpy.hilbert import L2
from regpy.solvers import Setting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
import regpy.stoprules as rules
import regpy.util as util

util.set_rng_seed(15873098306879350073259142812684978477)

from . import import_example_package

import_example_package("./examples/xray_phase_contrast/")

from xray_phase_contrast_operator import get_xray_phase_contrast

def test_xray_phase_contrast():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
    )


    # Example parameters
    fresnel_number = 5e-4    # Fresnel-number of the simulated imaging system, associated with the unit-lengthscale
                            # in grid (i.e. with the size of one pixel for the above choice of grid)
    noise_level = 0.01      # Noise level in the simulated data


    # Uniform grid of unit-spacing
    grid = UniformGridFcts(np.arange(1024), np.arange(1024))

    # Forward operator
    op = get_xray_phase_contrast(grid, fresnel_number)

    # Create phantom phase-image (= padded example-image)
    exact_solution = ascent().astype(np.float64)
    exact_solution /= exact_solution.max()
    pad_amount = tuple([(grid.shape[0] - exact_solution.shape[0])//2,
                        (grid.shape[1] - exact_solution.shape[1])//2])
    exact_solution = np.pad(exact_solution, pad_amount, 'constant', constant_values=0)

    # Create exact and noisy data
    exact_data = op(exact_solution)
    noise = noise_level * op.codomain.randn()
    data = exact_data + noise

    # Image-reconstruction using the IRGNM method
    setting = Setting(op=op, penalty=L2, data_fid=L2)
    solver = IrgnmCG(setting, data, regpar=10)
    stoprule = (
        rules.CountIterations(max_iterations=100) +
        rules.Discrepancy(
            setting.h_codomain.norm,
            data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=1.1
        )
    )

    reco, reco_data = solver.run(stoprule)

    assert stoprule.rules[1].triggered

sys.path.pop(0)
