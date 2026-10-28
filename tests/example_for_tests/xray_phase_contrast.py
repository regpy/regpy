"""Regression test for the X-ray phase-contrast tutorial notebook."""

import sys

import numpy as np
from scipy.datasets import ascent

from regpy.vecsps import UniformGridFcts
from regpy.hilbert import L2
from regpy.operators import FourierInterpolationOperator
from regpy.solvers import Setting
from regpy.util import set_rng_seed

from . import import_example_package

import_example_package("./examples/xray_phase_contrast/")
try:
    from xray_phase_contrast_operator import get_xray_phase_contrast
finally:
    sys.path.pop(0)


def test_xray_phase_contrast():
    """IRGNM reaches the notebook's discrepancy threshold within 30 steps."""
    fresnel_number = 5e-4
    noise_level = 0.01

    # Downsample the phantom as in the notebook, then pad each axis by 25%.
    image = ascent().astype(np.float64)
    image_grid = UniformGridFcts(*image.shape)
    trunc_op = FourierInterpolationOperator(
        image_grid, Fourier_truncation_amount=160,
    )
    exact_solution = trunc_op(image)
    exact_solution /= exact_solution.max()
    padding = tuple((n // 4, n // 4) for n in exact_solution.shape)
    exact_solution = np.pad(exact_solution, padding, mode="constant")

    # Construct the operator after resizing so its domain matches the image.
    grid = UniformGridFcts(*exact_solution.shape)
    op = get_xray_phase_contrast(grid, fresnel_number)
    assert exact_solution in op.domain
    exact_data = op(exact_solution)

    setting = Setting(op, L2, L2, exact_data=exact_data)
    set_rng_seed(20261011)
    setting.add_Gaussian_noise(
        relative_noise_level=noise_level, white_noise=False,
    )

    # Exercise the method registry and its default stopping-rule factory,
    # exactly as in the saved notebook.
    reco, reco_data = setting.run('IRGN_CG', max_it=30)
    stoprule = setting.get_stoprule('IRGN_CG')

    assert stoprule.converged, stoprule.stop_reason
    assert reco in op.domain
    assert reco_data in op.codomain
    assert np.all(np.isfinite(reco))
    assert np.all(np.isfinite(reco_data))
