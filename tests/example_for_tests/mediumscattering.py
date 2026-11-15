"""Regression tests for the masked-domain medium-scattering example.

The ``test_mediumscattering`` function is imported by ``test_examples.py``.
Keep the lightweight Krylov checks in that function so they are exercised by
the existing test runner without changing its imports.
"""

import io
import logging
import re
import sys

import numpy as np

from regpy.hilbert import Hm0, L2
from regpy.solvers import Setting
from regpy.vecsps.maskable import MaskedVectorSpace
import regpy.util as util

from . import import_example_package


util.set_rng_seed(15873098306879350073259142812684978477)

import_example_package("./examples/medium_scattering/")
from mediumscattering import MediumScattering  # noqa: E402

sys.path.pop(0)

def test_mediumscattering():
    radius = 1
    scattering = MediumScattering(
        gridshape=(64, 64),
        radius=radius,
        wave_number=1,
        inc_directions=util.linspace_circle(16),
        farfield_directions=util.linspace_circle(16),
        inner_solver="gmres",
        restart=30,
        block_size=1,
        logging_level=logging.WARNING,
    )

    # The forward operator stores one complex coefficient per supported grid
    # point. Spatial coordinates belong to the *full* computational grid.
    contrast = scattering.domain.zeros()
    distances = scattering.grid.coord_distances()[scattering.support]
    assert contrast.size == np.count_nonzero(scattering.support)
    inside = distances < radius
    contrast[inside] = np.exp(-1 / (radius**2 - distances[inside]**2))

    exact_data = scattering(contrast)

    # Hm0 is registered on masked spaces and acts directly on the reduced
    # coefficients, with zero extension outside the support.
    h_domain = Hm0(scattering.domain, index=2, h="physical")
    assert h_domain.vecsp == scattering.domain
    assert h_domain.gram.domain == scattering.domain
    assert h_domain.gram_inv.domain == scattering.domain

    setting = Setting(
        op=scattering,
        penalty=h_domain,
        data_fid=L2,
        exact_data=exact_data,
        init=scattering.domain.zeros(),
    )
    setting.add_Gaussian_noise(
        relative_noise_level=0.1,
        white_noise=False,
        seed=42,
    )

    setting.set_method_params("IRGN_CG", {"regpar": 1e-4, "regpar_step": 2 / 3})
    setting.run("IRGN_CG", max_it=30)

    assert setting.get_stoprule("IRGN_CG").main_rule.converged
