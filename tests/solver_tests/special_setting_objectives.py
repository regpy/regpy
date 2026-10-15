"""Public objective contract for Huber_XSetting.

FISTA/SSNewton stage transitions and default stopping rules are intentionally
not tested here: those are solver policies rather than mathematical outputs.
"""

import numpy as np
import pytest

from regpy.functionals import Huber, L1
from regpy.hilbert import L2
from regpy.operators import Identity
from regpy.solvers.special_settings import Huber_XSetting
from regpy.vecsps import UniformGridFcts


@pytest.mark.parametrize("sigma", [0., 0.05], ids=["L1", "Huber"])
def test_huber_x_setting_has_expected_data_fidelity(sigma):
    grid = UniformGridFcts((0., 1., 9))
    trial = 0.35 + 0.1 * np.cos(2 * np.pi * grid.axes[0])
    observed = 0.3 + 0.05 * np.sin(2 * np.pi * grid.axes[0])
    setting = Huber_XSetting(
        Identity(grid), data=observed, regpar=0.1,
        huber_sigma=sigma, penalty=L2,
    )
    raw = Huber(grid, sigma=sigma, L1_type=True) if sigma else L1(grid)
    expected = raw.as_data_func(observed)(trial)
    np.testing.assert_allclose(setting.data_fid(trial), expected,
                               rtol=1e-10, atol=1e-10)
