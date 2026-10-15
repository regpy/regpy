"""Coordinate-independent Fenchel-duality-gap consistency checks.

The value of a full primal/dual gap is a mathematical invariant, unlike the
solver's default stopping thresholds, monitoring schedules, or acceleration
choices. Do not make these checks depend on the implementation of PDHG.
"""

import numpy as np
import pytest

from regpy.hilbert import L2
from regpy.operators import Identity
from regpy.solvers import Setting
from regpy.vecsps import UniformGridFcts


@pytest.mark.parametrize("alpha", [0.001, 0.1, 2.])
def test_full_gap_is_invariant_under_primal_dual_coordinate_change(alpha):
    grid = UniformGridFcts((0., 1., 5), (0., 1., 6))
    primal = Setting(
        Identity(grid), penalty=L2, data_fid=L2,
        data=1.3 * grid.ones(), regpar=alpha,
        incorporate_init_in_penalty=False,
    )
    dual_setting = primal.get_dual_setting()

    f = 0.7 * grid.ones()
    p = 0.2 * grid.ones()
    # The second saddle variable of the transformed dual is q=-alpha*f.
    q = -alpha * f
    gap_primal = primal.duality_gap(
        primal=(f, primal.op(f)),
        dual=(p, primal.op.adjoint(p)),
    )
    gap_dual = dual_setting.duality_gap(
        primal=(p, dual_setting.op(p)),
        dual=(q, dual_setting.op.adjoint(q)),
    )
    assert np.isfinite(gap_primal)
    np.testing.assert_allclose(gap_dual, gap_primal, rtol=1e-10, atol=1e-9)
