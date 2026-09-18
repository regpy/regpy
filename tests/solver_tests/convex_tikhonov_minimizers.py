"""Regression test for convex Tikhonov solvers.

This reproduces the final example from ``KL_with_RE_and_nonnegL2_1D(3).ipynb``:
shifted, quadratically extended Kullback--Leibler data fidelity together with
an upper-bounded relative-entropy penalty.

Run explicitly because this requested filename does not match pytest's default
``test_*.py`` discovery pattern::

    pytest convex_tikhonov.py
"""

import numpy as np
import pytest

from regpy.functionals import KullbackLeibler, RelativeEntropy
from regpy.operators.convolution import GaussianBlur
from regpy.solvers import Setting
from regpy.vecsps import UniformGridFcts


SOLVERS = (
    pytest.param("ADMM", id="ADMM"),
    # The solver called "SS_Newton" in prose is registered as "SSNewton".
    pytest.param("SSNewton", id="SS_Newton"),
    pytest.param("dual_FISTA", id="dual_FISTA"),
    pytest.param("dual_PDHG", id="dual_PDHG"),
)

RELATIVE_ERROR_TOL = 1e-2
GAP_TOL = 5e-3
BOUND_TOL = 1e-6


@pytest.fixture(scope="module")
def problem():
    """Construct the deterministic inverse problem from the notebook."""
    grid = UniformGridFcts((0, 1, 512))
    coordinate = grid.coords[0]

    exact_solution = (
        20.0
        + 110.0 * ((0.12 <= coordinate) & (coordinate <= 0.30))
        + 170.0 * np.exp(-((coordinate - 0.52) / 0.055) ** 2)
        + 100.0
        * np.maximum(1.0 - np.abs(coordinate - 0.78) / 0.10, 0.0)
    )

    operator = GaussianBlur(grid, 0.05)
    expected_data = operator(exact_solution)
    expected_data[expected_data < 0] = 0.0

    rng = np.random.default_rng(20260908)
    data = rng.poisson(expected_data)

    sigma = 1.0
    data_fidelity = KullbackLeibler(
        grid,
        w=sigma,
        data=data,
        quad_taylor_l=sigma,
    ).shift(-sigma)

    initial_guess = 200.0 * grid.ones()
    penalty = RelativeEntropy(grid, w=initial_guess, constr_u=400)

    return {
        "operator": operator,
        "penalty": penalty,
        "data_fidelity": data_fidelity,
    }


def make_setting(problem):
    """Return a fresh setting so that solver runs cannot share state."""
    sett =  Setting(
        op=problem["operator"],
        penalty=problem["penalty"],
        data_fid=problem["data_fidelity"],
        regpar=1e-1,
    )
    sett.set_stoprule_params(list(sett.applicable_methods().keys()),{'reltol':1e-2})
    return sett

@pytest.fixture(scope="module")
def ss_newton_reference(problem):
    """Compute the reference reconstruction used in the notebook."""
    return make_setting(problem).run("SSNewton", gap_tol=GAP_TOL)[0]


@pytest.mark.parametrize("solver", SOLVERS)
def test_shifted_kl_with_constrained_relative_entropy(
    solver,
    problem,
    ss_newton_reference,
):
    """Each supported solver should reproduce the SS-Newton solution."""
    if solver == "SSNewton":
        reconstruction = ss_newton_reference
    else:
        reconstruction = make_setting(problem).run(solver, gap_tol=GAP_TOL)[0]

    assert np.all(np.isfinite(reconstruction))
    assert np.all(reconstruction >= -BOUND_TOL)
    assert np.all(reconstruction <= 400.0 + BOUND_TOL)

    relative_error = np.linalg.norm(reconstruction - ss_newton_reference)
    relative_error /= np.linalg.norm(ss_newton_reference)
    assert relative_error <= RELATIVE_ERROR_TOL
