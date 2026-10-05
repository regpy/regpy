"""Regression test for convex Tikhonov solvers.

This reproduces the final example from ``KL_with_RE_and_nonnegL2_1D(3).ipynb``:
shifted, quadratically extended Kullback--Leibler data fidelity together with
an upper-bounded relative-entropy penalty.

Run explicitly because this requested filename does not match pytest's default
``test_*.py`` discovery pattern::

    pytest convex_tikhonov_minimizers.py
"""

import numpy as np
import pytest

from regpy.functionals import KullbackLeibler, RelativeEntropy
from regpy.operators.convolution import GaussianBlur
from regpy.solvers import Setting
from regpy.vecsps import UniformGridFcts


SOLVERS = (
    pytest.param("ADMM", id="ADMM"),
    # The solver called "SS_Newton" in prose is registered as "SSN".
    pytest.param("SSN", id="SSN"),
    pytest.param("dFISTA", id="dFISTA"),
    pytest.param("dPDHG", id="dPDHG"),
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
    return make_setting(problem).run("SSN", gap_tol=GAP_TOL)[0]


@pytest.mark.parametrize("solver", SOLVERS)
def test_shifted_kl_with_constrained_relative_entropy(
    solver,
    problem,
    ss_newton_reference,
):
    """Each supported solver should reproduce the SS-Newton solution."""
    if solver == "SSN":
        reconstruction = ss_newton_reference
    else:
        reconstruction = make_setting(problem).run(solver, gap_tol=GAP_TOL)[0]

    assert np.all(np.isfinite(reconstruction))
    assert np.all(reconstruction >= -BOUND_TOL)
    assert np.all(reconstruction <= 400.0 + BOUND_TOL)

    relative_error = np.linalg.norm(reconstruction - ss_newton_reference)
    relative_error /= np.linalg.norm(ss_newton_reference)
    assert relative_error <= RELATIVE_ERROR_TOL


def test_ss_newton_two_steps_on_split_domain(problem):
    """Splitting coordinates must preserve two complete Newton iterations.

    Q = (P1, P2) maps the original domain to two complementary coordinate
    spaces. The equivalent forward operator on the direct sum is T * Q*.
    Explicit pointwise Grams preserve the original quadrature weights and
    keep the component SquaredNorm penalties recognizable as separable.
    """
    from copy import deepcopy

    from regpy.functionals.base import FunctionalOnDirectSum, SquaredNorm
    from regpy.hilbert import GramHilbertSpace, L2
    from regpy.operators import CoordinateProjection, PtwMultiplication, VectorOfOperators
    from regpy.solvers.linear import SemiSmoothNewton
    from regpy.vecsps.maskable import MaskableDirectSum

    operator = problem["operator"]
    domain = operator.domain
    mask = (np.arange(domain.dimension).reshape(domain.shape) % 3) == 0
    p1 = CoordinateProjection(domain, mask)
    p2 = CoordinateProjection(domain, ~mask)
    split_domain = MaskableDirectSum(p1.codomain, p2.codomain)
    split = VectorOfOperators([p1, p2], codomain=split_domain)
    split_operator = operator * split.adjoint

    initial = 60.0 * domain.ones()
    reference = 200.0 * domain.ones()
    full_hilbert = L2(domain)
    diagonal = full_hilbert.gram(domain.ones())
    components = [
        SquaredNorm(
            GramHilbertSpace(PtwMultiplication(p.codomain, p(diagonal))),
            shift=p(reference),
        )
        for p in (p1, p2)
    ]
    penalty = SquaredNorm(full_hilbert, shift=reference)
    split_penalty = FunctionalOnDirectSum(components, domain=split_domain)
    settings = [
        Setting(
            op=op,
            penalty=reg,
            data_fid=deepcopy(problem["data_fidelity"]),
            regpar=1e-3,
        )
        for op, reg in ((operator, penalty), (split_operator, split_penalty))
    ]

    # Verify that we compare the same objective and geometry.
    probe = initial + 10.0 * np.sin(np.arange(domain.dimension)).reshape(domain.shape)
    np.testing.assert_allclose(split.adjoint(split(probe)), probe)
    np.testing.assert_allclose(split_operator(split(probe)), operator(probe))
    assert split_penalty(split(probe)) == pytest.approx(penalty(probe), rel=1e-12)

    def make_solver(setting, init):
        return SemiSmoothNewton(
            setting,
            init=init,
            psi_minus=0.0,
            psi_plus=120.0,
            # Identical, accurate inner solves isolate representation errors.
            cg_reltols=[1e-10, 1e-10],
            logging_level="WARNING",
            cg_logging_level="WARNING",
        )

    natural = make_solver(settings[0], initial.copy())
    blocked = make_solver(settings[1], split(initial))
    np.testing.assert_allclose(split.adjoint(blocked.x), natural.x)
    for step in (1, 2):
        assert natural.next(), f"Natural solver stopped before step {step}"
        assert blocked.next(), f"Split solver stopped before step {step}"
        assert natural.iteration_step_nr == blocked.iteration_step_nr == step
        reconstructed = split.adjoint(blocked.x)
        assert np.all(np.isfinite(reconstructed))
        np.testing.assert_allclose(
            reconstructed, natural.x, rtol=1e-7, atol=1e-8,
            err_msg=f"Different SSN domain iterates at step {step}",
        )
        np.testing.assert_allclose(
            blocked.y, natural.y, rtol=1e-7, atol=1e-8,
            err_msg=f"Different SSN images at step {step}",
        )

    # Guard against a vacuous comparison of unchanged initial guesses.
    assert np.linalg.norm(natural.x - initial) > 1e-6
