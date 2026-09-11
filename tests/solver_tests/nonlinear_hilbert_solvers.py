"""Tests for nonlinear solvers on periodic and Gaussian deconvolution problems.

Keep this file in the same directory as ``periodic_c_problem.py``.
"""

import inspect
from itertools import combinations
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from .periodic_c_problem import PeriodicCProblem
from regpy.hilbert import Hm, L2
from regpy.solvers import Setting
from regpy.vecsps import UniformGridFcts
from regpy.operators import GaussianBlur
from regpy.solvers.linear import FISTA as LinearFISTA
from regpy.solvers.linear.spectral import Spectral_Landweber, Spectral_IteratedTikhonov
from regpy.solvers.nonlinear import (
    Landweber as NonlinearLandweber,
    FISTA as GeneralizedFISTA,
    IrgnmCG,
    LevenbergMarquardt,
)
import regpy.solvers.linear as linear_solvers
from regpy.stoprules import OptimalityCondStoppingNonConvex


@dataclass
class InverseProblem:
    operator: PeriodicCProblem
    exact_solution: Any
    exact_data: Any


@pytest.fixture
def inverse_problem() -> InverseProblem:
    n = 100
    operator = PeriodicCProblem(n=n, f=np.ones(n), k=10.0)
    points = operator.grid.coords[0]
    exact_solution = (1.0 + 0.3 * np.sin(2.0 * np.pi * points)) / (
        1.0 - 0.5 * np.cos(6.0 * np.pi * points)
    )
    exact_data = operator(exact_solution)
    return InverseProblem(operator, exact_solution, exact_data)


def _make_setting(
    problem: InverseProblem,
    *,
    codomain_free: bool,
    regpar: float | None = None,
) -> Setting:
    setting = Setting(
        problem.operator,
        Hm(index=1),
        L2,
        data=problem.exact_data,
        init=problem.operator.domain.ones(),
        codomain_free=codomain_free,
        regpar=regpar,
    )
    return setting


def _relative_domain_error(setting: Setting, actual: Any, expected: Any) -> float:
    return setting.h_domain.norm(actual - expected) / setting.h_domain.norm(
        expected
    )


def _relative_l2_error(actual: Any, expected: Any) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _relative_image_error(
    setting: Setting,
    actual: Any,
    expected: Any,
) -> float:
    return setting.h_codomain.norm(actual - expected) / setting.h_codomain.norm(
        expected
    )


def test_fista_codomain_free_matches_standard(
    inverse_problem: InverseProblem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare iteration paths with identical, fixed step lengths."""
    fixed_tau = 2e6
    estimates = []

    def fixed_stepsize(solver):
        # Finite-Lipschitz genFISTA estimates tau during construction rather
        # than using its tau argument. Replace only that estimator for this
        # equivalence test; the actual iteration code remains unchanged.
        solver.tau = fixed_tau
        estimates.append(solver)

    monkeypatch.setattr(GeneralizedFISTA, "_estimate_stepsize", fixed_stepsize)
    standard = _make_setting(
        inverse_problem,
        codomain_free=False,
        regpar=1e-10,
    )
    codomain_free = _make_setting(
        inverse_problem,
        codomain_free=True,
        regpar=1e-10,
    )

    standard_reconstruction, _ = standard.run(
        "genFISTA",
        max_it=2,
        stepsize_selection_steps=[0],
        logging_level="DEBUG",
    )
    codomain_free_reconstruction, _ = codomain_free.run(
        "genFISTA",
        max_it=2,
        stepsize_selection_steps=[0],
        logging_level="DEBUG",
    )

    assert len(estimates) == 2  # One initialization per run, no reestimation.
    assert all(solver.tau == fixed_tau for solver in estimates)

    error = _relative_domain_error(
        standard,
        standard_reconstruction,
        codomain_free_reconstruction,
    )
    assert error <= 1e-8


@pytest.mark.parametrize(
    ("method", "solver_options", "tolerance"),
    [
        pytest.param("nlLW", {"stepsize": 2e6}, 1e-8, id="landweber"),
        pytest.param("IRGNM_CG", {}, 1e-10, id="irgnm-cg"),
        pytest.param("LevMar", {}, 1e-10, id="levenberg-marquardt"),
    ],
)
def test_codomain_free_matches_standard(
    inverse_problem: InverseProblem,
    method: str,
    solver_options: dict[str, Any],
    tolerance: float,
) -> None:
    standard = _make_setting(inverse_problem, codomain_free=False)
    codomain_free = _make_setting(inverse_problem, codomain_free=True)

    standard_reconstruction, _ = standard.run(
        method,
        max_it=2,
        logging_level="DEBUG",
        **solver_options,
    )
    codomain_free_reconstruction, _ = codomain_free.run(
        method,
        max_it=2,
        logging_level="DEBUG",
        **solver_options,
    )

    error = _relative_domain_error(
        standard,
        standard_reconstruction,
        codomain_free_reconstruction,
    )
    assert error <= tolerance

@pytest.mark.parametrize(
    (
        "method",
        "max_iterations",
        "solver_options",
        "max_l2_error",
        "max_image_error",
    ),
    [
        pytest.param("nlLW", 400, {}, 1e-1, 1e-2, id="landweber"),
        pytest.param(
            "genFISTA", 100, {"stepsize_selection_steps": [0]},
            1e-1, 1e-2, id="FISTA",
        ),        
        pytest.param(
            "IRGNM_CG",
            20,
            {"regpar": 1e-9, "regpar_step": 1 / 3},
            1e-2,
            5e-4,
            id="irgnm-cg",
        ),
        pytest.param(
            "LevMar",
            15,
            {"regpar": 1e-8, "regpar_step": 1 / 2},
            1e-2,
            5e-4,
            id="levenberg-marquardt",
        ),
        pytest.param("NewtonCG", 25, {}, 2e-2, 8e-3, id="newton-cg"),
    ],
)
def test_exact_data_convergence(
    inverse_problem: InverseProblem,
    method: str,
    max_iterations: int,
    solver_options: dict[str, Any],
    max_l2_error: float,
    max_image_error: float,
) -> None:
    setting = _make_setting(inverse_problem, codomain_free=False)

    reconstruction, reconstructed_data = setting.run(
        method,
        max_it=max_iterations,
        ref_sol=inverse_problem.exact_solution,
        logging_level="WARNING",
        **solver_options,
    )
    l2_error = _relative_l2_error(
        reconstruction,
        inverse_problem.exact_solution,
    )
    image_error = _relative_image_error(
        setting,
        reconstructed_data,
        inverse_problem.exact_data,
    )

    assert l2_error <= max_l2_error
    if method in {"LevMar", "NewtonCG"}:
        assert image_error < max_image_error
    else:
        assert image_error <= max_image_error


def test_tikhonov_fista_with_optimality_stopping(
    inverse_problem: InverseProblem,
) -> None:
    """Require the optimality-rule experiment to stop before iteration 200."""
    setting = _make_setting(
        inverse_problem, codomain_free=False, regpar=1e-9,
    )
    setting.set_stoprule(
        OptimalityCondStoppingNonConvex(tol=0.1),
        method_names=["genFISTA"],
    )
    reconstruction, reconstructed_data = setting.run(
        "genFISTA", max_it=400, logging_level="DEBUG",
    )
    executed_rule = setting.get_stopping_rule("genFISTA")
    assert executed_rule.solver is not None
    iterations = executed_rule.solver.iteration_step_nr
    assert iterations < 200, f"Expected fewer than 200 iterations, got {iterations}."
    assert reconstruction in inverse_problem.operator.domain
    assert reconstructed_data in inverse_problem.operator.codomain
    assert np.isfinite(_relative_l2_error(
        reconstruction, inverse_problem.exact_solution,
    ))
    assert np.isfinite(_relative_image_error(
        setting, reconstructed_data, inverse_problem.exact_data,
    ))


# Linear operators are also admissible inputs to the nonlinear solvers.
# These tests mirror linear_vs_nonlinear.ipynb and add both codomain-free runs.
_DECONVOLUTION_CG_PARS = {
    "tol": 1e-10,
    "reltolx": 1e-10,
    "reltoly": None,
    "max_it": 1000,
}


@pytest.fixture
def deconvolution_setting_factory():
    """Create independent settings with the notebook's nonzero initial guess."""
    grid = UniformGridFcts((-1, 1, 128), periodic=True)
    points = grid.coords[0]
    exact = 1.5 + 0.4 * np.cos(np.pi * points) + 0.2 * np.sin(4 * np.pi * points)
    initial = 0.8 + 0.1 * np.cos(2 * np.pi * points)
    data = GaussianBlur(grid, sigma=0.08, self_adjoint=False)(exact).copy()

    def make_setting(*, codomain_free: bool, regpar: float | None = None):
        return Setting(
            op=GaussianBlur(grid, sigma=0.08, self_adjoint=False),
            penalty=L2,
            data_fid=L2,
            data=data.copy(),
            init=initial.copy(),
            regpar=regpar,
            codomain_free=codomain_free,
        )

    return make_setting


def _deconvolution_steps(solver: Any, number: int) -> list[tuple[Any, Any]]:
    """Copy accepted iterates; evaluate missing images only after all steps."""
    snapshots = []
    for step in range(number):
        assert solver.next(), (
            f"{type(solver).__name__} stopped before step {step + 1}"
        )
        snapshots.append((solver.x.copy(), None if solver.y is None else solver.y.copy()))
    return [
        (x, solver.op(x).copy() if y is None else y)
        for x, y in snapshots
    ]


def _assert_four_histories_agree(histories: dict, tolerance: float) -> None:
    """Compare all six pairs, including reconstructed forward data."""
    assert len(histories) == 4
    for (left_name, left), (right_name, right) in combinations(histories.items(), 2):
        assert len(left) == len(right) > 0
        for step, (left_pair, right_pair) in enumerate(zip(left, right, strict=True), 1):
            for quantity, actual, reference in zip(
                ("solution", "forward image"), left_pair, right_pair, strict=True,
            ):
                error = np.linalg.norm(actual - reference) / max(
                    np.linalg.norm(reference), np.finfo(float).tiny,
                )
                assert error <= tolerance, (
                    f"{left_name} vs {right_name}, step {step}, {quantity}: "
                    f"relative difference {error:.3e} exceeds {tolerance:.3e}"
                )


def _make_deconvolution_grid_solver(setting: Setting):
    """Support the alpha-grid interfaces used by the reference notebook."""
    grid_solver = getattr(linear_solvers, "TikhonovOnAlphaGrid", None)
    if grid_solver is None:
        grid_solver = linear_solvers.TikhonovAlphaGrid
    arguments = inspect.signature(grid_solver).parameters
    parameters = iter([1e-2, 5e-3, 2e-3])
    if "regpars" in arguments:
        options = {"regpars": parameters}
    elif "alphas" in arguments:
        options = {"alphas": parameters}
    else:
        raise TypeError(f"Unsupported parameter-sequence interface: {grid_solver}")
    if "cg_pars" in arguments:
        options["cg_pars"] = dict(_DECONVOLUTION_CG_PARS)
    else:
        if "tol_fac" in arguments:
            options["tol_fac"] = 1e-10
        elif "tol_fac_cg" in arguments:
            options["tol_fac_cg"] = 1e-10
        else:
            raise TypeError(f"Unsupported inner-CG tolerance interface: {grid_solver}")
        if "max_CG_iter" in arguments:
            options["max_CG_iter"] = 1000
    if "logging_level" in arguments:
        options["logging_level"] = "WARNING"
    return grid_solver(setting, **options)


@pytest.mark.parametrize("method_family", ["landweber", "fista", "irgnm", "levenberg-marquardt"])
def test_linear_and_nonlinear_deconvolution_agree(
    deconvolution_setting_factory,
    method_family: str,
) -> None:
    """Require linear/nonlinear and standard/codomain-free results to agree.

    Landweber, FISTA and IRGNM are compared at every requested step.
    Three Levenberg–Marquardt steps equal one spectral reconstruction with
    a threefold iterated Tikhonov filter. No outer stopping rule may truncate
    these comparisons; the Newton methods use tight inner-CG tolerances.
    """
    histories = {}
    tolerance = 1e-8 if method_family in {"landweber", "fista"} else 1e-6
    for codomain_free in (False, True):
        def setting(regpar=None):
            return deconvolution_setting_factory(
                codomain_free=codomain_free, regpar=regpar,
            )

        if method_family == "landweber":
            nonlinear = NonlinearLandweber(setting(), stepsize=1.0, logging_level="WARNING")
            linear = Spectral_Landweber(setting(), k=[1, 2, 3], stepsize=1.0)
            nonlinear_steps = linear_steps = 3
        elif method_family == "fista":
            nonlinear = GeneralizedFISTA(
                setting(1e-2), stepsize_selection_steps=[0], logging_level="WARNING",
            )
            # For finite-Lipschitz fidelity the constructor estimates tau.
            # Override it exactly as in the notebook, then reset momentum.
            nonlinear.tau = 0.9
            nonlinear._restart_three_term_recursion()
            linear = LinearFISTA(setting(1e-2), tau=0.9, logging_level="WARNING")
            nonlinear_steps = linear_steps = 30
        elif method_family == "irgnm":
            nonlinear = IrgnmCG(
                setting(), regpars=iter([1e-2, 5e-3, 2e-3]),
                cg_pars=dict(_DECONVOLUTION_CG_PARS), logging_level="WARNING",
            )
            linear = _make_deconvolution_grid_solver(setting())
            nonlinear_steps = linear_steps = 3
        else:
            nonlinear = LevenbergMarquardt(
                setting(), regpars=iter([1e-2] * 3),
                cg_pars=dict(_DECONVOLUTION_CG_PARS), logging_level="WARNING",
            )
            linear = Spectral_IteratedTikhonov(setting(), regpar=1e-2, k=3)
            nonlinear_steps, linear_steps = 3, 1

        nonlinear_history = _deconvolution_steps(nonlinear, nonlinear_steps)
        linear_history = _deconvolution_steps(linear, linear_steps)
        if method_family == "levenberg-marquardt":
            nonlinear_history = nonlinear_history[-1:]
        if method_family == "fista":
            assert linear.tau == nonlinear.tau == 0.9
        mode = "codomain-free" if codomain_free else "standard"
        histories[f"nonlinear/{mode}"] = nonlinear_history
        histories[f"linear/{mode}"] = linear_history

    _assert_four_histories_agree(histories, tolerance)
