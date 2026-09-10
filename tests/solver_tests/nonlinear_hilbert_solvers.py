"""Tests for nonlinear solvers on a periodic Hilbert-space problem.

Keep this file in the same directory as ``periodic_c_problem.py``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from .periodic_c_problem import PeriodicCProblem
from regpy.hilbert import Hm, L2
from regpy.solvers import Setting
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
) -> None:
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
        logging_level="DEBUG",
    )
    codomain_free_reconstruction, _ = codomain_free.run(
        "genFISTA",
        max_it=2,
        logging_level="DEBUG",
    )

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
