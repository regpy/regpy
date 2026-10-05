"""Tests for linear solvers on real and complex Hilbert spaces.

The tests are converted from ``test_linear_Hilbert_solvers.ipynb``.
"""

import logging

import numpy as np
import pytest

from regpy.hilbert import L2
from regpy.operators import (
    ExponentialConvolution,
    PeriodicShift,
    PtwMultiplication,
)
from regpy.solvers.linear import (
    FISTA,
    Landweber,
    NuMethod,
    NuMethodSelfAdjoint,
    Richardson,
)
from regpy.solvers.linear.cgne import CGNE
from regpy.solvers.linear.spectral import (
    Spectral_Landweber,
    Spectral_Tikhonov,
    SpectralSelfAdjoint_Richardson,
)
from regpy.solvers.linear.tikhonov import LavrentievCG, TikhonovCG
from regpy.solvers.setting import Setting
from regpy.stoprules import DualityGapStopping
from regpy.util import set_rng_seed
from regpy.vecsps import UniformGridFcts


LANDWEBER_STEPS = 300
OTHER_ITERATION_STEPS = 200
EQUIVALENCE_STEPS = 3
TIKHONOV_MAX_IT = 1000
STRICT_TOL = 1e-12
DOMAIN_ERROR_LIMIT = 1e-2
CODOMAIN_ERROR_LIMIT = 1e-3
REGPAR = 1e-4
NU = 1.0
STEPSIZE_SAFETY = 1.0
TIKHONOV_EQ_RELTOL = 1e-12
TIKHONOV_ABS_TOL = 1e-4
TIKHONOV_REL_TOL = 1e-4
ALTERNATIVE_REGPAR = 1e-2
ALTERNATIVE_ERROR_LIMIT = 1e-1

CASE_NAMES = (
    "real, self-adjoint",
    "real, non-self-adjoint",
    "complex, self-adjoint",
    "complex, non-self-adjoint",
)

set_rng_seed(15873098306879350073259142812684978477)


def _make_case(name):
    grid = UniformGridFcts((-1, 1, 100), periodic=True)
    points = grid.coords[0]

    if name == "real, self-adjoint":
        operator = 50.0 * ExponentialConvolution(grid, a=0.05)
        exact_solution = (
            1.0
            / (
                1
                - 0.4 * np.cos(5 * np.pi * points)
                + 0.3 * np.cos(np.pi * points)
            )
            + 1.0
        )
        self_adjoint = True
    elif name == "real, non-self-adjoint":
        operator = (
            50.0 * ExponentialConvolution(grid, a=0.05)
            * PeriodicShift(grid, shift=(0.3,), self_adjoint=False)
        )
        exact_solution = (
            1.0
            / (
                1
                - 0.4 * np.cos(5 * np.pi * points)
                + 0.3 * np.cos(np.pi * points)
            )
            + 1.0
        )
        self_adjoint = False
    else:
        complex_grid = grid.complex_space()
        exact_solution = (
            1.0
            / (
                1
                - 0.4 * np.exp(5 * np.pi * 1j * points)
                + 0.3 * np.cos(np.pi * points)
            )
            + 1.0
        )
        if name == "complex, self-adjoint":
            operator = 0.02 * ExponentialConvolution(
                complex_grid, a=0.05
            )
            self_adjoint = True
        elif name == "complex, non-self-adjoint":
            operator = (
                50.0 * ExponentialConvolution(complex_grid, a=0.02)
                * PtwMultiplication(
                    complex_grid,
                    1.0 / (1.0 - 0.2 * np.exp(1j * np.pi * points)),
                )
            )
            self_adjoint = False
        else:
            raise ValueError(f"Unknown test case {name!r}.")

    case = {
        "name": name,
        "op": operator,
        "x_exact": exact_solution,
        "self_adjoint": self_adjoint,
    }
    case["data"] = operator(exact_solution)
    case["init"] = operator.domain.zeros()

    standard_setting = _make_setting(case)
    case["h_domain"] = standard_setting.h_domain
    case["h_codomain"] = standard_setting.h_codomain

    case["op_norm"] = operator.norm(
        standard_setting.h_domain,
        standard_setting.h_codomain,
    )
    if self_adjoint:
        case["self_adjoint_op_norm"] = operator.norm_sa(
            standard_setting.h_domain
        )
    case["landweber_stepsize"] = (
        STEPSIZE_SAFETY / case["op_norm"] ** 2
    )
    return case


@pytest.fixture(scope="module", params=CASE_NAMES, ids=CASE_NAMES)
def linear_case(request):
    """Provide the four operator/Hilbert-space combinations."""
    return _make_case(request.param)


def _make_setting(case, *, codomain_free=False, regpar=None):
    return Setting(
        op=case["op"],
        penalty=L2,
        data_fid=L2,
        init=case["init"],
        incorporate_init_in_penalty=False,
        data=case["data"],
        regpar=regpar,
        codomain_free=codomain_free,
        explicitDataWarning=False,
    )


def _difference_norm(hspace, left, right):
    difference = hspace.vecsp.empty()
    hspace.vecsp.copyto(difference, left)
    difference -= right
    return hspace.norm(difference)


def _quadratic_difference_norm(functional, vecsp, left, right):
    """Return the norm of ``left - right`` induced by a Hessian."""
    difference = vecsp.empty()
    vecsp.copyto(difference, left)
    difference -= right
    hessian_difference = vecsp.empty()
    functional.hessian(right)(difference, out=hessian_difference)
    squared_norm = vecsp.vdot(difference, hessian_difference).real
    return np.sqrt(max(squared_norm, 0.0))


def _relative_difference(hspace, left, right):
    denominator = max(
        hspace.norm(left),
        hspace.norm(right),
        np.finfo(float).tiny,
    )
    return _difference_norm(hspace, left, right) / denominator


def _run_steps(solver, max_steps):
    approximation, image = solver.run(max_it=max_steps)
    return approximation, image, solver.iteration_step_nr


def _require_applicable(solver_class, setting):
    result, _ = solver_class.check_applicability(setting)
    assert result["applicable"], (
        f"{solver_class.__name__} unexpectedly inapplicable: "
        f"{result['info']}"
    )


def _assert_strictly_close(hspace, left, right, label):
    error = _relative_difference(hspace, left, right)
    assert error < STRICT_TOL, (
        f"{label}: relative difference {error:.3e} is not below "
        f"{STRICT_TOL:.1e}."
    )


def _assert_internal_bound(
    solver, history_key, measured_error, requested_tolerance
):
    histories = solver.get_convergence_histories()
    estimates = histories.get(history_key, ())
    assert len(estimates), f"No convergence history found for {history_key!r}."
    estimated_error = estimates[-1]
    permitted = requested_tolerance * (1 + 1e-12)
    assert measured_error <= permitted, (
        f"Measured error {measured_error:.3e} exceeds the promised "
        f"bound {requested_tolerance:.1e}."
    )
    assert estimated_error <= permitted, (
        f"Internal estimate {estimated_error:.3e} exceeds the promised "
        f"bound {requested_tolerance:.1e}."
    )


def test_operator_self_adjoint_flag(linear_case):
    assert (
        linear_case["op"].self_adjoint
        is linear_case["self_adjoint"]
    )


@pytest.mark.parametrize(
    "method_name",
    ("Landweber", "NuMethod", "TikhonovCG"),
)
def test_standard_and_codomain_free_agree(linear_case, method_name):
    case = linear_case
    regpar = REGPAR if method_name == "TikhonovCG" else None
    standard_setting = _make_setting(case, regpar=regpar)
    free_setting = _make_setting(
        case, codomain_free=True, regpar=regpar
    )

    if method_name == "Landweber":
        solver_class = Landweber
        parameters = {
            "stepsize": case["landweber_stepsize"],
            "op_norm": case["op_norm"],
            "change_init": False,
            "logging_level": logging.WARNING,
        }
    elif method_name == "NuMethod":
        solver_class = NuMethod
        parameters = {
            "nu": NU,
            "stepsize": case["landweber_stepsize"],
            "op_norm": case["op_norm"],
            "change_init": False,
            "logging_level": logging.WARNING,
        }
    else:
        solver_class = TikhonovCG
        parameters = {
            "tol": None,
            "reltolx": TIKHONOV_EQ_RELTOL,
            "reltoly": None,
            "logging_level": logging.WARNING,
        }

    _require_applicable(solver_class, standard_setting)
    _require_applicable(solver_class, free_setting)
    standard_solver = solver_class(standard_setting, **parameters)
    free_solver = solver_class(free_setting, **parameters)
    standard_solution, _, _ = _run_steps(
        standard_solver, EQUIVALENCE_STEPS
    )
    free_solution, _, _ = _run_steps(free_solver, EQUIVALENCE_STEPS)

    _assert_strictly_close(
        case["h_domain"],
        standard_solution,
        free_solution,
        f"{method_name} domain solutions",
    )
    _assert_strictly_close(
        case["h_codomain"],
        case["op"](standard_solution),
        case["op"](free_solution),
        f"{method_name} images",
    )


@pytest.mark.parametrize("codomain_free", (False, True), ids=("standard", "codomain-free"))
def test_iterative_and_spectral_landweber_agree(
    linear_case, codomain_free
):
    case = linear_case
    setting = _make_setting(case, codomain_free=codomain_free)
    _require_applicable(Landweber, setting)
    _require_applicable(Spectral_Landweber, setting)

    iterative = Landweber(
        setting,
        stepsize=case["landweber_stepsize"],
        op_norm=case["op_norm"],
        change_init=False,
        logging_level=logging.WARNING,
    )
    spectral = Spectral_Landweber(
        _make_setting(case, codomain_free=codomain_free),
        k=EQUIVALENCE_STEPS,
        stepsize=case["landweber_stepsize"],
    )
    iterative_solution, _, _ = _run_steps(
        iterative, EQUIVALENCE_STEPS
    )
    spectral_solution, _, _ = _run_steps(spectral, 1)

    _assert_strictly_close(
        case["h_domain"],
        iterative_solution,
        spectral_solution,
        "iterative and spectral Landweber solutions",
    )


@pytest.mark.parametrize("codomain_free", (False, True), ids=("standard", "codomain-free"))
def test_iterative_and_spectral_richardson_agree(
    linear_case, codomain_free
):
    case = linear_case
    if not case["self_adjoint"]:
        pytest.skip("Richardson requires a self-adjoint operator.")

    setting = _make_setting(case, codomain_free=codomain_free)
    stepsize = STEPSIZE_SAFETY / case["self_adjoint_op_norm"]
    _require_applicable(Richardson, setting)
    _require_applicable(SpectralSelfAdjoint_Richardson, setting)

    iterative = Richardson(
        setting,
        stepsize=stepsize,
        op_norm=case["self_adjoint_op_norm"],
        change_init=False,
        logging_level=logging.WARNING,
    )
    spectral = SpectralSelfAdjoint_Richardson(
        _make_setting(case, codomain_free=codomain_free),
        k=EQUIVALENCE_STEPS,
        stepsize=stepsize,
    )
    iterative_solution, _, _ = _run_steps(
        iterative, EQUIVALENCE_STEPS
    )
    spectral_solution, _, _ = _run_steps(spectral, 1)

    _assert_strictly_close(
        case["h_domain"],
        iterative_solution,
        spectral_solution,
        "iterative and spectral Richardson solutions",
    )


TIKHONOV_STOPPING_CASES = (
    pytest.param(False, "absolute domain", id="standard-absolute-domain"),
    pytest.param(False, "relative domain", id="standard-relative-domain"),
    pytest.param(
        False,
        "relative codomain",
        id="standard-relative-codomain",
    ),
    pytest.param(True, "absolute domain", id="codomain-free-absolute-domain"),
    pytest.param(True, "relative domain", id="codomain-free-relative-domain"),
)


@pytest.mark.parametrize(
    "codomain_free, rule_name", TIKHONOV_STOPPING_CASES
)
def test_tikhonov_cg_stopping_rule_guarantee(
    linear_case, codomain_free, rule_name
):
    case = linear_case
    reference_solver = Spectral_Tikhonov(
        _make_setting(case, codomain_free=codomain_free),
        regpar=REGPAR,
    )
    reference_solution, _, _ = _run_steps(reference_solver, 1)
    reference_image = (
        None if codomain_free else case["op"](reference_solution)
    )

    if rule_name == "absolute domain":
        requested_tolerance = TIKHONOV_ABS_TOL
        history_key = "abs_err"
        solver_parameters = {
            "tol": requested_tolerance,
            "reltolx": None,
            "reltoly": None,
        }
    elif rule_name == "relative domain":
        requested_tolerance = TIKHONOV_REL_TOL
        history_key = "rel_err_x"
        solver_parameters = {
            "tol": None,
            "reltolx": requested_tolerance,
            "reltoly": None,
        }
    else:
        requested_tolerance = TIKHONOV_REL_TOL
        history_key = "rel_err_y"
        solver_parameters = {
            "tol": None,
            "reltolx": None,
            "reltoly": requested_tolerance,
        }

    setting = _make_setting(
        case, codomain_free=codomain_free, regpar=REGPAR
    )
    solver = TikhonovCG(
        setting,
        logging_level=logging.WARNING,
        x_exact=reference_solution,
        y_exact=reference_image,
        **solver_parameters,
    )
    approximation, image, _ = _run_steps(solver, TIKHONOV_MAX_IT)

    if rule_name == "absolute domain":
        measured_error = _quadratic_difference_norm(
            setting.penalty,
            setting.op.domain,
            approximation,
            reference_solution,
        )
    elif rule_name == "relative domain":
        reference_increment = _quadratic_difference_norm(
            setting.penalty,
            setting.op.domain,
            reference_solution,
            setting.init,
        )
        measured_error = (
            _quadratic_difference_norm(
                setting.penalty,
                setting.op.domain,
                approximation,
                reference_solution,
            )
            / max(reference_increment, np.finfo(float).tiny)
        )
    else:
        initial_image = setting.op(setting.init)
        reference_increment = _quadratic_difference_norm(
            setting.data_fid,
            setting.op.codomain,
            reference_image,
            initial_image,
        )
        measured_error = (
            _quadratic_difference_norm(
                setting.data_fid,
                setting.op.codomain,
                image,
                reference_image,
            )
            / max(reference_increment, np.finfo(float).tiny)
        )

    _assert_internal_bound(
        solver,
        history_key,
        measured_error,
        requested_tolerance,
    )


DUAL_TIKHONOV_STOPPING_CASES = (
    pytest.param(
        "absolute domain",
        TIKHONOV_ABS_TOL,
        "abs_err",
        {"tol": TIKHONOV_ABS_TOL, "reltolx": None, "reltoly": None},
        id="absolute-domain",
    ),
    pytest.param(
        "relative domain",
        TIKHONOV_REL_TOL,
        "rel_err_x",
        {"tol": None, "reltolx": TIKHONOV_REL_TOL, "reltoly": None},
        id="relative-domain",
    ),
    pytest.param(
        "relative codomain",
        TIKHONOV_REL_TOL,
        "rel_err_y",
        {"tol": None, "reltolx": None, "reltoly": TIKHONOV_REL_TOL},
        id="relative-codomain",
    ),
)


@pytest.mark.parametrize(
    "rule_name, tolerance, history_key, solver_parameters",
    DUAL_TIKHONOV_STOPPING_CASES,
)
def test_dual_tikhonov_cg_internal_stopping_rule(
    linear_case,
    rule_name,
    tolerance,
    history_key,
    solver_parameters,
):
    """Check direct CG bounds in the transformed dual Hessian norms."""
    case = linear_case
    primal_setting = _make_setting(case, regpar=REGPAR)
    reference_solver = Spectral_Tikhonov(
        _make_setting(case), regpar=REGPAR
    )
    reference_solution, _, _ = _run_steps(reference_solver, 1)
    reference_image = case["op"](reference_solution)

    dual_setting = primal_setting.get_dual_setting()
    dual_reference = primal_setting.primal_to_dual(
        (reference_solution, reference_image)
    )
    dual_reference_image = dual_setting.op(dual_reference)
    solver = TikhonovCG(
        dual_setting,
        logging_level=logging.WARNING,
        x_exact=dual_reference,
        y_exact=dual_reference_image,
        **solver_parameters,
    )
    approximation, image, _ = _run_steps(solver, TIKHONOV_MAX_IT)

    if rule_name == "absolute domain":
        measured_error = _quadratic_difference_norm(
            dual_setting.penalty,
            dual_setting.op.domain,
            approximation,
            dual_reference,
        )
    elif rule_name == "relative domain":
        numerator = _quadratic_difference_norm(
            dual_setting.penalty,
            dual_setting.op.domain,
            approximation,
            dual_reference,
        )
        denominator = _quadratic_difference_norm(
            dual_setting.penalty,
            dual_setting.op.domain,
            dual_reference,
            dual_setting.init,
        )
        measured_error = numerator / max(
            denominator, np.finfo(float).tiny
        )
    else:
        initial_image = dual_setting.op(dual_setting.init)
        numerator = _quadratic_difference_norm(
            dual_setting.data_fid,
            dual_setting.op.codomain,
            image,
            dual_reference_image,
        )
        denominator = _quadratic_difference_norm(
            dual_setting.data_fid,
            dual_setting.op.codomain,
            dual_reference_image,
            initial_image,
        )
        measured_error = numerator / max(
            denominator, np.finfo(float).tiny
        )

    _assert_internal_bound(
        solver, history_key, measured_error, tolerance
    )


LAVRENTIEV_STOPPING_CASES = (
    pytest.param(
        "absolute domain",
        TIKHONOV_ABS_TOL,
        "abs_err",
        {"tol": TIKHONOV_ABS_TOL, "reltolx": None},
        id="absolute-domain",
    ),
    pytest.param(
        "relative domain",
        TIKHONOV_REL_TOL,
        "rel_err_x",
        {"tol": None, "reltolx": TIKHONOV_REL_TOL},
        id="relative-domain",
    ),
)


@pytest.mark.parametrize(
    "rule_name, tolerance, history_key, solver_parameters",
    LAVRENTIEV_STOPPING_CASES,
)
def test_lavrentiev_cg_internal_stopping_rule(
    linear_case,
    rule_name,
    tolerance,
    history_key,
    solver_parameters,
):
    """Exercise the inner solver used by codomain-free TikhonovCG."""
    case = linear_case
    reference_solver = Spectral_Tikhonov(
        _make_setting(case, codomain_free=True), regpar=REGPAR
    )
    reference_solution, _, _ = _run_steps(reference_solver, 1)
    free_setting = _make_setting(
        case, codomain_free=True, regpar=REGPAR
    )
    normal_template = TikhonovCG(
        free_setting,
        tol=None,
        reltolx=TIKHONOV_REL_TOL,
        reltoly=None,
        logging_level=logging.WARNING,
    )
    normal_setting = normal_template._normal_setting
    solver = LavrentievCG(
        normal_setting,
        data=normal_template.backprop_data,
        init=normal_template.init,
        x0=free_setting.init,
        x_exact=reference_solution,
        logging_level=logging.WARNING,
        **solver_parameters,
    )
    approximation, _, _ = _run_steps(solver, TIKHONOV_MAX_IT)

    numerator = _quadratic_difference_norm(
        normal_setting.penalty,
        normal_setting.op.domain,
        approximation,
        reference_solution,
    )
    if rule_name == "absolute domain":
        measured_error = numerator
    else:
        denominator = _quadratic_difference_norm(
            normal_setting.penalty,
            normal_setting.op.domain,
            reference_solution,
            free_setting.init,
        )
        measured_error = numerator / max(
            denominator, np.finfo(float).tiny
        )

    _assert_internal_bound(
        solver, history_key, measured_error, tolerance
    )


ALTERNATIVE_TIKHONOV_METHODS = (
    "FB",
    "dFB",
    "FISTA",
    "dFISTA",
    "PDHG",
    "dPDHG",
)


@pytest.fixture(scope="module")
def alternative_tikhonov_problem():
    """Return a spectral reference for a representative real problem."""
    case = _make_case("real, non-self-adjoint")
    reference_solver = Spectral_Tikhonov(
        _make_setting(case), regpar=ALTERNATIVE_REGPAR
    )
    reference_solution, _, _ = _run_steps(reference_solver, 1)
    return case, reference_solution


@pytest.mark.parametrize("method_name", ALTERNATIVE_TIKHONOV_METHODS)
def test_alternative_tikhonov_minimizer(
    alternative_tikhonov_problem, method_name
):
    case, reference_solution = alternative_tikhonov_problem
    setting = _make_setting(case, regpar=ALTERNATIVE_REGPAR)
    setting.set_stoprule(
        lambda setting, params: DualityGapStopping(
            tol=ALTERNATIVE_ERROR_LIMIT,
            logging_level=logging.DEBUG,
        )
    )

    approximation, _ = setting.run(
        method_name,
        ref_sol=reference_solution,
        max_it=TIKHONOV_MAX_IT,
    )
    relative_error = (
        np.linalg.norm(approximation - reference_solution)
        / np.linalg.norm(reference_solution)
    )
    assert relative_error <= ALTERNATIVE_ERROR_LIMIT, (
        f"{method_name} relative error {relative_error:.3e} exceeds "
        f"{ALTERNATIVE_ERROR_LIMIT:.1e}."
    )


APPROXIMATION_METHODS = (
    pytest.param("Landweber", LANDWEBER_STEPS, id="Landweber"),
    pytest.param("NuMethod", OTHER_ITERATION_STEPS, id="NuMethod"),
    pytest.param("FISTA", OTHER_ITERATION_STEPS, id="FISTA"),
    pytest.param("CGNE", OTHER_ITERATION_STEPS, id="CGNE"),
    pytest.param("Richardson", OTHER_ITERATION_STEPS, id="Richardson"),
    pytest.param(
        "NuMethodSelfAdjoint",
        OTHER_ITERATION_STEPS,
        id="NuMethodSelfAdjoint",
    ),
)


@pytest.mark.parametrize("method_name, max_steps", APPROXIMATION_METHODS)
def test_approximation_quality(linear_case, method_name, max_steps):
    case = linear_case
    setting = _make_setting(case)

    if method_name in {"Richardson", "NuMethodSelfAdjoint"}:
        if not case["self_adjoint"]:
            pytest.skip(f"{method_name} requires a self-adjoint operator.")
        self_adjoint_stepsize = (
            STEPSIZE_SAFETY / case["self_adjoint_op_norm"]
        )

    if method_name == "Landweber":
        solver_class = Landweber
        solver = solver_class(
            setting,
            stepsize=case["landweber_stepsize"],
            op_norm=case["op_norm"],
            change_init=False,
            logging_level=logging.WARNING,
        )
    elif method_name == "NuMethod":
        solver_class = NuMethod
        solver = solver_class(
            setting,
            nu=NU,
            stepsize=case["landweber_stepsize"],
            op_norm=case["op_norm"],
            change_init=False,
            logging_level=logging.WARNING,
        )
    elif method_name == "FISTA":
        solver_class = FISTA
        solver = solver_class(
            setting,
            op_norm=case["op_norm"],
            logging_level=logging.WARNING,
        )
    elif method_name == "CGNE":
        solver_class = CGNE
        solver = solver_class(
            setting,
            change_init=False,
            logging_level=logging.WARNING,
        )
    elif method_name == "Richardson":
        solver_class = Richardson
        solver = solver_class(
            setting,
            stepsize=self_adjoint_stepsize,
            op_norm=case["self_adjoint_op_norm"],
            change_init=False,
            logging_level=logging.WARNING,
        )
    else:
        solver_class = NuMethodSelfAdjoint
        solver = solver_class(
            setting,
            nu=NU,
            stepsize=self_adjoint_stepsize,
            op_norm=case["self_adjoint_op_norm"],
            change_init=False,
            logging_level=logging.WARNING,
        )

    _require_applicable(solver_class, setting)
    approximation, _, performed_steps = _run_steps(solver, max_steps)
    image = case["op"](approximation)
    domain_error = _relative_difference(
        case["h_domain"], approximation, case["x_exact"]
    )
    codomain_error = _relative_difference(
        case["h_codomain"], image, case["data"]
    )

    assert performed_steps <= max_steps
    assert domain_error <= DOMAIN_ERROR_LIMIT, (
        f"{method_name} domain error {domain_error:.3e} exceeds "
        f"{DOMAIN_ERROR_LIMIT:.1e} after {performed_steps} iterations."
    )
    assert codomain_error <= CODOMAIN_ERROR_LIMIT, (
        f"{method_name} codomain error {codomain_error:.3e} exceeds "
        f"{CODOMAIN_ERROR_LIMIT:.1e} after {performed_steps} iterations."
    )
