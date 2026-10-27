"""Stable mathematical and public-interface contracts for TV_Setting.

These tests intentionally avoid choosing a default solver, inspecting method
registries, asserting stopping-rule thresholds, or depending on the layout of
internal functional wrappers. They should survive changes in solver heuristics.

Run from the RegPy repository root::

    pytest -q tests/solver_tests/test_tv_setting.py
"""

import numpy as np
import pytest

from regpy.functionals import KL
from regpy.functionals.base import as_functional
from regpy.hilbert import L2
from regpy.operators import Identity
from regpy.solvers.special_settings import TV_Setting
from regpy.stoprules import CountIterations
from regpy.vecsps import UniformGridFcts


def _image_problem(*, color=False):
    """Positive rectangular test image; KL is used only on scalar grids."""
    opts = {"shape_codomain": (3,)} if color else {}
    grid = UniformGridFcts((0., 1., 7), (0., 1., 9), **opts)
    x = np.asarray(grid.axes[0])[:, None]
    y = np.asarray(grid.axes[1])[None, :]
    scalar = 0.8 + 0.12 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    if color:
        image = np.stack((scalar, 0.7 + 0.2*x + 0*y,
                          0.9 + 0*x - 0.12*y), axis=-1)
        observed = image + np.array([0.08, -0.04, 0.06])
    else:
        image = scalar
        observed = image + 0.08 + 0.01 * x
    return grid, image, observed


def _integrated_tv(setting, image, beta):
    """Independent isotropic-TV formula, not a call to setting.tv_func.

    The Jacobian has shape (rows, cols, directions) for scalar images or
    (rows, cols, channels, directions) for RGB. The latter MUST be reduced
    entrywise: np.linalg.norm(..., ord=2) would give a spectral matrix norm.
    """
    derivative = np.asarray(setting.grad(image))
    value_axes = tuple(range(2, derivative.ndim))
    magnitudes = np.sqrt(np.sum(np.abs(derivative)**2, axis=value_axes))
    if beta == 0:
        density = magnitudes
    else:
        density = np.where(magnitudes <= beta,
                           magnitudes**2 / (2 * beta),
                           magnitudes - beta / 2)
    weights = setting.grad.codomain.scalar_space().measure
    return float(np.sum(weights * density))


@pytest.mark.parametrize("color", [False, True], ids=["gray", "RGB"])
@pytest.mark.parametrize("beta", [0., 0.04], ids=["TV", "Huber-TV"])
@pytest.mark.parametrize("scale", [1., 8.])
def test_l2_tv_objective_is_independent_of_operator_balance(color, beta, scale):
    grid, image, observed = _image_problem(color=color)
    alpha = 0.2
    setting = TV_Setting(
        Identity(grid), data_fid=L2, data=observed,
        regpar=alpha, beta=beta, data_operator_scale=scale,
    )
    data_term = as_functional(L2, grid).as_data_func(observed)
    expected = data_term(image) / alpha + _integrated_tv(setting, image, beta)
    result = setting.data_fid(setting.op(image)) / alpha + setting.penalty(image)
    np.testing.assert_allclose(result, expected, rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize("beta", [0., 0.05])
@pytest.mark.parametrize("scale", [1., 7., 200.])
def test_scalar_kl_tv_objective_is_independent_of_operator_balance(beta, scale):
    grid, image, observed = _image_problem()
    alpha = 0.012
    setting = TV_Setting(
        Identity(grid), data_fid=KL(offset=0.1), data=observed,
        regpar=alpha, beta=beta, data_operator_scale=scale,
    )
    kl = as_functional(KL(offset=0.1), grid).as_data_func(observed)
    expected = kl(image) / alpha + _integrated_tv(setting, image, beta)
    result = setting.data_fid(setting.op(image)) / alpha + setting.penalty(image)
    np.testing.assert_allclose(result, expected, rtol=3e-9, atol=3e-9)


@pytest.mark.parametrize("fidelity", [L2, KL(offset=0.1)], ids=["L2", "KL"])
@pytest.mark.parametrize("beta", [0., 0.03])
def test_changing_alpha_preserves_fidelity_and_tv_weight(fidelity, beta):
    grid, image, observed = _image_problem()
    setting = TV_Setting(
        Identity(grid), data_fid=fidelity, data=observed,
        regpar=0.1, beta=beta, data_operator_scale=7.,
    )
    physical_fidelity = as_functional(fidelity, grid).as_data_func(observed)
    tv_value = _integrated_tv(setting, image, beta)
    for alpha in (0.2, 0.025, 0.75):
        setting.regpar = alpha
        result = setting.data_fid(setting.op(image)) / alpha + setting.penalty(image)
        np.testing.assert_allclose(
            result, physical_fidelity(image) / alpha + tv_value,
            rtol=3e-9, atol=3e-9,
        )


@pytest.mark.parametrize("fidelity", [L2, KL(offset=0.1)], ids=["L2", "KL"])
def test_observation_updates_accept_physical_and_augmented_data(fidelity):
    grid, image, observed = _image_problem()
    alpha, beta, scale = 0.1, 0.05, 5.
    setting = TV_Setting(
        Identity(grid), data_fid=fidelity, data=observed,
        regpar=alpha, beta=beta, data_operator_scale=scale,
    )
    fixed_tv = _integrated_tv(setting, image, beta)

    def check(new_observations):
        data_term = as_functional(fidelity, grid).as_data_func(new_observations)
        actual = setting.data_fid(setting.op(image)) / alpha + setting.penalty(image)
        np.testing.assert_allclose(
            actual, data_term(image) / alpha + fixed_tv,
            rtol=3e-9, atol=3e-9,
        )

    first = observed + 0.03
    setting.data = first  # original physical data coordinates
    check(first)

    second = observed + 0.07
    augmented = setting.op.codomain.zeros()
    augmented[0][...] = scale * second
    setting.data = augmented  # coordinates of the scaled forward block
    check(second)
    setting.data = setting.data  # no double scaling or data shift
    check(second)

    invalid = setting.op.codomain.zeros()
    invalid[0][...] = scale * second
    invalid[1][...] = 1.
    with pytest.raises(ValueError):
        setting.data = invalid


def test_additional_penalty_is_added_exactly_once():
    grid, image, observed = _image_problem()
    alpha, beta = 0.3, 0.04
    additional = 0.17 * as_functional(L2, grid)
    setting = TV_Setting(
        Identity(grid), data_fid=L2, penalty=additional,
        data=observed, regpar=alpha, beta=beta,
        data_operator_scale=6.,
    )
    data_term = as_functional(L2, grid).as_data_func(observed)
    expected = (data_term(image) / alpha
                + _integrated_tv(setting, image, beta)
                + additional(image))
    actual = setting.data_fid(setting.op(image)) / alpha + setting.penalty(image)
    np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-10)


def test_automatic_balancing_preserves_the_physical_objective():
    grid, image, observed = _image_problem()
    alpha = 0.2
    setting = TV_Setting(
        Identity(grid), data_fid=L2, data=observed,
        beta=0.04, regpar=alpha, data_operator_scale="auto",
    )
    expected = (as_functional(L2, grid).as_data_func(observed)(image) / alpha
                + _integrated_tv(setting, image, 0.04))
    actual = setting.data_fid(setting.op(image)) / alpha + setting.penalty(image)
    np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize("method", ["PDHG", "dPDHG"])
@pytest.mark.parametrize("color", [False, True], ids=["gray", "RGB"])
def test_pdhg_returns_finite_original_image_including_dual_method(method, color):
    """Black-box smoke test: never inspect saddle variables or stop defaults."""
    grid, _, observed = _image_problem(color=color)
    setting = TV_Setting(
        Identity(grid), data_fid=L2, data=observed,
        regpar=0.1, beta=0.05, data_operator_scale=3.,
    )
    reconstruction, image = setting.run(
        method, stoprule=CountIterations(max_iterations=4), op_norm=100.,
    )
    assert reconstruction in grid
    assert np.all(np.isfinite(reconstruction))
    if image is not None:
        assert image in setting.op.codomain
    if method == "dPDHG":
        # The transformed-dual method must return an ORIGINAL-primal image.
        # Compare direct-sum components separately: np.asarray(direct_sum)
        # is ragged and cannot be passed to assert_allclose directly.
        assert image is not None
        expected = setting.op(reconstruction)
        for received_block, expected_block in zip(image, expected):
            np.testing.assert_allclose(
                received_block, expected_block, rtol=1e-9, atol=1e-9,
            )


@pytest.mark.parametrize("parameters", [
    {"beta": -0.2}, {"beta": np.inf},
    {"regpar": 0.}, {"regpar": np.nan},
    {"data_operator_scale": 0.}, {"data_operator_scale": -1.},
    {"data_operator_scale": np.inf},
])
def test_invalid_tv_parameters_are_rejected(parameters):
    grid, _, _ = _image_problem()
    with pytest.raises(ValueError):
        TV_Setting(Identity(grid), **parameters)
