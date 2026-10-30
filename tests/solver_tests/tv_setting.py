"""Stable mathematical and public-interface contracts for TV_Setting/TVaug_Setting.

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
from regpy.solvers.special_settings import TV_Setting, TVaug_Setting
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
    # The physical TV_Setting stores the gradient in its TV functional,
    # not on the Setting itself; TVaug_Setting exposes it as .grad.
    grad = setting.tv.grad if isinstance(setting, TV_Setting) else setting.grad
    derivative = np.asarray(grad(image))
    value_axes = tuple(range(2, derivative.ndim))
    magnitudes = np.sqrt(np.sum(np.abs(derivative)**2, axis=value_axes))
    if beta == 0:
        density = magnitudes
    else:
        density = np.where(magnitudes <= beta,
                           magnitudes**2 / (2 * beta),
                           magnitudes - beta / 2)
    weights = grad.codomain.scalar_space().measure
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
    augmented = setting.tvaug_setting
    assert augmented is not None
    assert augmented.forward_op is setting.op
    augmented_result = (augmented.data_fid(augmented.op(image)) / alpha
                        + augmented.penalty(image))
    np.testing.assert_allclose(result, expected, rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(augmented_result, expected, rtol=3e-10, atol=3e-10)


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
    augmented = setting.tvaug_setting
    assert augmented is not None
    augmented_result = (augmented.data_fid(augmented.op(image)) / alpha
                        + augmented.penalty(image))
    np.testing.assert_allclose(result, expected, rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(augmented_result, expected, rtol=3e-9, atol=3e-9)


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
    augmented = setting.tvaug_setting
    for alpha in (0.2, 0.025, 0.75):
        setting.regpar = alpha
        result = setting.data_fid(setting.op(image)) / alpha + setting.penalty(image)
        expected = physical_fidelity(image) / alpha + tv_value
        assert setting.tvaug_setting is augmented
        assert augmented.regpar == pytest.approx(alpha)
        augmented_result = (augmented.data_fid(augmented.op(image)) / alpha
                            + augmented.penalty(image))
        np.testing.assert_allclose(result, expected, rtol=3e-9, atol=3e-9)
        np.testing.assert_allclose(augmented_result, expected,
                                   rtol=3e-9, atol=3e-9)


@pytest.mark.parametrize("fidelity", [L2, KL(offset=0.1)], ids=["L2", "KL"])
def test_observation_updates_accept_physical_and_augmented_data(fidelity):
    """Standard data are physical; TVaug additionally accepts stacked data."""
    grid, image, observed = _image_problem()
    alpha, beta, scale = 0.1, 0.05, 5.
    setting = TV_Setting(
        Identity(grid), data_fid=fidelity, data=observed,
        regpar=alpha, beta=beta, data_operator_scale=scale,
    )
    augmented = setting.tvaug_setting
    fixed_tv = _integrated_tv(setting, image, beta)

    def check(new_observations):
        data_term = as_functional(fidelity, grid).as_data_func(new_observations)
        expected = data_term(image) / alpha + fixed_tv
        actual = setting.data_fid(setting.op(image)) / alpha + setting.penalty(image)
        augmented_actual = (augmented.data_fid(augmented.op(image)) / alpha
                            + augmented.penalty(image))
        np.testing.assert_allclose(actual, expected, rtol=3e-9, atol=3e-9)
        np.testing.assert_allclose(augmented_actual, expected, rtol=3e-9, atol=3e-9)
        np.testing.assert_allclose(setting.data, new_observations)
        np.testing.assert_allclose(augmented.data[0], scale * new_observations)
        assert np.all(np.asarray(augmented.data[1]) == 0)

    first = observed + 0.03
    setting.data = first  # physical coordinates; automatically updates TVaug
    check(first)

    second = observed + 0.07
    stacked = augmented.op.codomain.zeros()  # NOT setting.op.codomain
    stacked[0][...] = scale * second
    augmented.data = stacked  # TVaug accepts (c * data, zero)
    setting.data = second    # standard Setting accepts physical observations
    check(second)
    # Assigning existing physical observations must not double-scale/shift.
    setting.data = setting.data
    check(second)

    invalid = augmented.op.codomain.zeros()
    invalid[0][...] = scale * second
    invalid[1][...] = 1.
    with pytest.raises(ValueError):
        augmented.data = invalid


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
    augmented = setting.tvaug_setting
    augmented_actual = (augmented.data_fid(augmented.op(image)) / alpha
                        + augmented.penalty(image))
    np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(augmented_actual, expected, rtol=3e-10, atol=3e-10)


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
    augmented = setting.tvaug_setting
    augmented_actual = (augmented.data_fid(augmented.op(image)) / alpha
                        + augmented.penalty(image))
    np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(augmented_actual, expected, rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize("color", [False, True], ids=["gray", "RGB"])
def test_pdhgaug_returns_finite_physical_image(color):
    """PDHGaug returns T(x), not the stacked (c*T(x), grad(x)) image."""
    grid, _, observed = _image_problem(color=color)
    op = Identity(grid)
    setting = TV_Setting(
        op, data_fid=L2, data=observed, regpar=0.1,
        beta=0.05, data_operator_scale=3.,
    )
    reconstruction, image = setting.run(
        "PDHGaug", stoprule=CountIterations(max_iterations=4),
        op_norm=1., norm_grad=100.,
    )
    assert reconstruction in grid
    assert np.all(np.isfinite(reconstruction))
    assert image in op.codomain
    np.testing.assert_allclose(image, op(reconstruction), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("method", ["PDHG", "dPDHG"])
@pytest.mark.parametrize("color", [False, True], ids=["gray", "RGB"])
def test_pdhg_returns_finite_original_image_including_dual_method(method, color):
    """PDHG/dPDHG run on TVaug; dPDHG requires a suitable conjugate penalty.

    The null additional penalty in TVaug normally prevents dPDHG from being
    applicable (the conjugate has no subgradient). An L2 penalty permits both.
    """
    grid, _, observed = _image_problem(color=color)
    setting = TVaug_Setting(
        Identity(grid), data_fid=L2, penalty=L2, data=observed,
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
        assert image is not None
        # The original-primal image has two blocks; compare them separately.
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
    # The standard TV_Setting permits regpar=None; scale is only validated
    # when a convex Tikhonov setting actually constructs its TVaug child.
    with pytest.raises(ValueError):
        TV_Setting(Identity(grid), **{'regpar': 0.1,
                                     'explicitDataWarning': False,
                                     **parameters})

# Objective and synchronization tests moved from special_setting_objectives.py
# and merged from the separate test_tv_setting_objectives.py.

@pytest.mark.parametrize('beta', [0., 0.03], ids=['TV', 'Huber-TV'])
@pytest.mark.parametrize('scale', [1., 7., 200.])
def test_tv_setting_scaled_scalar_kl_has_identical_objective(beta, scale):
    """Standard TV and its TVaug child agree for non-translation KL data.

    Only the physical data term sees the observations; scaling T must not
    shift the TV block. Data and alpha updates preserve the same objective.
    """
    grid, image, data = _image_problem()
    op = Identity(grid)
    alpha = 0.012
    setting = TV_Setting(
        op, data_fid=KL(offset=0.1), data=data, regpar=alpha,
        beta=beta, data_operator_scale=scale,
    )
    augmented = setting.tvaug_setting
    assert setting.op is op
    assert isinstance(augmented, TVaug_Setting)
    assert augmented.forward_op is op
    assert augmented.data_operator_scale == pytest.approx(scale)

    def check_objectives(observations, regpar):
        raw = as_functional(KL(offset=0.1), grid).as_data_func(observations)
        expected = raw(op(image)) / regpar + _integrated_tv(setting, image, beta)
        standard = setting.data_fid(setting.op(image)) / regpar + setting.penalty(image)
        augmented_value = (augmented.data_fid(augmented.op(image)) / regpar
                           + augmented.penalty(image))
        np.testing.assert_allclose(standard, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(augmented_value, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(setting.penalty(image), setting.tv(image),
                                   rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            augmented.data_fid(augmented.op(image)) / regpar,
            raw(op(image)) / regpar + augmented.tv_func(augmented.grad(image)),
            rtol=1e-9, atol=1e-9,
        )
        np.testing.assert_allclose(setting.data, observations)
        np.testing.assert_allclose(augmented.data[0], scale * observations)
        assert np.all(np.asarray(augmented.data[1]) == 0)

    check_objectives(data, alpha)

    new_data = data + 0.03
    setting.change_data(new_data)
    assert setting.tvaug_setting is augmented
    check_objectives(new_data, alpha)

    setting.regpar = 2. * alpha
    assert setting.tvaug_setting is augmented
    assert augmented.regpar == pytest.approx(2. * alpha)
    check_objectives(new_data, 2. * alpha)


@pytest.mark.parametrize('beta', [0., 0.03], ids=['TV', 'Huber-TV'])
def test_tvaug_setting_direct_construction_preserves_kl_tv_objective(beta):
    """The stand-alone augmented formulation is named TVaug_Setting."""
    grid, image, data = _image_problem()
    op = Identity(grid)
    alpha = 0.012
    scale = 7.
    augmented = TVaug_Setting(
        op, data_fid=KL(offset=0.1), data=data, regpar=alpha,
        beta=beta, data_operator_scale=scale,
    )
    for observations in (data, data + 0.03):
        if observations is not data:
            augmented.change_data(observations)
        raw = as_functional(KL(offset=0.1), grid).as_data_func(observations)
        expected = raw(op(image)) / alpha + _integrated_tv(augmented, image, beta)
        actual = (augmented.data_fid(augmented.op(image)) / alpha
                  + augmented.penalty(image))
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(augmented.data[0], scale * observations)
        assert np.all(np.asarray(augmented.data[1]) == 0)


def test_tv_setting_additional_quadratic_penalty_not_double_counted():
    """Optional R occurs once in the objective of both TV formulations."""
    grid, image, data = _image_problem()
    op = Identity(grid)
    alpha = 0.1
    setting = TV_Setting(
        op, data_fid=KL(offset=0.1), penalty=L2,
        regpar=alpha, data=data, data_operator_scale=1.,
    )
    augmented = setting.tvaug_setting
    raw = as_functional(KL(offset=0.1), grid).as_data_func(data)
    extra = as_functional(L2, grid)
    expected = raw(op(image)) / alpha + _integrated_tv(setting, image, 0.) + extra(image)
    standard = setting.data_fid(op(image)) / alpha + setting.penalty(image)
    augmented_value = (augmented.data_fid(augmented.op(image)) / alpha
                       + augmented.penalty(image))
    np.testing.assert_allclose(standard, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(augmented_value, expected, rtol=1e-9, atol=1e-9)


def test_unregularized_tv_setting_has_no_augmented_child():
    """No regularization parameter is legal for ordinary TV settings."""
    grid, image, observed = _image_problem()
    op = Identity(grid)
    setting = TV_Setting(op, data_fid=L2, data=observed, regpar=None)
    assert setting.op is op
    assert not setting.is_tikhonov
    assert setting.tvaug_setting is None
    np.testing.assert_allclose(setting.penalty(image),
                               _integrated_tv(setting, image, 0.),
                               rtol=3e-10, atol=3e-10)
