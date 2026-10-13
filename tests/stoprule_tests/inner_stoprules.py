"""Regression tests for codomain-free DualityGapStopping and reference refinement.

Run in the RegPy repository with the three companion source changes installed:
    pytest -v -s test_codomain_free_duality_gap.py
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from regpy.vecsps import UniformGridFcts
from regpy.hilbert import L2
from regpy.operators import Operator
from regpy.solvers import Setting
from regpy.stoprules import DualityGapStopping
from regpy.stoprules.default_factories import default_convex_stoprule
from regpy.stoprules.inner_stoprules import _refine_penalty_minimum


class PeriodicBlur(Operator):
    def __init__(self, domain, sigma=1.2):
        super().__init__(domain=domain, codomain=domain, linear=True)
        impulse = np.zeros(domain.shape)
        impulse[0] = 1.
        self.multiplier = np.fft.rfft(gaussian_filter1d(impulse, sigma, mode='wrap'))

    def _eval(self, x):
        return np.fft.irfft(self.multiplier * np.fft.rfft(x), n=x.size).reshape(x.shape)

    def _adjoint(self, y):
        return np.fft.irfft(np.conj(self.multiplier) * np.fft.rfft(y), n=y.size).reshape(y.shape)


def make_pair(n=32, alpha=0.05):
    domain = UniformGridFcts(np.linspace(-1, 1, n, endpoint=False))
    positions = np.linspace(-1, 1, n, endpoint=False)
    truth = 0.1 + 0.7 * np.exp(-((positions + 0.25) / 0.3)**2)
    # Separate operators: creating a codomain-free setting may set a Gram factor.
    op_explicit = PeriodicBlur(domain)
    op_free = PeriodicBlur(domain)
    data = op_explicit(truth)
    explicit = Setting(op_explicit, L2, L2, regpar=alpha,
                       data=data, codomain_free=False, logging_level='WARNING')
    free = Setting(op_free, L2, L2, regpar=alpha,
                   data=data, codomain_free=True, logging_level='WARNING')
    return explicit, free, positions


@pytest.mark.parametrize('multiplier', [0.0, 0.5, 1.0, 2.0])
def test_codomain_free_gap_equals_explicit_gap(multiplier):
    explicit, free, positions = make_pair()
    f = multiplier * (0.12 + 0.3 * np.sin(3 * positions)**2)
    full_gap = explicit.duality_gap(primal=(f, None))
    free_gap = free.duality_gap(primal=(f, None))
    assert np.isfinite(free_gap)
    assert free_gap >= -1e-10
    np.testing.assert_allclose(free_gap, full_gap, atol=1e-10, rtol=1e-10)


def test_codomain_free_gap_uses_backprojected_data_and_only_normal_action():
    _, free, positions = make_pair()
    f = 0.1 + 0.2 * np.sin(positions)
    Tsp = free.op.adjoint_data(free.data)
    Tsp -= free.op.adjoint_eval(f)
    Tsp /= free.regpar
    expected = free.penalty.bregman(f, None, ystar=Tsp)
    np.testing.assert_allclose(free.duality_gap(primal=(f, None)), expected,
                               atol=1e-13, rtol=1e-13)


def test_codomain_free_gap_requires_primal_and_bregman_formula():
    _, free, positions = make_pair()
    with pytest.raises(ValueError, match='primal'):
        free.duality_gap(dual=(None, free.op.domain.zeros()))
    with pytest.raises(ValueError, match='use_bregman_formula'):
        free.duality_gap(primal=(positions, None), use_bregman_formula=False)


def test_codomain_free_default_selects_duality_gap_when_applicable():
    _, free, _ = make_pair()
    # Use the same registry-driven selection as Setting.run, not a hardcoded factory.
    from regpy.solvers.linear.proximal_gradient import FISTA
    rule = default_convex_stoprule(free, {
        'solver_class': FISTA, 'max_it': 10, 'logging_level': 'WARNING'
    })
    nodes = list(_iter_rules(rule))
    assert any(isinstance(node, DualityGapStopping) for node in nodes)


def _iter_rules(rule):
    yield rule
    for child in getattr(rule, 'rules', ()):
        yield from _iter_rules(child)


class _NumPyDomain:
    def __init__(self, n):
        self.n = n

    def empty(self):
        return np.empty(self.n)

    def zeros(self, read_only=False):
        x = np.zeros(self.n)
        if read_only:
            x.flags.writeable = False
        return x

    def copyto(self, out, value):
        np.copyto(out, value)


class _InvertibleDiagonal:
    def __init__(self, diag):
        self.diag = diag

    def inverse(self, v):
        return v / self.diag


class _QuadraticPenalty:
    """Toy penalty with an intentionally inaccurate feasible reference vector."""
    def __init__(self):
        self.domain = _NumPyDomain(3)
        self.center = np.array([1., -2., 0.5])
        self.diag = np.array([1., 2., 4.])
        self.newton_calls = 0

    def __call__(self, x):
        delta = x - self.center
        return float(0.5 * np.dot(delta, self.diag * delta) + 0.7)

    def subgradient(self, x):
        return self.diag * (x - self.center)

    def dist_subdiff(self, vstar, x):
        return float(np.linalg.norm(vstar - self.subgradient(x)))

    def hessian(self, x):
        self.newton_calls += 1
        return _InvertibleDiagonal(self.diag)


def test_reference_newton_refines_without_modifying_input():
    penalty = _QuadraticPenalty()
    initial = penalty.center + np.array([1e-3, 2e-3, -4e-3])
    original = initial.copy()
    refined = _refine_penalty_minimum(penalty, initial, max_steps=2)
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_allclose(refined, penalty.center, atol=1e-13, rtol=0)
    assert 1 <= penalty.newton_calls <= 2
