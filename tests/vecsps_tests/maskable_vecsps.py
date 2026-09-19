"""General maskable-space contract tests, including iterator defaults.

Run explicitly with pytest vecsps_tests/maskable_vecsps.py, or import this
module from the project's vector-space test collector.
"""
import numpy as np
import pytest

from regpy.vecsps.base import DirectSum
from regpy.vecsps.maskable import (
    MaskableVectorSpace, MaskableDirectSum, MaskedVectorSpace, TupleMask,
)
from regpy.vecsps.numpy import NumPyVectorSpace


class IteratorSpace(MaskableVectorSpace):
    """Minimal real backend exercising the generic iterator-based methods."""

    def __init__(self, size=6):
        super().__init__(np.ndarray, size, type=np)

    def iter_coefficients(self, x):
        if x not in self:
            raise ValueError("Vector outside space.")
        return iter(x)

    def set_coefficients(self, x, coefficients):
        values = list(coefficients)
        if x not in self or len(values) != self.size:
            raise ValueError("Invalid coefficient count or vector.")
        x[:] = values


@pytest.fixture(params=["iterator", "numpy", "sum", "nested"])
def space(request):
    if request.param == "iterator":
        return IteratorSpace()
    if request.param == "numpy":
        return NumPyVectorSpace((2, 3))
    a, b = NumPyVectorSpace((2, 2)), NumPyVectorSpace(3)
    if request.param == "sum":
        return MaskableDirectSum(a, b)
    return MaskableDirectSum(MaskableDirectSum(a, b), IteratorSpace(2))


def vector(space, values):
    x = space.empty()
    space.set_coefficients(x, values)
    return x


def coefficients(space, x):
    return np.asarray(list(space.iter_coefficients(x)))


def test_comparisons(space):
    values = np.resize(np.array([-2., 0., 3.]), space.size)
    x = vector(space, values)
    assert space.is_ordered
    for mask, expected in [
        (space.IfPos(x), values > 0),
        (space.IfNonneg(x), values >= 0),
        (space.IfZero(x), values == 0),
    ]:
        assert space.is_mask(mask)
        np.testing.assert_array_equal(list(space.iter_mask(mask)), expected)
        assert space.nrTrues(mask) == int(np.count_nonzero(expected))


def test_mask_logic(space):
    a = np.arange(space.size) % 2 == 0
    b = np.arange(space.size) % 3 == 0
    ma, mb = space.mask_from_iterable(a), space.mask_from_iterable(b)
    for result, expected in [
        (space.logical_and(ma, mb), a & b),
        (space.logical_or(ma, mb), a | b),
        (space.logical_xor(ma, mb), a ^ b),
        (space.logical_not(ma), ~a),
    ]:
        assert space.is_mask(result)
        np.testing.assert_array_equal(list(space.iter_mask(result)), expected)


@pytest.mark.parametrize("selection", ["empty", "full", "partial"])
def test_restriction_extension(space, selection):
    keep = np.arange(space.size) % 2 == 0
    if selection != "partial":
        keep[:] = selection == "full"
    mask = space.mask_from_iterable(keep)
    restricted = space.get_masked_space(mask)
    assert isinstance(restricted, MaskedVectorSpace)
    assert restricted.get_full_space() is space
    assert restricted.size == np.count_nonzero(keep)
    x = vector(space, np.arange(space.size) + 1.)
    small = restricted.restrict(x)
    assert small in restricted
    np.testing.assert_array_equal(small, coefficients(space, x)[keep])
    np.testing.assert_array_equal(
        coefficients(space, restricted.extend(small)),
        np.where(keep, coefficients(space, x), 0.),
    )
    np.testing.assert_array_equal(restricted.restrict(restricted.extend(small)), small)
    # Compatibility alias and explicit full-space operations.
    assert space.masked_space(mask) == restricted
    np.testing.assert_array_equal(space.restrict(x, mask), small)
    np.testing.assert_array_equal(
        coefficients(space, space.extend(small, mask)),
        coefficients(space, restricted.extend(small)),
    )


def test_masked_writes_and_source_preservation(space):
    rng = np.random.default_rng(20260919)
    values = rng.normal(size=space.size)
    x = vector(space, values)
    keep = np.arange(space.size) % 2 == 0
    mask = space.mask_from_iterable(keep)
    out = space.zeros()
    space.fill(out, -4.)
    space.copyto(out, x, where=mask)
    np.testing.assert_array_equal(coefficients(space, out), np.where(keep, values, -4.))
    space.fill(out, 3., where=mask)
    np.testing.assert_array_equal(coefficients(space, out), np.where(keep, 3., -4.))
    np.testing.assert_array_equal(coefficients(space, x), values)
    space.copyto(x, x, where=mask)
    np.testing.assert_array_equal(coefficients(space, x), values)


def test_restriction_and_extension_are_adjoint(space):
    rng = np.random.default_rng(83)
    mask = space.mask_from_iterable(np.arange(space.size) % 2 == 0)
    restricted = space.get_masked_space(mask)
    x = vector(space, rng.normal(size=space.size))
    y = rng.normal(size=restricted.size)
    np.testing.assert_allclose(
        restricted.vdot(restricted.restrict(x), y),
        space.vdot(x, restricted.extend(y)), rtol=1e-13, atol=1e-13,
    )


def test_mask_validation(space):
    assert not space.is_mask(None)
    assert not space.is_mask(iter([True] * space.size))
    with pytest.raises(ValueError):
        space.mask_from_iterable([True] * (space.size + 1))
    with pytest.raises(ValueError):
        space.get_masked_space(None)


def test_direct_sum_construction_and_tuple_masks():
    a, b = NumPyVectorSpace(2), NumPyVectorSpace((2, 2))
    for space in (DirectSum(a, b), a + b, a ** 2):
        assert isinstance(space, MaskableDirectSum)
        mask = space.IfZero(space.zeros())
        assert isinstance(mask, TupleMask)
        assert mask.all()
        assert not (~mask).any()
        assert (mask ^ mask).sum() == 0
        assert len(tuple(mask)) == len(space.summands)
    with pytest.raises(ValueError):
        bool(mask)


def test_mask_snapshot():
    space = NumPyVectorSpace(3)
    mask = np.array([True, False, True])
    restricted = space.get_masked_space(mask)
    mask[:] = False
    np.testing.assert_array_equal(restricted.mask, [True, False, True])


@pytest.mark.parametrize("mixed", [False, True])
def test_complex_ordering(mixed):
    complex_space = NumPyVectorSpace(3, dtype=complex)
    space = DirectSum(NumPyVectorSpace(2), complex_space) if mixed else complex_space
    assert not space.is_ordered
    assert space.is_complex == (not mixed)
    for comparison in (space.IfPos, space.IfNonneg):
        with pytest.raises(TypeError):
            comparison(space.zeros())
    assert space.nrTrues(space.IfZero(space.zeros())) == space.size
