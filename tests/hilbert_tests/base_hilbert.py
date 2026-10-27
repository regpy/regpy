from copy import deepcopy
import traceback

import numpy as np
import pytest

from regpy.vecsps import NumPyVectorSpace, UniformGridFcts, DirectSum as VectorDirectSum
from regpy.operators import MatrixMultiplication, Identity
from regpy.functionals.base import as_functional
from regpy.hilbert import as_hilbert_space
from regpy.hilbert.base import *
from regpy.hilbert.base import (
    AbstractScaledSpace,
    AbstractSum,
    ScaledHilbertSpace,
    DirectSum as HilbertDirectSum,
)
from regpy.util import set_rng_seed, Errors

set_rng_seed(15873098306879350073259142812684978477)


def call_safe(obj, method_name, *args, **kwargs):
    """
    Calls a method of an object safely.
    
    Returns
    -------
    result
    
    Raises
    ------
    Assertion Error if the method fails
    """
    method = getattr(obj, method_name)
    try:
        result = method(*args, **kwargs)
        return result
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"The method {method_name} of {obj} with arguments {args} and keyword arguments {kwargs} could not construct an object resulting in exception {e}."+"\n"+"Traceback from:\n"+f" {tb}",obj=obj,meth=method_name))

def check_parallelogram_identity(h_space, rel = 1e-6,tol=1e-10, u= None, v = None):
    u = h_space.vecsp.randn() if u is None else u
    v = h_space.vecsp.randn() if v is None else v
    diff = abs(h_space.norm(u+v)**2 + h_space.norm(u-v)**2 - 2*h_space.norm(u)**2 - 2*h_space.norm(v)**2)
    assert diff == pytest.approx(0, rel = rel, abs = tol), Errors.failed_test(f"The parallelogram identity for {h_space} failed with diff = {diff}.",h_space,meth="Parallelogram Identity")


def check_polarization_identity(h_space, rel = 1e-6,tol=1e-10, u= None, v = None):
    u = h_space.vecsp.randn() if u is None else u
    v = h_space.vecsp.randn() if v is None else v
    if np.issubdtype(h_space.vecsp.dtype,np.floating) or h_space.as_real_space:
        diff = abs(h_space.norm(u+v)**2-h_space.norm(u-v)**2-4*(h_space.inner(u,v)))
    else:
        diff = abs(      h_space.norm(u+v)**2 
                   + 1j *h_space.norm(u+1j*v)**2
                   -     h_space.norm(u-v)**2 
                   - 1j *h_space.norm(u-1j*v)**2
                   -4*(h_space.inner(u,v)))
    assert diff == pytest.approx(0, rel = rel, abs = tol), Errors.failed_test(f"The polarization identity for {h_space} failed with diff = {diff}.",h_space,meth="Parallelogram Identity")

def hilbert_basics(sp,test_methods = False,**kwargs):
    if not isinstance(sp,HilbertSpace):
        raise ValueError(f"This test should test HilbertSpace instances not {sp}")
    
    if test_methods:
        x = sp.vecsp.randn()
        y = sp.vecsp.randn()
        _ = call_safe(sp,"inner",x,y)
        _ = call_safe(sp,"norm",x)
        try:
            _ = sp.gram_inv(x)
            _ = call_safe(sp,"norm_functional",x)
            _ = call_safe(sp,"dual_space")
        except NotImplementedError:
            pass

    sp_alt = deepcopy(sp)
    try:
        assert sp == sp_alt, Errors.failed_test(f"Making a deepcopy of {sp} creates a different object {sp_alt}",sp,meth="copy")
        _ = sp + sp_alt
        _ = 6 * sp 
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"The addition and power implementations for {sp} do not properly work. Throwing and exception {e}. Resulting from {tb}",sp,meth="Linear Combinations"))

    tol = 1e-10 if "tol" not in kwargs else kwargs["tol"]
    u = None if "u" not in kwargs else kwargs["u"]
    v = None if "v" not in kwargs else kwargs["v"]
    
    check_parallelogram_identity(sp,tol=tol,u=u,v=v)

@pytest.mark.parametrize("vs",[ 
    NumPyVectorSpace((2,4),dtype=float),
    NumPyVectorSpace((2,4),dtype=complex)
])
def test_L2Generic(vs):
    l2 = L2Generic(vs)
    hilbert_basics(l2,test_methods=True)

def test_GramHilbertSpace():
    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),inverse='cholesky',domain= NumPyVectorSpace(4),codomain=NumPyVectorSpace(4))
    gram_sp = GramHilbertSpace(op_mat)
    hilbert_basics(gram_sp,test_methods=True)

@pytest.mark.parametrize("inverse", [ 
    None,
    "cholesky",
    "conjugate"
])
def test_HilbertPullBack(inverse):
    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),inverse='cholesky',domain= NumPyVectorSpace(4),codomain=NumPyVectorSpace(4))
    l2 = L2Generic(op_mat.domain)
    pullback = HilbertPullBack(l2,op_mat, inverse = inverse)
    hilbert_basics(pullback,test_methods=True)

def test_TensorProd():
    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),inverse='cholesky',domain= NumPyVectorSpace(4),codomain=NumPyVectorSpace(4))
    l2 = L2Generic(op_mat.domain)
    pullback = HilbertPullBack(l2,op_mat)
    prod = TensorProd((3.0,l2),(1,pullback))
    
    hilbert_basics(prod,test_methods=True)

def test_AbstractSpace():
    try:
        ab_sp = AbstractSpace("TestSpace")
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"Trying to create a new Abstract space failed with exception {e} from: {tb}",AbstractSpace,meth="Creation"))
    vs = NumPyVectorSpace((4,2),dtype=complex)
    try:
        ab_sp.register(NumPyVectorSpace,L2Generic)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"Trying to register space failed with exception {e} from: {tb}",AbstractSpace,meth="register"))
    try:
        sp = ab_sp(vs)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"Trying to evaluate on the registered space failed with exception {e} from: {tb}",AbstractSpace,meth="evaluation"))


# Masked Hilbert geometries: deterministic vectors make these tests reproducible.
from regpy.hilbert import L2
from regpy.vecsps import UniformGridFcts
from regpy.vecsps.maskable import MaskableDirectSum, TupleMask
from regpy.operators import Identity, PtwMultiplication
from regpy.operators import DirectSum as OperatorDirectSum


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("kind", ["identity", "scaled_identity", "pointwise"])
def test_masked_hilbert_diagonal_inverse(dtype, kind):
    full = NumPyVectorSpace(4, dtype=dtype)
    diagonal = {
        "identity": np.ones(4),
        "scaled_identity": np.full(4, 3.0),
        "pointwise": np.array([2., 3., 5., 7.]),
    }[kind]
    gram = (Identity(full) if kind == "identity" else
            3 * Identity(full) if kind == "scaled_identity" else
            PtwMultiplication(full, diagonal))
    masked = full.get_masked_space(np.array([True, False, True, False]))
    h = MaskedHilbertSpace(GramHilbertSpace(gram), masked)
    x = np.array([1., -2.], dtype=dtype)
    if dtype is complex:
        x += np.array([2j, 1j])
    expected = diagonal[[0, 2]] * x
    np.testing.assert_allclose(h.gram(x), expected)
    np.testing.assert_allclose(h.gram_inv(expected), x)
    np.testing.assert_allclose(h.gram(h.gram_inv(x)), x)
    assert h.inner(x, x) == pytest.approx(h.space.inner(masked.extend(x), masked.extend(x)))
    assert h.gram_inv is h.gram_inv  # Cached construction.


def test_masked_l2_registration_preserves_grid_weights():
    full = UniformGridFcts(np.linspace(0., 1., 5))
    mask = np.array([True, False, True, True, False])
    masked = full.get_masked_space(mask)
    weights = np.array([1., 2., 3., 4., 5.])
    h = L2(masked, weights=weights)
    assert isinstance(h, MaskedHilbertSpace)
    x = np.array([1., 2., -1.])
    diagonal = full.volume_elem * weights[mask]
    np.testing.assert_allclose(h.gram(x), diagonal * x)
    np.testing.assert_allclose(h.gram_inv(x), x / diagonal)
    assert h.norm(x) == pytest.approx(L2(full, weights=weights).norm(masked.extend(x)))


def test_masked_l2_registration_nested_masks():
    full = NumPyVectorSpace(4)
    first = full.get_masked_space(np.array([True, False, True, True]))
    second = first.get_masked_space(np.array([False, True, True]))
    # A restricted Gram is represented by projections, not by one of the
    # recognized diagonal operator types. The registry deliberately rejects it.
    with pytest.raises(ValueError, match="recognized as coordinate-diagonal"):
        L2(second)
    h = MaskedHilbertSpace(L2(first), second)
    assert isinstance(h.space, MaskedHilbertSpace)
    x = np.array([2., -3.])
    np.testing.assert_allclose(h.gram(x), x)


def test_masked_hilbert_direct_sum_inverse():
    a, b = NumPyVectorSpace(3), NumPyVectorSpace(2)
    full = MaskableDirectSum(a, b)
    gram = OperatorDirectSum(
        2 * Identity(a), PtwMultiplication(b, np.array([3., 5.])),
        domain=full, codomain=full,
    )
    mask = TupleMask((np.array([True, False, True]), np.array([False, True])))
    masked = full.get_masked_space(mask)
    h = MaskedHilbertSpace(GramHilbertSpace(gram), masked)
    x = np.array([1., -2., 4.])
    np.testing.assert_allclose(h.gram(x), np.array([2., 2., 5.]) * x)
    np.testing.assert_allclose(h.gram_inv(h.gram(x)), x)
    default = L2(masked)
    assert isinstance(default, MaskedHilbertSpace)
    np.testing.assert_allclose(default.gram(x), x)
    np.testing.assert_allclose(default.gram_inv(x), x)


def test_masked_hilbert_coupled_gram():
    full = NumPyVectorSpace(3)
    matrix = np.array([[4., 1., 2.], [1., 3., .5], [2., .5, 5.]])
    gram = MatrixMultiplication(matrix, domain=full, codomain=full)
    masked = full.get_masked_space(np.array([True, False, True]))
    h = MaskedHilbertSpace(GramHilbertSpace(gram), masked)
    restricted = matrix[np.ix_([0, 2], [0, 2])]
    x = np.array([1., -2.])
    np.testing.assert_allclose(h.gram(x), restricted @ x)
    with pytest.raises(NotImplementedError):
        _ = h.gram_inv
    inverse = MatrixMultiplication(
        np.linalg.inv(restricted), domain=masked, codomain=masked
    )
    explicit = MaskedHilbertSpace(h.space, masked, inverse=inverse)
    assert explicit.gram_inv is inverse
    np.testing.assert_allclose(explicit.gram_inv(h.gram(x)), x)
    # Restricting the full inverse would incorrectly retain eliminated couplings.
    assert not np.allclose(
        np.linalg.inv(restricted), np.linalg.inv(matrix)[np.ix_([0, 2], [0, 2])]
    )


@pytest.mark.parametrize("matrix", [
    np.diag([2., 3., 4.]),
    np.array([[4., 1., 0.], [1., 3., 0.], [0., 0., 2.]]),
])
def test_masked_l2_rejects_unrecognized_gram_types(monkeypatch, matrix):
    import regpy.hilbert as hilbert

    full = NumPyVectorSpace(3)
    masked = full.get_masked_space(np.array([True, False, True]))
    geometry = GramHilbertSpace(
        MatrixMultiplication(matrix, domain=full, codomain=full)
    )
    # Use the registered dispatcher, but replace only its recursive full-space
    # lookup. Even a diagonal dense matrix is intentionally not recognized.
    registered_l2 = hilbert.L2

    def full_l2(vecsp, **kwargs):
        assert vecsp == full
        return geometry

    monkeypatch.setattr(hilbert, "L2", full_l2)
    with pytest.raises(ValueError, match="recognized as coordinate-diagonal"):
        registered_l2(masked)
    explicit = MaskedHilbertSpace(geometry, masked)
    x = np.array([1., -2.])
    np.testing.assert_allclose(
        explicit.gram(x), matrix[np.ix_([0, 2], [0, 2])] @ x
    )


def small_grid():
    return UniformGridFcts((0., 1., 7), (0., 1., 8))


@pytest.mark.parametrize('factor', [0.001, 0.5, 1., 2., np.float64(3.), np.int64(4)])
def test_abstract_and_concrete_scale_norm_on_same_vector_space(factor):
    grid = small_grid()
    x = grid.ones()
    x[2:5, 1:4] += 0.3
    base = L2(grid)
    deferred = factor * L2
    actual = as_hilbert_space(deferred, grid)
    direct = factor * base

    assert isinstance(deferred, AbstractScaledSpace)
    assert isinstance(actual, ScaledHilbertSpace)
    assert isinstance(direct, ScaledHilbertSpace)
    assert actual.vecsp == grid
    assert direct.vecsp == grid
    assert not isinstance(actual, HilbertDirectSum)
    np.testing.assert_allclose(actual.norm(x), factor * base.norm(x))
    np.testing.assert_allclose(direct.norm(x), factor * base.norm(x))
    np.testing.assert_allclose(actual.gram(x), factor**2 * base.gram(x))
    np.testing.assert_allclose(actual.gram_inv(x), float(factor)**-2 * base.gram_inv(x))
    np.testing.assert_allclose(as_functional(deferred, grid)(x),
                               factor**2 * as_functional(L2, grid)(x))


def test_nested_scaling_is_combined():
    grid = small_grid()
    abstract = 2. * (3. * L2)
    concrete = 2. * (3. * L2(grid))
    assert abstract.factor == 6.
    assert concrete.factor == 6.
    assert abstract.space is L2
    assert not isinstance(concrete.space, ScaledHilbertSpace)
    np.testing.assert_allclose(abstract(grid).gram(grid.ones()),
                               36. * L2(grid).gram(grid.ones()))


def test_abstract_direct_sum_addition_is_not_changed():
    grid1, grid2 = small_grid(), small_grid()
    vecsp = VectorDirectSum(grid1, grid2)
    abstract = L2 + 2. * L2
    concrete = abstract(vecsp)
    assert isinstance(abstract, AbstractSum)
    assert isinstance(concrete, HilbertDirectSum)
    assert concrete.vecsp == vecsp
    x = vecsp.ones()
    expected = L2(grid1).norm(x[0])**2 + 4. * L2(grid2).norm(x[1])**2
    np.testing.assert_allclose(concrete.norm(x)**2, expected)


def test_scalar_multiplier_is_a_norm_factor_not_a_penalty_weight():
    grid = small_grid()
    weight = 0.001
    x = grid.ones()
    # To obtain weight * ||x||_L2**2 one must scale the Hilbert norm by sqrt(weight).
    penalty = as_functional(np.sqrt(weight) * L2, grid)
    reference = weight * as_functional(L2, grid)
    np.testing.assert_allclose(penalty(x), reference(x))
    np.testing.assert_allclose(penalty.conj(grid.ones()),
                               reference.conj(grid.ones()))


@pytest.mark.parametrize('invalid', [0., -2., np.nan, np.inf])
def test_invalid_norm_scales_rejected(invalid):
    grid = small_grid()
    with pytest.raises(ValueError):
        _ = invalid * L2
    with pytest.raises(ValueError):
        _ = invalid * L2(grid)

