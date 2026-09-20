# Poisson tests revision: 20260920-real-only
from random import random
import traceback

import pytest
import numpy as np

import regpy.vecsps.base as vs_base
from regpy.util import set_rng_seed,Errors

set_rng_seed(15873098306879350073259142812684978477)
    
def vecsps_basics(vs,*args,test_methods = False,**kwargs):
    """Initializes an object of `vs` with `kwargs` and tests it basic functionality. If `test_methods` is true it test the standard methods that should be available. 

    Parameters
    ----------
    vs : object
        The object to initialize
    test_methods : bool, optional
        Flag if to test methods, by default False
    kwargs : dict
        The keywords to pass to the initialization of the `vs` instance.
        
    Raises
    ------
    AssertionError
        Should any of the tests fail.
    """
    errors = []
    print(kwargs)
    VS = vs(*args,**kwargs)
    if test_methods:
        _ = VS.zeros()
        _ = VS.ones()
        _ = VS.empty()
        _ = VS.rand()
        if not VS.is_complex:
            _ = VS.poisson(VS.ones())
        res = VS.vdot(VS.zeros(),VS.rand())
        assert res == pytest.approx(0), f"The vdot method tested with a zero and random vector resulted in an non-zero answer of {res}"
    VS_alt = vs(*args,**kwargs)
    assert VS == VS_alt, f"The equivalence method __eq__ for {vs} is not properly working."

    try:
        for x_i in VS:
            _ = x_i
    except NotImplementedError:
        pass
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"The iteration over {VS} does not properly work. Throwing and exception {e}. Resulting from {tb}",VS,meth="Iteration"))

    try:
        _ = VS + VS_alt
        VS += VS_alt
        _ = VS**4
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise AssertionError(Errors.failed_test(f"The addition and power implementations for {VS} do not properly work. Throwing and exception {e}. Resulting from {tb}",VS,meth="Linear Combinations"))
    
def vector_basics(vs,*args, test_comparison = True, ones_is_onevec= True, N = 5,**kwargs):
    tol = kwargs["tol"] if "tol" in kwargs else 1e-10
    VS = vs(*args,**kwargs)
    if not VS.as_real_space:
        for _ in range(N):
            v_1 = VS.rand()
            v_2 = VS.randn()
            v_3 = VS.randn()
            scalar = random() + 1j*random()
            comb = v_1 + scalar * v_2
            vdot_1 = VS.vdot(comb,v_3)
            vdot_2 = VS.vdot(v_1,v_3)+scalar.conjugate()*VS.vdot(v_2,v_3)
            assert vdot_1 == pytest.approx(vdot_2,rel=tol), f"Trying to compute the vector dot product of a linear combination of random vectors of {vs} and another random vector failed with `rel_tol` = {tol}. got for the combination {vdot_1} and expressed as sum of vdots {vdot_2} "
    else:
        for _ in range(N):
            v_1 = VS.rand()
            v_2 = VS.randn()
            v_3 = VS.randn()
            scalar = random()
            v_1 *= scalar
            v_1 /= scalar
            comb = v_1 + scalar * v_2
            vdot_1 = VS.vdot(comb,v_3)
            vdot_2 = VS.vdot(v_1,v_3)+scalar*VS.vdot(v_2,v_3)
            assert vdot_1 == pytest.approx(vdot_2,rel=tol), f"Trying to compute the vector dot product of a linear combination of random vectors of {vs} and another random vector failed with `rel_tol` = {tol}. got for the combination {vdot_1} and expressed as sum of vdots {vdot_2} "
            v_1 += v_2
            v_1 -= v_2
            comb = v_1/scalar - v_2
            if test_comparison:
                _ = v_1 < v_2
                _ = v_1 <= v_2
                _ = v_1 >= v_2
                reg = v_1 > v_2
                _ = reg.all()
                _ = reg.any()
    if ones_is_onevec:
        assert VS.ones() in VS
    

def test_VecSpaceBase():
    space = vs_base.VectorSpaceBase(None, 0)
    assert space.dimension == 0
    with pytest.raises(NotImplementedError):
        space.zeros()
    
def test_DirectSum():
    from regpy.vecsps import NumPyVectorSpace
    vecsps_basics(vs_base.DirectSum, NumPyVectorSpace(2), NumPyVectorSpace(3))


class Hermitian3Space(vs_base.VectorSpaceBase):
    """Shape-free real space of Hermitian 3x3 matrices, of dimension nine."""
    def __init__(self):
        super().__init__(np.ndarray, dimension=9)

    def __contains__(self, x):
        return (isinstance(x, np.ndarray) and x.shape == (3, 3)
                and np.allclose(x, x.conj().T))

    def empty(self):
        return np.empty((3, 3), dtype=complex)

    def zeros(self, read_only=False):
        x = np.zeros((3, 3), dtype=complex)
        return self.read_only(x) if read_only else x

    def ones(self, read_only=False):
        x = self.fromflat(np.ones(9))
        return self.read_only(x) if read_only else x

    def read_only(self, x):
        out = x.view()
        out.flags.writeable = False
        return out

    def __eq__(self, other):
        return isinstance(other, Hermitian3Space)

    def compatible(self, other):
        return self == other

    def copyto(self, x, y):
        np.copyto(x, y)

    def fill(self, x, value):
        self.set_coefficients(x, [value] * 9)

    def iter_coefficients(self, x):
        yield from x.diagonal().real
        for i, j in ((0, 1), (0, 2), (1, 2)):
            yield x[i, j].real
            yield x[i, j].imag

    def set_coefficients(self, x, coefficients):
        c = np.asarray(list(coefficients))
        if c.shape != (9,) or np.any(c.imag != 0):
            raise ValueError("Expected nine real coordinates.")
        x[:] = 0
        np.fill_diagonal(x, c[:3])
        for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            x[i, j] = c[3+2*k] + 1j*c[4+2*k]
            x[j, i] = x[i, j].conjugate()

    def vdot(self, x, y):
        return np.vdot(self.flatten(x), self.flatten(y)).real


@pytest.mark.parametrize("dtype,as_real,dim", [
    (float, True, 3), (complex, True, 6), (complex, False, 3),
])
def test_scalar_field_coordinates(dtype, as_real, dim):
    from regpy.vecsps import NumPyVectorSpace
    space = NumPyVectorSpace(3, dtype=dtype, as_real_space=as_real)
    assert isinstance(space, vs_base.VectorSpace)
    assert space.dimension == dim
    values = np.arange(1, dim+1, dtype=float)
    if not as_real:
        values = values + 1j * values[::-1]
    x = space.fromflat(values)
    assert x in space
    np.testing.assert_allclose(space.flatten(x), values)
    np.testing.assert_allclose(list(space.iter_coefficients(x)), values)
    out = space.empty()
    space.set_coefficients(out, iter(values))
    np.testing.assert_allclose(out, x)
    basis = [space.flatten(v).copy() for v in space.iter_basis()]
    np.testing.assert_allclose(basis, np.eye(dim))
    with pytest.raises(ValueError):
        space.set_coefficients(out, values[:-1])
    with pytest.raises(ValueError):
        space.fromflat(np.zeros(dim+1))
    with pytest.deprecated_call():
        assert space.size == dim
    with pytest.deprecated_call():
        assert space.realsize == dim


def test_shape_free_protocol():
    space = Hermitian3Space()
    assert isinstance(space, vs_base.VectorSpace)
    for name in ("shape", "ndim", "real_space", "complex_space"):
        assert not hasattr(space, name)
    x = space.fromflat(np.arange(9, dtype=float))
    assert x in space
    assert space.dimension == 9
    np.testing.assert_allclose(space.flatten(x), np.arange(9))
    basis = [space.flatten(v).copy() for v in space]
    np.testing.assert_allclose(basis, np.eye(9))
    out = space.empty()
    space.copyto(out, x)
    np.testing.assert_allclose(out, x)
    space.fill(out, 2.)
    np.testing.assert_allclose(space.flatten(out), np.full(9, 2.))
    assert space.zeros() in space and space.ones() in space
    assert space.randn() in space and space.rand() in space
    assert space.norm(x)**2 == pytest.approx(space.vdot(x, x))
    assert space.compatible(Hermitian3Space()) and space == Hermitian3Space()
    with pytest.raises(ValueError):
        space.read_only(x)[0, 0] = 2.
    total = vs_base.DirectSum(space, space)
    assert total.dimension == 18
    z = total.fromflat(np.arange(18, dtype=float))
    np.testing.assert_allclose(total.flatten(z), np.arange(18))


def test_direct_sum_scalar_field_coercion():
    from regpy.vecsps import NumPyVectorSpace
    r = NumPyVectorSpace(2)
    c = NumPyVectorSpace(3, dtype=complex, as_real_space=False)
    with pytest.warns(RuntimeWarning, match="Mixed scalar fields"):
        total = vs_base.DirectSum(r, c)
    assert total.as_real_space and r.as_real_space and c.as_real_space
    assert total.dimension == 8
    with pytest.raises(ValueError):
        r.as_real_space = False
    with pytest.raises(ValueError):
        total.as_real_space = False
    with pytest.raises(TypeError):
        c.as_real_space = "real"


@pytest.mark.parametrize("as_real", [True, False])
def test_masked_coordinates_preserve_scalar_field(as_real):
    from regpy.vecsps import NumPyVectorSpace
    full = NumPyVectorSpace(3, dtype=complex, as_real_space=as_real)
    masked = full.get_masked_space(np.array([True, False, True]))
    x = np.array([1+2j, 3+4j, 5+6j])
    assert masked.as_real_space == as_real
    assert masked.dimension == (4 if as_real else 2)
    restricted = masked.restrict(x)
    np.testing.assert_allclose(masked.fromflat(masked.flatten(restricted)), restricted)
    np.testing.assert_allclose(masked.extend(restricted), [1+2j, 0, 5+6j])
