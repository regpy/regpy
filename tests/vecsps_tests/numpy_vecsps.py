# Poisson tests revision: 20260920-real-only
import numpy as np
import pytest

from regpy.vecsps.base import DirectSum
from regpy.vecsps.maskable import MaskableVectorSpace, MaskableDirectSum
from regpy.vecsps.numpy import *
from regpy.util import Errors

from .base_vecsps import vecsps_basics,vector_basics
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)

@pytest.mark.parametrize("shape, dtype",[
    ((2,4,3,7),float),
    ((2,4,3,7),complex),
    ((2,4),float),
    ((2,4),complex)
])
class TestNumPyVectorSpace():
    def test_vecsps_basic(self, shape, dtype):
        vecsps_basics(NumPyVectorSpace, shape,test_methods=True, dtype = dtype)

    def test_vector_basic(self, shape, dtype):
        vector_basics(NumPyVectorSpace, shape, dtype = dtype)
        d = NumPyVectorSpace(shape,dtype=dtype)
        v_1 = d.randn()
        assert v_1.imag == pytest.approx(-v_1.conj().imag),f"Tying to compare v.imag with v.conj().imag failed for {v_1}"


class TestMeasureSpaceFcts():
    @pytest.mark.parametrize("measure, shape, dtype",[
        (3,(2,4,3,7),float),
        (1.6,(2,4,3,7),complex),
        (None,(2,4),float),
        (None,(2,4),complex)
    ])
    def test_vs_basic(self,measure,shape,dtype):
        vecsps_basics(MeasureSpaceFcts,test_methods=True, measure = measure,shape = shape, dtype = dtype)

    @pytest.mark.parametrize("measure, shape, dtype",[
        (3,(2,4,3,7),float),
        (1.6,(2,4,3,7),complex),
        (None,(2,4),float),
        (None,(2,4),complex)
    ])
    def test_vector_basics(self,measure,shape,dtype):
        vector_basics(MeasureSpaceFcts, measure = measure,shape = shape, dtype = dtype)
        d = MeasureSpaceFcts(measure = measure,shape = shape, dtype = dtype)
        v_1 = d.randn()
        assert v_1.imag == pytest.approx(-v_1.conj().imag),f"Tying to compare v.imag with v.conj().imag failed for {v_1}"


    def test_equality(self):
        m1=MeasureSpaceFcts(shape=(3,2))
        m2=MeasureSpaceFcts(measure=3*np.ones((3,2)))
        assert m1!=m2, Errors.failed_test(f"Two MeasureSpaceFcts that should not be equal are not.",MeasureSpaceFcts)
    
        m2.measure=1
        assert m1==m2, Errors.failed_test("After setting the measure to constant one the two instances should be equal are but are not.",MeasureSpaceFcts)

@pytest.mark.parametrize("tuples",[
    ((3,(2,4,3,7),float),(1.6,(2,4,3,7),float)),
    ((None,(2,4),complex),(None,(2,4),complex)),
    ((None,(2,4),float),(None,(2,4),complex),(1.6,(2,4,3,7),float)),
])
def test_DirectSum_vector_basics(tuples):
    vector_basics(DirectSum,*[MeasureSpaceFcts(measure=t[0],shape=t[1], dtype = t[2]) for t in tuples])
    d = DirectSum(*[MeasureSpaceFcts(measure=t[0],shape=t[1], dtype = t[2]) for t in tuples])
    v_1 = d.randn()
    assert all(tuple(v_i.imag == pytest.approx(-v_i.conj().imag) for v_i in v_1)),f"Tying to compare v.imag with v.conj().imag failed for {v_1}"


class TestGridFcts():
    errors = []

    @pytest.mark.parametrize("coords, use_cell_measure, dtype, boundary_ext, ext_const",[
        ([np.array([2,4,8]),np.array([-1,2,12,112])],False, complex, 'sym', None),
        ([np.array([2,4,8]),np.array([-1,2,12,112])],False, float, 'sym', None),
        ([np.array([2,4,8]),np.array([-1,0,5,15])], False, float,'zero',None),
        ([np.array([2,4,8]),np.array([-1,0,5,15])],False, float, 'const',10),
        ([np.array([2,4,8]),np.array([-1,0,5,15])],False,float,'const',(1,(2,3)))
    ])
    def test_vs_basic(self,coords, use_cell_measure, dtype, boundary_ext, ext_const):
        vecsps_basics(GridFcts,*coords,test_methods=True,use_cell_measure=use_cell_measure, dtype = dtype,boundary_ext = boundary_ext, ext_const=ext_const)

    @pytest.mark.parametrize("coords, use_cell_measure, measure_compare",[
        ([np.array([2,4,8]),np.array([-1,2,12,112])], False,1.0),
        ([np.array([2,4,8]),np.array([-1,0,5,15])], True,np.array([[ 2,6,15,20],[3,9,22.5,30],[4,12,30,40]]))
    ])
    def test_measure_comp(self,coords, use_cell_measure, measure_compare):
        gf=GridFcts(*coords,use_cell_measure=use_cell_measure)
        assert gf.measure == pytest.approx(measure_compare), Errors.failed_test(f"Measure computation is wrong",GridFcts)

class Test_UniformGridFcts():

    @pytest.mark.parametrize("coords, dtype",[
        ([np.array([2,4,6]),np.array([-1,2,5,8])], complex),
        ([np.array([2,4,6]),np.array([-1,2,5,8])], float),
        ([10,(-1,1,10)], complex),
        ([(-1,1,10),(-1,1,10),(-1,1,10)], float),
    ])
    def test_vs_basics(self,coords,dtype):
        vecsps_basics(UniformGridFcts,*coords,test_methods=True, dtype = dtype)
    
    @pytest.mark.parametrize("coords, measure_compare",[
        ([np.array([2,4,6]),np.array([-1,2,5,8])],6.),
    ])
    def test_measure_comp(self,coords, measure_compare):
        gf=UniformGridFcts(*coords)
        assert gf.measure == pytest.approx(measure_compare), Errors.failed_test(f"The volume element of {gf} with coords = {gf.coords} should be {measure_compare} but got {gf.volume_elem}.")
                                                                            
    def test_set_measure(self):
        gf=UniformGridFcts(3,4)
        gf.measure=3*np.ones((3,4))
        assert gf.volume_elem == pytest.approx(3), Errors.failed_test(f"Setting the new measure succeeded but either the volume_elem {gf.volume_elem} or the measure {gf.measure} is not equal to the new value 3.")

def test_Prod():
    gf1=MeasureSpaceFcts(np.array([[2.0,4.0,8.0],[10,12,14]]))
    gf2=MeasureSpaceFcts(np.array([1.0,3.0]))

    vecsps_basics(Prod,gf1,gf2,test_methods=True)
    prod=Prod(gf1,gf2)
    vecsps_basics(Prod,prod,gf2,test_methods=True,flatten = True)
    vecsps_basics(Prod,prod,gf2,test_methods=True,flatten = False)


@pytest.mark.parametrize("dtype", [float, complex])
def test_numpy_masked_storage(dtype):
    space = NumPyVectorSpace((2, 3), dtype=dtype)
    assert isinstance(space, MaskableVectorSpace)
    rng = np.random.default_rng(20260919)
    x = rng.normal(size=space.shape).astype(dtype)
    if dtype is complex:
        x += 1j * rng.normal(size=space.shape)
    original = x.copy()
    mask = np.array([[True, False, True], [False, True, False]])
    out = space.zeros()
    space.fill(out, -2.)
    space.copyto(out, x, where=mask)
    np.testing.assert_array_equal(out, np.where(mask, x, -2.))
    space.fill(out, 7., where=~mask)
    np.testing.assert_array_equal(out, np.where(mask, x, 7.))
    restricted = space.get_masked_space(mask)
    np.testing.assert_array_equal(restricted.restrict(x), x[mask])
    np.testing.assert_array_equal(restricted.extend(x[mask]), np.where(mask, x, 0.))
    np.testing.assert_array_equal(x, original)


def test_numpy_mask_shape_and_comparisons():
    space = NumPyVectorSpace((2, 3))
    x = np.array([[-1., 0., 2.], [0., -3., 4.]])
    assert space.is_ordered
    np.testing.assert_array_equal(space.IfPos(x), x > 0)
    np.testing.assert_array_equal(space.geq_mask(x, space.zeros()), x >= 0)
    np.testing.assert_array_equal(space.eq_mask(x, space.zeros()), x == 0)
    assert space.nrTrues(space.geq_mask(x, space.zeros())) == 4
    assert not space.is_mask(np.ones((2, 3)))
    assert not space.is_mask(np.ones(6, dtype=bool))
    assert not space.is_mask(np.ones((1, 3), dtype=bool))
    for mask in (np.ones((2, 3)), np.ones(6, dtype=bool)):
        with pytest.raises(ValueError):
            space.copyto(space.zeros(), x, where=mask)


@pytest.mark.parametrize("dtypes, complex_expected, ordered_expected", [
    ((float, float), False, True),
    ((complex, complex), True, False),
    ((float, complex), False, False),
])
def test_direct_sum_scalar_field_regression(dtypes, complex_expected, ordered_expected):
    # A mixed real/complex direct sum is only closed under real scalars.
    space = DirectSum(*(NumPyVectorSpace(3, dtype=d) for d in dtypes))
    assert isinstance(space, MaskableDirectSum)
    assert space.is_complex == complex_expected
    assert space.is_ordered == ordered_expected
    x = space.ones()
    assert 2.5 * x in space
    np.testing.assert_allclose(space.vdot(x, x), 6.)
    if complex_expected:
        assert (1. + 2j) * x in space
    else:
        assert (1. + 2j) * x not in space


@pytest.mark.parametrize("as_real", [True, False])
def test_poisson_rejects_complex_spaces(as_real):
    space = NumPyVectorSpace((2, 3), dtype=complex, as_real_space=as_real)
    for rates in (space.ones(), np.ones(space.shape)):
        with pytest.raises(TypeError, match="real storage"):
            space.poisson(rates, seed=20260920)


def test_poisson_real_space_seeded():
    space = NumPyVectorSpace((2, 3))
    rates = np.array([[0., 1., 2.], [3., 4., 5.]])
    sample = space.poisson(rates, seed=20260920)
    np.testing.assert_array_equal(sample, space.poisson(rates, seed=20260920))
    assert sample in space
    assert np.all(sample >= 0)
    np.testing.assert_array_equal(sample, np.floor(sample))
    assert sample[0, 0] == 0


@pytest.mark.parametrize("bad_rate", [-1., np.nan, np.inf])
def test_poisson_rejects_invalid_rates(bad_rate):
    space = NumPyVectorSpace(3)
    rates = np.ones(3)
    rates[1] = bad_rate
    with pytest.raises(ValueError, match="Poisson rates"):
        space.poisson(rates, seed=20260920)


def test_complex_prod_uses_storage_dimensions():
    # Default complex spaces are realified: dimension is twice storage size.
    a = NumPyVectorSpace(2, dtype=complex)
    b = NumPyVectorSpace(3, dtype=complex)
    product_space = Prod(a, b)
    assert product_space.shape == (2, 3)
    assert product_space.dimension == 12
    x = np.array([1+2j, 3-4j])
    y = np.array([2j, 1-1j, 3.])
    np.testing.assert_allclose(product_space.product(x, y), np.outer(x, y))
