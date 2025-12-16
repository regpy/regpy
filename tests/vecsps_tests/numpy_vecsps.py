import numpy as np
import pytest

from regpy.vecsps.base import DirectSum
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
