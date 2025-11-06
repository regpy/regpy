import numpy as np

from regpy.vecsps.numpy import *

from .base_vecsps import vecsps_basics,vector_basics


def test_NumPyVectorSpace():
    errors = []
    errors += vecsps_basics(NumPyVectorSpace,test_methods=True,shape = (2,4), dtype = complex)
    errors += vecsps_basics(NumPyVectorSpace,test_methods=True,shape = (2,4), dtype = float)
    errors += vector_basics(NumPyVectorSpace,shape = (2,4),dtype = complex)
    errors += vector_basics(NumPyVectorSpace,shape = (2,4),dtype = float)
    if errors:
        # Combine all errors and raise a single AssertionError
        raise AssertionError("\n".join(errors))
    
def test_MeasureSpaceFcts():
    errors = []
    errors += vecsps_basics(MeasureSpaceFcts,test_methods=True,shape = (2,4), dtype = complex)
    errors += vecsps_basics(MeasureSpaceFcts,test_methods=True,shape = (2,4), dtype = float)
    errors += vecsps_basics(MeasureSpaceFcts,measure=4,shape=(5,6,7))

    m1=MeasureSpaceFcts(shape=(3,2))
    m2=MeasureSpaceFcts(measure=3*np.ones((3,2)))
    if m1==m2:
        errors.append("Two MeasureSpaceFcts that should not be equal are not.")
    try:
        m2.measure=1
    except Exception as e:
        errors.append(f"While trying to set the measure received exception {e}")
    if m1!=m2:
        errors.append("After setting the measure to constant one the two instances should be equal are but are not.")
    if errors:
        # Combine all errors and raise a single AssertionError
        raise AssertionError("\n".join(errors))
 
def test_GridFcts():
    errors = []
    errors += vecsps_basics(GridFcts,np.array([2,4,8]),np.array([-1,2,12,112]),test_methods=True,use_cell_measure=False, dtype = complex)
    errors += vecsps_basics(GridFcts,np.array([2,4,8]),np.array([-1,2,12,112]),test_methods=True,use_cell_measure=False, dtype = float)
    errors += vecsps_basics(GridFcts,np.array([2,4,8]),np.array([-1,0,5,15]),boundary_ext='zero')
    errors += vecsps_basics(GridFcts,np.array([2,4,8]),np.array([-1,0,5,15]),boundary_ext='const',ext_const=10)
    errors += vecsps_basics(GridFcts,np.array([2,4,8]),np.array([-1,0,5,15]),boundary_ext='const',ext_const=(1,(2,3)))

    gf=GridFcts(np.array([2,4,8]),np.array([-1,2,12,112]),use_cell_measure=False)
    if gf.measure.flat[0]!=1.0:
        errors.append(f"Not using cell measure should create a constant one measure but got measure = {gf.measure}")
    gf=GridFcts(np.array([2,4,8]),np.array([-1,0,5,15]))
    if not np.array_equal(gf.measure,np.array([[ 2,6,15,20],[3,9,22.5,30],[4,12,30,40]])):
        errors.append(f"The measure for a GridFcts with coords = {gf.coords} should be {np.array([[ 2,6,15,20],[3,9,22.5,30],[4,12,30,40]])} but got measure = {gf.measure}.")
    if errors:
        # Combine all errors and raise a single AssertionError
        raise AssertionError("\n".join(errors))

def test_UniformGridFcts():
    errors = []
    errors += vecsps_basics(UniformGridFcts,np.array([2,4,6]),np.array([-1,2,5,8]),test_methods=True, dtype = complex)
    errors += vecsps_basics(UniformGridFcts,np.array([2,4,6]),np.array([-1,2,5,8]),test_methods=True, dtype = float)
    
    gf=UniformGridFcts(np.array([2,4,6]),np.array([-1,2,5,8]))
    if gf.volume_elem!=6:
        errors.append(f"The volume element of {gf} with coords = {gf.coords} should be 6 but got {gf.volume_elem}.")
    try:
        gf.measure=3*np.ones((3,4))
        if gf.volume_elem!=3 or gf.measure.flat[0]!=3:
            errors.append(f"Setting the new measure succeeded but either the volume_elem {gf.volume_elem} or the measure {gf.measure} is not equal to the new value 3.")
    except Exception as e:
        errors.append(f"Trying to redefine the measure to 3 of {gf} failed with exception {e}.")
    if errors:
        # Combine all errors and raise a single AssertionError
        raise AssertionError("\n".join(errors))

def test_Prod():
    gf1=MeasureSpaceFcts(np.array([[2.0,4.0,8.0],[10,12,14]]))
    gf2=MeasureSpaceFcts(np.array([1.0,3.0]))
    errors = []
    errors += vecsps_basics(Prod,gf1,gf2,test_methods=True)
    prod=Prod(gf1,gf2)
    errors += vecsps_basics(Prod,prod,gf2,test_methods=True,flatten = True)
    errors += vecsps_basics(Prod,prod,gf2,test_methods=True,flatten = False)
    
    if errors:
        # Combine all errors and raise a single AssertionError
        raise AssertionError("\n".join(errors))

#     hprod=L2(prod)
#     solution=np.array([[2.,6.],[ 4.,12.],[ 8.,24.],[10.,30.],[12.,36.],[14.,42.]])
#     assert np.array_equal(hprod.gram._eval(np.ones((6,2))),solution)


