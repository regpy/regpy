from regpy.vecsps import MeasureSpaceFcts, GridFcts, UniformGridFcts
from regpy.vecsps import Prod
from regpy.hilbert import L2
import numpy as np


def test_base():
    m1=MeasureSpaceFcts(shape=(3,2))
    m2=MeasureSpaceFcts(measure=3*np.ones((3,2)))
    assert m1!=m2
    m2.measure=1
    assert m1==m2
    m3=MeasureSpaceFcts(measure=4,shape=(5,6,7))

def test_grid_functions():
    gf1=GridFcts(np.array([2,4,8]),np.array([-1,2,12,112]),use_cell_measure=False)
    assert gf1.measure==1.0
    gf2=GridFcts(np.array([2,4,8]),np.array([-1,0,5,15]))
    assert np.array_equal(gf2.measure,np.array([[ 2,6,15,20],[3,9,22.5,30],[4,12,30,40]]))
    gf3=GridFcts(np.array([2,4,8]),np.array([-1,0,5,15]),boundary_ext='zero')
    gf4=GridFcts(np.array([2,4,8]),np.array([-1,0,5,15]),boundary_ext='const',ext_const=10)
    gf5=GridFcts(np.array([2,4,8]),np.array([-1,0,5,15]),boundary_ext='const',ext_const=(1,(2,3)))

def test_uniform_grid_functions():
    gf=UniformGridFcts(np.array([2,4,6]),np.array([-1,2,5,8]))
    assert gf.volume_elem==6
    gf.measure=3*np.ones((3,4))
    assert gf.volume_elem==3
    assert gf.measure==3

def test_prod_measure_space():
    gf1=MeasureSpaceFcts(np.array([[2.0,4.0,8.0],[10,12,14]]))
    gf2=MeasureSpaceFcts(np.array([1.0,3.0]))
    prod=Prod(gf1,gf2)
    hprod=L2(prod)
    solution=np.array([[2.,6.],[ 4.,12.],[ 8.,24.],[10.,30.],[12.,36.],[14.,42.]])
    assert np.array_equal(hprod.gram._eval(np.ones((6,2))),solution)

