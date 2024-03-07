from regpy.vecsps import MeasureSpaceFcts, GridFcts
import numpy as np


def test_base():
    m1=MeasureSpaceFcts(shape=(3,2))
    m2=MeasureSpaceFcts(measure=3*np.ones((3,2)))
    assert m1!=m2
    m2.measure=np.ones((3,2))
    assert m1==m2
    m3=MeasureSpaceFcts(measure=3,shape=(5,6,7))

def test_grid_functions():
    gf1=GridFcts(np.array([2,4,8]),np.array([-1,2,12,112]))
    print()
    assert np.array_equal(gf1.measure,np.ones((3,4)))
    gf2=GridFcts(np.array([2,4,8]),np.array([-1,0,5,15]),use_cell_measure=True)
    assert np.array_equal(gf2.measure,np.array([[ 2,6,15,20],[3,9,22.5,30],[4,12,30,40]]))
    gf3=GridFcts(np.array([2,4,8]),np.array([-1,0,5,15]),use_cell_measure=True,boundary_ext='zero')
    gf4=GridFcts(np.array([2,4,8]),np.array([-1,0,5,15]),use_cell_measure=True,boundary_ext='const',ext_const=10)
    gf5=GridFcts(np.array([2,4,8]),np.array([-1,0,5,15]),use_cell_measure=True,boundary_ext='const',ext_const=(1,(2,3)))

test_grid_functions()



# vs=[np.array([1,3,4,5,8]),np.array([1,3,4,5,8]),np.array([1,2,3])]

# ext_vs=[np.insert(v,[0,v.shape[0]],np.array([2*v[0]-v[1],2*v[-1]-v[-2]])) for v in vs]
# sum_s=[chr(k) for k in range(65,65+3)]
# print(','.join(sum_s))
# es=np.einsum(','.join(sum_s),*[0.5*(ext_v[2:]-ext_v[:-2]) for ext_v in ext_vs])

# print(es)