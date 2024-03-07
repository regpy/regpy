from regpy.vecsps import MeasureSpaceFcts
import numpy as np


def test_base():
    m1=MeasureSpaceFcts(shape=(3,2))
    m2=MeasureSpaceFcts(measure=3*np.ones((3,2)))
    assert m1!=m2
    m2.measure=np.ones((3,2))
    assert m1==m2
    m3=MeasureSpaceFcts(measure=3,shape=(5,6,7))

