import numpy as np
import pytest

from regpy.vecsps.numpy import *
from regpy.operators.numpy import *
from regpy.hilbert import *
from regpy.hilbert.numpy import *
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)

from .base_hilbert import hilbert_basics

@pytest.mark.parametrize("vs", [ 
    MeasureSpaceFcts(np.arange(1,9).reshape(2,4)),
    MeasureSpaceFcts(measure=np.arange(1,9).reshape(2,4),dtype=complex)
])
def test_L2MeasureSpaceFcts(vs):
    l2 = L2MeasureSpaceFcts(vs)
    hilbert_basics(l2,test_methods=True)

@pytest.mark.parametrize("vs", [ 
    UniformGridFcts(4,2),
    UniformGridFcts(4,2,dtype=complex)
])
def test_L2UniformGridFcts(vs):
    l2 = L2UniformGridFcts(vs)
    hilbert_basics(l2,test_methods=True)

@pytest.mark.parametrize("vs", [ 
    UniformGridFcts(4,2),
    UniformGridFcts(4,2,dtype=complex)
])
def test_SobolevUniformGridFcts(vs):
    sob = SobolevUniformGridFcts(vs)
    hilbert_basics(sob,test_methods=True)

class TestHmDomain():
    errors = []

    mask_1 = np.zeros((10,6))
    mask_1[:,0] = 1
    mask_1[:,-1] = 1
    mask_2 = np.zeros((4,2))
    mask_2[:,0] = 1
    mask_2[:,-1] = 1

    @pytest.mark.parametrize("vs,mask,h",[ 
        (UniformGridFcts(10,6),mask_1,"normalized"),
        (UniformGridFcts(4,2,dtype=complex),None,"normalized"),
        (UniformGridFcts(4,2,dtype=complex),mask_2,"normalized"),
        (UniformGridFcts(10,6),mask_1,"physical"),
        (UniformGridFcts(4,2,dtype=complex),None,"physical"),
    ])
    def test_basics(self,vs,mask,h):
        hm = HmDomain(vs,mask=mask,h=h,index=2)
        hilbert_basics(hm,test_methods=True)
