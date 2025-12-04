import numpy as np

from regpy.vecsps.numpy import *
from regpy.operators.numpy import *
from regpy.hilbert import *
from regpy.hilbert.numpy import *
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)

from .base_hilbert import hilbert_basics,collect_errors

def test_L2MeasureSpaceFcts():
    errors = []

    vs = MeasureSpaceFcts(np.arange(1,9).reshape(2,4))
    l2 = L2MeasureSpaceFcts(vs)
    
    errors += hilbert_basics(l2,test_methods=True)

    vs = MeasureSpaceFcts(measure=np.arange(1,9).reshape(2,4),dtype=complex)
    l2 = L2MeasureSpaceFcts(vs,weights=np.random.rand(8).reshape(2,4))
    
    errors += hilbert_basics(l2,test_methods=True)

    collect_errors(L2MeasureSpaceFcts,errors)

def test_L2UniformGridFcts():
    errors = []

    vs = UniformGridFcts(4,2)
    l2 = L2UniformGridFcts(vs)
    
    errors += hilbert_basics(l2,test_methods=True)

    vs = UniformGridFcts(4,2,dtype=complex)
    l2 = L2UniformGridFcts(vs,weights=np.random.rand(8).reshape(4,2))
    
    errors += hilbert_basics(l2,test_methods=True)

    collect_errors(L2UniformGridFcts,errors)

def test_SobolevUniformGridFcts():
    errors = []

    vs = UniformGridFcts(4,2)
    sob = SobolevUniformGridFcts(vs)
    
    errors += hilbert_basics(sob,test_methods=True)

    vs = UniformGridFcts(4,2,dtype=complex)
    sob = SobolevUniformGridFcts(vs,index = 2.5)
    
    errors += hilbert_basics(sob,test_methods=True)

    collect_errors(SobolevUniformGridFcts,errors)

def test_HmDomain():
    errors = []

    vs = UniformGridFcts(10,6)
    mask = np.zeros((10,6))
    mask[:,0] = 1
    mask[:,-1] = 1

    hm = HmDomain(vs,mask=mask,index=2)
    
    errors += hilbert_basics(hm,test_methods=True)

    vs = UniformGridFcts(4,2,dtype=complex)

    hm = HmDomain(vs)
    
    errors += hilbert_basics(hm,test_methods=True)

    collect_errors(HmDomain,errors)