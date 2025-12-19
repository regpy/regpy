import numpy as np
import pytest

from regpy.vecsps.curve import *

from .base_vecsps import vecsps_basics,vector_basics
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)

def ShapeCurves(name,n=20,der=3):
    cls = globals()[name](n,der)
    try: 
        if der >= 0:
            _ = cls.z
        if der >= 1:
            _ = cls.zp
            _ = cls.zpabs
            _ = cls.normal
        if der >= 2:
            _ = cls.zpp
        if der >= 3:
            _ = cls.zppp
    except Exception as e:
        return [f"While tying to excess the constructed curve and its derivatives an exception {e} was raised."]
    return []

@pytest.mark.parametrize("shape, der",[
    ("kite",3),("peanut",3),("round_rect",2),("apple",3),("three_lobes",3),("pinched_ellipse",3),("smoothed_rectangle",3),("nonsym_shape",3),("circle",3)
])
def test_shapes(shape,der):
        ShapeCurves(shape,n=20,der=der)
  
def test_GenTrigSpc():
    vecsps_basics(GenTrigSpc,40)

    vs = GenTrigSpc(40)
    samples = np.asarray([[1.,0.,0.25],[1,0.25,0.5]])
    _ = vs.bd_eval(samples=samples,nvals=40,nderivs=3)

    
def test_StarTrigRadialFcts():
    vecsps_basics(StarTrigRadialFcts,40)

    vs = StarTrigRadialFcts(40)
    coeff = vs.sample(lambda t: np.sqrt(6*np.cos(1.5*t)**2+1)/3)
    _ = vs.eval_curve(coeffs=coeff)
