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
    ("Kite",3),("Peanut",3),("Round_rect",2),("Apple",3),("Three_lobes",3),("Pinched_ellipse",3),("Smoothed_rectangle",3),("Nonsym_shape",3),("Circle",3)
])
def test_shapes(shape,der):
        ShapeCurves(shape,n=20,der=der)
  
def test_GenTrigSpc():
    vecsps_basics(GenTrigSpc,10,n=50)

    vs = GenTrigSpc(4,n=40)
    samples = np.asarray([[1.,0.],[0.25,1],[0.25,0.5],[1.,1.]])
    _ = vs.coeff2curve(coeff=samples,nderivs=3)

    
def test_StarTrigRadialFcts():
    vecsps_basics(StarTrigRadialFcts,40,n=50)

    vs = StarTrigRadialFcts(40,n=40)
    curve = vs.radialfct2curve(lambda t: np.sqrt(6*np.cos(1.5*t)**2+1)/3)
    _ = vs.coeff2curve(coeff=curve.radial())
