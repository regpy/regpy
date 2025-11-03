import numpy as np

from regpy.vecsps.curve import *

from .base_vecsps import vecsps_basics,vector_basics


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

def test_shapes():
    shapes = ["kite","peanut","round_rect","apple","three_lobes","pinched_ellipse","smoothed_rectangle","nonsym_shape","circle"]
    ders = {"kite": 3,"peanut":3,"round_rect":2,"apple":3,"three_lobes":3,"pinched_ellipse":3,"smoothed_rectangle":3,"nonsym_shape":3,"circle":3}
    errors = []
    for shape in shapes:
        errors += ShapeCurves(shape,n=20,der=ders[shape])
    if errors:
        # Combine all errors and raise a single AssertionError
        raise AssertionError("\n".join(errors))
    
def test_GenTrigDiscr():
    errors = []
    errors += vecsps_basics(GenTrigDiscr,40)

    vs = GenTrigDiscr(40)
    coeff = np.asarray([1,0.5,0.25,1,0.25,0.5])
    _ = vs.bd_eval(coeffs=coeff,nvals=40,nderivs=3)
    _ = vs.z
    _ = vs.zp
    _ = vs.zpp
    _ = vs.zppp
    _ = vs.zpabs
    _ = vs.normal
    _ = vs.der_normal
    _ = vs.adjoint_der_normal
    
def test_StarTrigDiscr():
    errors = []
    errors += vecsps_basics(StarTrigDiscr,40)

    vs = StarTrigDiscr(40)
    coeff = vs.sample(lambda t: np.sqrt(6*np.cos(1.5*t)**2+1)/3)
    _ = vs.eval_curve(coeffs=coeff)
