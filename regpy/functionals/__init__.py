import logging

from regpy.vecsps import *
from regpy import hilbert

from .base import Functional,AbstractFunctional,SquaredNorm
from .numpy import *

__all__ = ["Functional", "L1", "Lpp", "TV", "KL", "RE", "Hub", "QuadIntv", "QuadNonneg", "QuadBil", "QuadLow", "QuadPosSemi", "HilbertNorm", "VFunc"]


L1 = AbstractFunctional('L1')
Lpp = AbstractFunctional('Lpp')
TV = AbstractFunctional('TV')
KL = AbstractFunctional('KL')
RE = AbstractFunctional("RE")
Hub = AbstractFunctional("Hub")
QuadIntv = AbstractFunctional("QuadIntv")
QuadNonneg = AbstractFunctional("QuadNonneg")
QuadBil = AbstractFunctional("QuadBil")
QuadLow = AbstractFunctional("QuadLow")
QuadPosSemi = AbstractFunctional("QuadPosSemi")
HilbertNorm = AbstractFunctional('HilbertNorm')
VFunc = AbstractFunctional('Vector Integral Functional')

def HilbertNormOnAbstractSpace(vecsp, h_space=hilbert.L2):
    return HilbertNorm(h_space(vecsp))

def _register_functionals():
    r"""Auxiliary method to register abstract functionals for various vector spaces. Using the decorator
    method described in `AbstractFunctional` does not work due to circular depenencies when
    loading modules.

    This is called from the `regpy` top-level module once, and can be ignored otherwise.
    """
    HilbertNorm.register(hilbert.HilbertSpace, SquaredNorm)
    HilbertNorm.register(VectorSpaceBase,HilbertNormOnAbstractSpace)

    L1.register(NumPyVectorSpace, L1Generic)
    L1.register(MeasureSpaceFcts, L1MeasureSpace)

    TV.register(NumPyVectorSpace, TVGeneric)
    TV.register(UniformGridFcts, TVUniformGridFcts)

    Lpp.register(MeasureSpaceFcts, LppPower)
 
    KL.register(MeasureSpaceFcts,KullbackLeibler)
    
    RE.register(MeasureSpaceFcts,RelativeEntropy)

    Hub.register(MeasureSpaceFcts,Huber)

    QuadIntv.register(MeasureSpaceFcts,QuadraticIntv)
    
    QuadNonneg.register(MeasureSpaceFcts,QuadraticNonneg)

    QuadBil.register(MeasureSpaceFcts,QuadraticBilateralConstraints)
    
    QuadLow.register(MeasureSpaceFcts,QuadraticLowerBound)
    
    QuadPosSemi.register(UniformGridFcts,QuadraticPositiveSemidef)

    VFunc.register(MeasureSpaceFcts,VectorIntegralFunctional)

    # Import of ngsolve functionals if possible to import 
    try:
        from .ngsolve import NgsL1,NgsTV

        L1.register(NgsVectorSpace, NgsL1)
        TV.register(NgsVectorSpace,NgsTV)
    except :
        logging.info("'Ngsolve' appears to be not installed not registering the respective functionls.")

