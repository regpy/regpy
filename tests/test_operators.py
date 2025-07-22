import numpy as np
from regpy.operators import *
from regpy.operators.convolution import *
import regpy.util.operator_tests as ot
from regpy import vecsps

def test_identity():
    #real
    dom=vecsps.VectorSpace((2,2))
    op=Identity(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-x)<1e-20)
    ot.test_operator(op)
    #complex
    dom=vecsps.VectorSpace((2,2),np.complex128)
    op=Identity(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-x)<1e-20)
    ot.test_operator(op)

def test_exponential():
    #real
    dom=vecsps.VectorSpace((2,2))
    op=Exponential(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.exp(x))<1e-10)
    ot.test_operator(op)
    #complex
    dom=vecsps.VectorSpace((2,2),np.complex128)
    op=Exponential(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.exp(x))<1e-10)
    ot.test_operator(op)

def test_real_part():
    #real
    dom=vecsps.VectorSpace((2,2))
    op=RealPart(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.real(x))<1e-20)
    ot.test_operator(op)
    #complex
    dom=vecsps.VectorSpace((2,2),np.complex128)
    op=RealPart(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.real(x))<1e-20)
    ot.test_operator(op)

def test_imaginary_part():
    #complex
    dom=vecsps.VectorSpace((2,2),np.complex128)
    op=ImaginaryPart(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.imag(x))<1e-20)
    ot.test_operator(op)

def test_zero():
    #real
    dom=vecsps.VectorSpace((2,2))
    op=Zero(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x))<1e-20)
    ot.test_operator(op)
    #complex
    dom=vecsps.VectorSpace((2,2),np.complex128)
    op=Zero(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x))<1e-20)
    ot.test_operator(op)

def test_squared_modulus():
    #real
    dom=vecsps.VectorSpace((2,2))
    op=SquaredModulus(domain=dom)
    x=dom.ones()
    x[0,0]=2
    assert np.max(np.abs(op(x)-np.abs(x)**2)<1e-10)
    ot.test_operator(op)
    #complex
    dom=vecsps.VectorSpace((2,2),np.complex128)
    op=SquaredModulus(domain=dom)
    x=dom.ones()*1j
    x[0,0]=2+1j
    assert np.max(np.abs(op(x)-np.abs(x)**2)<1e-10)
    ot.test_operator(op)

def test_coordinate_projection():
    #real
    dom=vecsps.VectorSpace((2,2))
    mask=np.array([[1,0],[0,1]],dtype=bool)
    op=CoordinateProjection(dom,mask)
    x=dom.ones()
    x[0,0]=2
    assert np.max(np.abs(op(x)-np.array([2,1]))<1e-10)
    ot.test_operator(op)
    #complex
    dom=vecsps.VectorSpace((2,2),np.complex128)
    mask=np.array([[1,0],[0,1]],dtype=bool)
    op=CoordinateProjection(dom,mask)
    x=1j*dom.ones()
    x[0,0]=2+1j
    assert np.max(np.abs(op(x)-np.array([2+1j,1j]))<1e-10)
    ot.test_operator(op)

def test_coordinate_mask():
    #real
    dom=vecsps.VectorSpace((2,2))
    mask=np.array([[1,0],[0,1]],dtype=bool)
    op=CoordinateMask(dom,mask)
    x=dom.ones()
    x[0,0]=2
    assert np.max(np.abs(op(x)-np.array([[2,0],[0,1]]))<1e-10)
    ot.test_operator(op)
    #complex
    dom=vecsps.VectorSpace((2,2),np.complex128)
    mask=np.array([[1,0],[0,0]],dtype=bool)
    op=CoordinateMask(dom,mask)
    x=1j*dom.ones()
    x[0,0]=2+1j
    assert np.max(np.abs(op(x)-np.array([[2+1j,0],[0,0]]))<1e-10)
    ot.test_operator(op)

def test_pow():#uses PtwMultiplication
    #real
    dom=vecsps.VectorSpace((2,2))
    mult_op=PtwMultiplication(dom,factor=2)
    op=Pow(mult_op,3)
    x=dom.ones()
    x[0,0]=2
    assert np.max(np.abs(op(x)-np.array([[16,8],[8,8]]))<1e-10)
    ot.test_operator(op)
    #complex
    dom=vecsps.VectorSpace((2,2),np.complex128)
    mult_op=PtwMultiplication(dom,factor=1j)
    op=Pow(mult_op,3)
    x=dom.ones()
    x[0,0]=2+1j
    assert np.max(np.abs(op(x)-np.array([[1-2j,-1j],[-1j,-1j]]))<1e-10)
    ot.test_operator(op)

def test_matrix_multiplication():
    op = MatrixMultiplication(np.random.rand(20,21),domain= UniformGridFcts(21),codomain=UniformGridFcts(20))
    ot.test_operator(op)
    op = MatrixMultiplication(np.random.rand(20,21)+1j*np.random.rand(20,21))
    ot.test_operator(op)