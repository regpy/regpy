import numpy as np
from regpy.operators import *
from regpy.operators.convolution import *
import regpy.util.operator_tests as ot
from regpy import vecsps

def test_identity():
    #real
    dom=vecsps.NumPyVectorSpace((2,2))
    op=Identity(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-x)<1e-20)
    ot.test_operator(op)
    #complex
    dom=vecsps.NumPyVectorSpace((2,2),np.complex128)
    op=Identity(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-x)<1e-20)
    ot.test_operator(op)

def test_exponential():
    #real
    dom=vecsps.NumPyVectorSpace((2,2))
    op=Exponential(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.exp(x))<1e-10)
    ot.test_operator(op)
    #complex
    dom=vecsps.NumPyVectorSpace((2,2),np.complex128)
    op=Exponential(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.exp(x))<1e-10)
    ot.test_operator(op)

def test_real_part():
    #real
    dom=vecsps.NumPyVectorSpace((2,2))
    op=RealPart(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.real(x))<1e-20)
    ot.test_operator(op)
    #complex
    dom=vecsps.NumPyVectorSpace((2,2),np.complex128)
    op=RealPart(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.real(x))<1e-20)
    ot.test_operator(op)

def test_imaginary_part():
    #complex
    dom=vecsps.NumPyVectorSpace((2,2),np.complex128)
    op=ImaginaryPart(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x)-np.imag(x))<1e-20)
    ot.test_operator(op)

def test_zero():
    #real
    dom=vecsps.NumPyVectorSpace((2,2))
    op=Zero(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x))<1e-20)
    ot.test_operator(op)
    #complex
    dom=vecsps.NumPyVectorSpace((2,2),np.complex128)
    op=Zero(domain=dom)
    x=dom.randn()
    assert np.max(np.abs(op(x))<1e-20)
    ot.test_operator(op)

def test_squared_modulus():
    #real
    dom=vecsps.NumPyVectorSpace((2,2))
    op=SquaredModulus(domain=dom)
    x=dom.ones()
    x[0,0]=2
    assert np.max(np.abs(op(x)-np.abs(x)**2)<1e-10)
    ot.test_operator(op)
    #complex
    dom=vecsps.NumPyVectorSpace((2,2),np.complex128)
    op=SquaredModulus(domain=dom)
    x=dom.ones()*1j
    x[0,0]=2+1j
    assert np.max(np.abs(op(x)-np.abs(x)**2)<1e-10)
    ot.test_operator(op)

def test_coordinate_projection():
    #real
    dom=vecsps.NumPyVectorSpace((2,2))
    mask=np.array([[1,0],[0,1]],dtype=bool)
    op=CoordinateProjection(dom,mask)
    x=dom.ones()
    x[0,0]=2
    assert np.max(np.abs(op(x)-np.array([2,1]))<1e-10)
    ot.test_operator(op)
    #complex
    dom=vecsps.NumPyVectorSpace((2,2),np.complex128)
    mask=np.array([[1,0],[0,1]],dtype=bool)
    op=CoordinateProjection(dom,mask)
    x=1j*dom.ones()
    x[0,0]=2+1j
    assert np.max(np.abs(op(x)-np.array([2+1j,1j]))<1e-10)
    ot.test_operator(op)

def test_coordinate_mask():
    #real
    dom=vecsps.NumPyVectorSpace((2,2))
    mask=np.array([[1,0],[0,1]],dtype=bool)
    op=CoordinateMask(dom,mask)
    x=dom.ones()
    x[0,0]=2
    assert np.max(np.abs(op(x)-np.array([[2,0],[0,1]]))<1e-10)
    ot.test_operator(op)
    #complex
    dom=vecsps.NumPyVectorSpace((2,2),np.complex128)
    mask=np.array([[1,0],[0,0]],dtype=bool)
    op=CoordinateMask(dom,mask)
    x=1j*dom.ones()
    x[0,0]=2+1j
    assert np.max(np.abs(op(x)-np.array([[2+1j,0],[0,0]]))<1e-10)
    ot.test_operator(op)

def test_pow():#uses PtwMultiplication
    #real
    dom=vecsps.NumPyVectorSpace((2,2))
    mult_op=PtwMultiplication(dom,factor=2)
    op=Pow(mult_op,3)
    x=dom.ones()
    x[0,0]=2
    assert np.max(np.abs(op(x)-np.array([[16,8],[8,8]]))<1e-10)
    ot.test_operator(op)
    #complex
    dom=vecsps.NumPyVectorSpace((2,2),np.complex128)
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

def test_PtwMultiplication():
    dom = vecsps.VectorSpace((3,2))
    factor = dom.rand()  
    op = PtwMultiplication(dom,factor)
    ot.test_operator(op)
    dom = vecsps.VectorSpace((3,2),dtype=np.complex128)
    factor = dom.rand()  
    op = PtwMultiplication(dom,factor)
    ot.test_operator(op)
    
def test_OuterShift():
    dom=vecsps.VectorSpace((10,5),np.complex128)
    offset = dom.rand()
    op_unshifted = Exponential(domain=dom)
    op_shifted = OuterShift(op_unshifted,offset)
    ot.test_operator(op_shifted)
    x = dom.rand()
    assert np.max(np.abs(op_unshifted(x)-op_shifted(x)+offset))<1e-15

      
def test_InnerShift():
    dom=vecsps.VectorSpace((10,5),np.complex128)
    offset = dom.rand()
    op_unshifted = Exponential(domain=dom)
    op_shifted = InnerShift(op_unshifted,offset)
    ot.test_operator(op_shifted)

def test_power():
    #real
    dom=vecsps.UniformGridFcts(2,2)
    op_power=Power(3.0,dom,integer=True)
    x=np.arange(4).reshape(2,2)
    assert np.max(np.abs(op_power(x)-np.array([[0,1],[8,27]])))<1e-15
    ot.test_operator(op_power)
    op_power_float=Power(0.5,dom)
    x=np.arange(1,5).reshape(2,2)
    assert np.max(np.abs(op_power_float(x)-np.array([[1,np.sqrt(2)],[np.sqrt(3),2]])))<1e-15
    y, deriv = op_power_float.linearize(x)
    ot.test_operator(deriv)
    h = op_power_float.domain.rand()
    normh = np.linalg.norm(h)
    g = deriv(h)
    seq=[np.linalg.norm((op_power_float(x + step * h) - y) / step - g) / normh for step in [10**k for k in range(-1, -8, -1)]]
    assert all(seq_i >= seq_j for seq_i, seq_j in zip(seq, seq[1:])),f"convergence errors: {seq}"
    #complex
    dom=vecsps.UniformGridFcts(2,2,dtype=np.complex128)
    op_power=Power(3.0,dom,integer=True)
    x=np.array([[1j,2],[1+1j,-2j]])
    assert np.max(np.abs(op_power(x)-np.array([[-1j,8],[-2+2j,8j]])))<1e-15
    ot.test_operator(op_power)
    op_power_float=Power(0.5,dom)
    x=np.array([[1j,2],[1+1j,-2j]])
    print(op_power_float(x))
    assert np.max(np.abs(op_power_float(x)-np.array([[(1+1j)*np.sqrt(2)/2,np.sqrt(2)],[2**(0.25)*(np.cos(np.pi/8)+np.sin(np.pi/8)*1j),1-1j]])))<1e-15
    ot.test_operator(op_power_float)

def test_direct_sum():#uses Exponential and PtwMultiplication
    #real
    dom1=vecsps.UniformGridFcts(2,2)
    dom2=vecsps.UniformGridFcts(2,3,4)
    op1=Exponential(dom1)
    op2=PtwMultiplication(dom2,3)
    op3=PtwMultiplication(dom2,4)
    op=DirectSum(op1,op2)
    x=np.arange(28)
    assert np.max(np.abs(op(x)-np.array([1,np.exp(1),np.exp(2),np.exp(3)]+[3*i for i in range(4,28)])))<1e-15
    ot.test_operator(op)
    op=DirectSum(op2,op3)
    x=np.arange(48)
    assert np.max(np.abs(op(x)-np.array([3*i for i in range(0,24)]+[4*i for i in range(24,48)])))<1e-15
    ot.test_operator(op)
    #complex
    dom1=vecsps.UniformGridFcts(2,2,dtype=np.complex128)
    dom2=vecsps.UniformGridFcts(2,3,4)
    dom3=vecsps.UniformGridFcts(2,dtype=np.complex128)
    op1=Exponential(dom1)
    op2=PtwMultiplication(dom2,3)
    op3=Exponential(dom3)
    op=DirectSum(op1,op2)
    x=np.array([1,0,0,0,0,np.pi,2,-np.pi]+[i for i in range(24)])
    y=np.array([np.exp(1),0,1,0,-1,0,-np.exp(2),0]+[3*i for i in range(24)])
    assert np.max(np.abs(op(x)-y))<1e-10
    ot.test_operator(op)
    x=np.array([1,0,0,0,0,np.pi,2,-np.pi,2,0,3,np.pi])
    y=np.array([np.exp(1),0,1,0,-1,0,-np.exp(2),0,np.exp(2),0,-np.exp(3),0])
    op=DirectSum(op1,op3)
    assert np.max(np.abs(op(x)-y))<1e-10
    ot.test_operator(op)

# def test_vector_of_operators():#uses Exponential and PtwMultiplication
#     #real
#     dom=vecsps.UniformGridFcts(2,2)
#     op1=Exponential(dom)
#     op2=PtwMultiplication(dom,2)
#     op3=PtwMultiplication(dom,3)
#     op=VectorOfOperators([op1,op2,op3])
#     x=np.arange(4).reshape(2,2)
#     y=np.array([1,np.exp(1),np.exp(2),np.exp(3)]+[2*i for i in range(4)]+[3*i for i in range(4)])
#     assert np.max(np.abs(op(x)-y))<1e-15
#     ot.test_operator(op)
#     #complex
#     dom=vecsps.UniformGridFcts(2,2,dtype=np.complex128)
#     op1=Exponential(dom)
#     op2=PtwMultiplication(dom,2j)
#     op3=PtwMultiplication(dom,3)
#     op=VectorOfOperators([op1,op2,op3])
#     x=np.arange(4).reshape(2,2)
#     y=np.array([1,0,np.exp(1),0,np.exp(2),0,np.exp(3),0,0,0,0,2,0,4,0,6,0,0,3,0,6,0,9,0])
#     print(op(x))
#     print(y)
#     assert np.max(np.abs(op(x)-y))<1e-15
#     ot.test_operator(op)

def test_matrix_of_operators():#uses Exponential and PtwMultiplication
    #real
    dom=vecsps.UniformGridFcts(2,2)
    op1=Exponential(dom)
    op2=PtwMultiplication(dom,2)
    op3=PtwMultiplication(dom,3)
    op4=PtwMultiplication(dom,4)
    ops=[[op1,op2,None],[None,op3,op4]]
    op=MatrixOfOperators(ops)
    x=np.arange(8)
    y=np.array([1,np.exp(1),np.exp(2),np.exp(3),12,17,22,27,16,20,24,28])
    assert np.max(np.abs(op(x)-y))<1e-15
    ot.test_operator(op)

def test_padding_operator():
    #real
    dom=vecsps.UniformGridFcts(2,2)
    op=PaddingOperator(dom,((1,2),(3,4)))
    y=np.zeros((5,9))
    y[1:3,3:5]=1
    assert np.max(np.abs(op(dom.ones())-y))<1e-15
    ot.test_operator(op)
    #complex
    dom=vecsps.UniformGridFcts(2,2,dtype=np.complex128)
    op=PaddingOperator(dom,((1,2),(3,4)))
    y=np.zeros((5,9),dtype=np.complex128)
    y[1:3,3:5]=1j
    assert np.max(np.abs(op(dom.ones()*1j)-y))<1e-15
    ot.test_operator(op)

def test_convolution_operator():
    #real
    dom=vecsps.UniformGridFcts(10,10)
    kernel=np.arange(15*7).reshape(15,7)
    op=ConvolutionOperator(dom,fourier_multiplier=kernel,pad_amount=((2,3),(1,2)))
    ot.test_operator(op)
    kernel2=lambda a,b:a*b*1j
    op=ConvolutionOperator(dom,fourier_multiplier=kernel2,pad_amount=((2,3),(1,2)))
    ot.test_operator(op)
    #complex
    dom=vecsps.UniformGridFcts(10,10,dtype=np.complex128)
    kernel=np.arange(15*13).reshape(15,13)
    op=ConvolutionOperator(dom,fourier_multiplier=kernel,pad_amount=((2,3),(1,2)))
    ot.test_operator(op)
    kernel2=lambda a,b:a*np.conj(b)
    op=ConvolutionOperator(dom,fourier_multiplier=kernel2,pad_amount=((2,3),(1,2)))
    ot.test_operator(op)

def test_gaussian_blur():
    #real
    dom=vecsps.UniformGridFcts(10,10)
    op=GaussianBlur(dom,5,(2,1),pad_amount=((2,3),(1,2)),first_conv_axis=1)
    ot.test_operator(op)
    #complex
    dom=vecsps.UniformGridFcts(10,10,dtype=np.complex128)
    op=GaussianBlur(dom,5,(2,1),pad_amount=((2,3),(1,2)))
    ot.test_operator(op)

def test_exponential_convolution():
    #real
    dom=vecsps.UniformGridFcts(10,10)
    op=ExponentialConvolution(dom,0.5,pad_amount=((2,3),(1,2)),first_conv_axis=1)
    ot.test_operator(op)
    #complex
    dom=vecsps.UniformGridFcts(10,10,dtype=np.complex128)
    op=ExponentialConvolution(dom,0.5,pad_amount=((2,3),(1,2)))
    ot.test_operator(op)

def test_fresnel_propagator():
    #real
    dom=vecsps.UniformGridFcts(10,10)
    op=ExponentialConvolution(dom,2.5,pad_amount=((2,3),(1,2)),first_conv_axis=1)
    ot.test_operator(op)
    #complex
    dom=vecsps.UniformGridFcts(10,10,dtype=np.complex128)
    op=ExponentialConvolution(dom,2.5,pad_amount=((2,3),(1,2)))
    ot.test_operator(op)
