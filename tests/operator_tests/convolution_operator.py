import numpy as np

from regpy.vecsps.numpy import *
from regpy.operators.convolution import *

from .base_operator import op_basics_wrapper,op_evaluation_and_ot,collect_errors

def test_PaddingOperator():
    errors = []
    vs = UniformGridFcts(2,2)
    errors += op_basics_wrapper(PaddingOperator,vs,test_methods=True)

    op=PaddingOperator(vs,((1,2),(3,4)))
    x=vs.ones()
    res=np.zeros((5,9))
    res[1:3,3:5]=1

    errors += op_evaluation_and_ot(op,x=x,res=res)

    vs = UniformGridFcts(2,2,dtype=complex)
    op=PaddingOperator(vs,((1,2),(3,4)))
    x=vs.ones()*1j
    res=np.zeros((5,9),dtype=complex)
    res[1:3,3:5]=1j

    errors += op_evaluation_and_ot(op,x=x,res=res)

    collect_errors(PaddingOperator,errors)

def test_ConvolutionOperator():
    errors = []
    vs = UniformGridFcts(10,10)
    kernel = np.arange(15*7).reshape(15,7)
    errors += op_basics_wrapper(ConvolutionOperator,vs,test_methods=True, rel_tol_norm = 1e-3,fourier_multiplier=kernel,pad_amount=((2,3),(1,2)))

    op=ConvolutionOperator(vs,fourier_multiplier=kernel,pad_amount=((2,3),(1,2)))

    errors += op_evaluation_and_ot(op)

    kernel2=lambda a,b:a*b*1j
    op=ConvolutionOperator(vs,fourier_multiplier=kernel2,pad_amount=((2,3),(1,2)))

    errors += op_evaluation_and_ot(op)

    #complex
    vs=UniformGridFcts(10,10,dtype=np.complex128)
    kernel=np.arange(15*13).reshape(15,13)
    op=ConvolutionOperator(vs,fourier_multiplier=kernel,pad_amount=((2,3),(1,2)))

    errors += op_evaluation_and_ot(op)

    kernel2=lambda a,b:a*np.conj(b)
    op=ConvolutionOperator(vs,fourier_multiplier=kernel2,pad_amount=((2,3),(1,2)))

    errors += op_evaluation_and_ot(op)

    collect_errors(ConvolutionOperator,errors)

def test_GaussianBlur():
    errors = []
    vs = UniformGridFcts(10,10)
    errors += op_basics_wrapper(GaussianBlur,vs,5,(2,1),test_methods=True, pad_amount=((2,3),(1,2)),first_conv_axis=1)

    op=GaussianBlur(vs,5,(2,1),pad_amount=((2,3),(1,2)),first_conv_axis=1)

    errors += op_evaluation_and_ot(op)

    #complex
    vs=UniformGridFcts(10,10,dtype=np.complex128)
    errors += op_basics_wrapper(GaussianBlur,vs,5,(2,1),test_methods=True, pad_amount=((2,3),(1,2)),first_conv_axis=1)

    op=GaussianBlur(vs,5,(2,1),pad_amount=((2,3),(1,2)))

    errors += op_evaluation_and_ot(op)

    collect_errors(GaussianBlur,errors)

def test_ExponentialConvolution():
    errors = []
    vs = UniformGridFcts(10,10)
    errors += op_basics_wrapper(ExponentialConvolution,vs,0.5,test_methods=True, pad_amount=((2,3),(1,2)),first_conv_axis=1)

    op=ExponentialConvolution(vs,0.5,pad_amount=((2,3),(1,2)),first_conv_axis=1)

    errors += op_evaluation_and_ot(op)

    #complex
    vs=UniformGridFcts(10,10,dtype=np.complex128)
    errors += op_basics_wrapper(ExponentialConvolution,vs,0.5,test_methods=True, pad_amount=((2,3),(1,2)),first_conv_axis=1)
    
    op=ExponentialConvolution(vs,0.5,pad_amount=((2,3),(1,2)))

    errors += op_evaluation_and_ot(op)

    collect_errors(ExponentialConvolution,errors)

def test_FresnelPropagator():
    errors = []
    vs=UniformGridFcts(10,10,dtype=np.complex128)
    errors += op_basics_wrapper(FresnelPropagator,vs,2.5,test_methods=True, rel_tol_norm=1e-3, pad_amount=((2,3),(1,2)),first_conv_axis=1)
    
    op=FresnelPropagator(vs,2.5,pad_amount=((2,3),(1,2)))

    errors += op_evaluation_and_ot(op)

    collect_errors(FresnelPropagator,errors)