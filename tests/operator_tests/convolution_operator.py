import numpy as np

from regpy.vecsps.numpy import *
from regpy.operators.convolution import *
from numpy import pi 
from numpy.linalg import norm

from .base_operator import op_basics_wrapper,op_evaluation_and_ot,collect_errors

def test_PaddingOperator():
    errors = []
    vs = UniformGridFcts(3,3)
    errors += op_basics_wrapper(PaddingOperator,vs,test_methods=True)

    op=PaddingOperator(vs,[2,3])
    x=vs.ones()
    res=np.zeros((7,9))
    res[2:5,3:6]=1

    errors += op_evaluation_and_ot(op,x=x,res=res)

    vs = UniformGridFcts(2,3,dtype=complex)
    op=PaddingOperator(vs,2)
    x=vs.ones()*1j

    res=np.zeros((6,7),dtype=complex)
    res[2:4,2:5]=1j

    errors += op_evaluation_and_ot(op,x=x,res=res)

    collect_errors(PaddingOperator,errors)

def test_ConvolutionOperator():
    errors = []
    vs = UniformGridFcts(8,12)
    kernel = np.arange(12*9).reshape(12,9)
    errors += op_basics_wrapper(ConvolutionOperator,vs,test_methods=True, rel_tol_norm = 1e-3,
                                fourier_multiplier=kernel,pad_amount=2)

    op=ConvolutionOperator(vs,fourier_multiplier=kernel,pad_amount=2)

    errors += op_evaluation_and_ot(op)

    kernel2=lambda a,b:a*b*1j
    op=ConvolutionOperator(vs,fourier_multiplier=kernel2,pad_amount=2)

    errors += op_evaluation_and_ot(op)

    #complex
    vs=UniformGridFcts(10,10,dtype=np.complex128)
    kernel=np.arange(14*14).reshape(14,14)
    op=ConvolutionOperator(vs,fourier_multiplier=kernel,pad_amount=2)

    errors += op_evaluation_and_ot(op)

    kernel2=lambda a,b:a*np.conj(b)
    op=ConvolutionOperator(vs,fourier_multiplier=kernel2,pad_amount=2)

    errors += op_evaluation_and_ot(op)

    collect_errors(ConvolutionOperator,errors)

def test_GaussianBlur():
    errors = []
    vs = UniformGridFcts(10,10)
    errors += op_basics_wrapper(GaussianBlur,vs,5,test_methods=True, pad_amount=2,convolution_axes=[1])

    op=GaussianBlur(vs,5,pad_amount=2,convolution_axes=[1])

    errors += op_evaluation_and_ot(op)

    #complex
    vs=UniformGridFcts(10,10,dtype=np.complex128)
    errors += op_basics_wrapper(GaussianBlur,vs,5,test_methods=True, pad_amount=2,convolution_axes=[1])

    op=GaussianBlur(vs,5,pad_amount=2)

    errors += op_evaluation_and_ot(op)

    collect_errors(GaussianBlur,errors)

def test_ExponentialConvolution():
    errors = []
    vs = UniformGridFcts(10,10)
    errors += op_basics_wrapper(ExponentialConvolution,vs,0.5,test_methods=True, pad_amount=2,convolution_axes=[0])

    op=ExponentialConvolution(vs,0.5,pad_amount=2,convolution_axes=[0])

    errors += op_evaluation_and_ot(op)

    #complex
    vs=UniformGridFcts(10,10,dtype=np.complex128)
    errors += op_basics_wrapper(ExponentialConvolution,vs,0.5,test_methods=True, pad_amount=2,convolution_axes=None)
    
    op=ExponentialConvolution(vs,0.5,pad_amount=2)

    errors += op_evaluation_and_ot(op)

    collect_errors(ExponentialConvolution,errors)

def test_FresnelPropagator():
    errors = []
    vs=UniformGridFcts(10,10,dtype=np.complex128)
    errors += op_basics_wrapper(FresnelPropagator,vs,2.5,test_methods=True, rel_tol_norm=1e-3, pad_amount=2,convolution_axes=None)
    
    op=FresnelPropagator(vs,2.5,pad_amount=2)

    errors += op_evaluation_and_ot(op)

    collect_errors(FresnelPropagator,errors)

def test_differential_operators():
    errors = []

    grid = UniformGridFcts((-pi,pi,10), (-pi,pi,9),dtype=float,shape_codomain=(1,))
    pad_amount = [2,0]

    errors += op_basics_wrapper(gradient,grid.vector_valued_space(1),pad_amount=pad_amount)
    errors += op_basics_wrapper(divergence,grid.vector_valued_space(2),pad_amount=pad_amount)   
    errors += op_basics_wrapper(Laplacian,grid.scalar_space(),pad_amount=pad_amount)   

    grad =  gradient(grid.vector_valued_space(1),pad_amount=pad_amount)
    div = divergence(grid.vector_valued_space(2),pad_amount=pad_amount)  
    Lap = Laplacian(grid,pad_amount=pad_amount)
    div_grad = div.composition(grad)
    assert np.allclose(Lap.fourier_multiplier,div_grad.fourier_multiplier)

    errors += op_evaluation_and_ot(grad)
    collect_errors(gradient,errors)
    errors += op_evaluation_and_ot(div)
    collect_errors(divergence,errors)
    errors += op_evaluation_and_ot(Lap)
    collect_errors(Laplacian,errors)

    # test identities curl grad = 0,  div curl = 0, and \Delta = grad div - curl curl
    for type in [float,complex]:
        for pad_amount in [None,  [2,0,3]]: # different implementations of convolution operators with and without padding 
            grid3D = UniformGridFcts((-pi,pi,20), (-pi,pi,24),(-pi,pi,15),dtype=type,shape_codomain=(1,))

            grad = gradient(grid3D.vector_valued_space(1),pad_amount=pad_amount)   
            curlop = curl(grid3D.vector_valued_space(3),pad_amount=pad_amount)
            div = divergence(grid3D.vector_valued_space(3),pad_amount=pad_amount)
            Lap3D = Laplacian(grid3D.vector_valued_space(3),pad_amount=pad_amount)

            curl_grad =  curlop.composition(grad)
            assert norm(curl_grad.fourier_multiplier)==0
            div_curl =  div.composition(curlop)
            assert norm(div_curl.fourier_multiplier)==0
            test = grad.composition(div) - curlop.composition(curlop) 
            test -= Lap3D
            assert norm(test.fourier_multiplier)<=1e-10

            # make sure this also holds true approximately with periodization errors
            X,Y,Z = grid3D.coords
            f3d = grid3D.zeros()
            f3d[...,0] = np.exp(-400*(X**2+Y**2+Z**2))*np.cos(3*X-Z+2*Z)
            g3d = grid3D.vector_valued_space(3).zeros()
            g3d[...,0] = f3d[...,0]
            g3d[...,1] = np.exp(-400*(X**2+Y**2+Z**2))*np.sin(3*X)
            g3d[...,2] = np.exp(-400*(X**2+Y**2+Z**2))*np.sin(-Z+2*Y)

            assert np.allclose(curlop(grad(f3d)),np.zeros_like(g3d))
            assert np.allclose(grad.adjoint(curlop(g3d)),np.zeros_like(f3d))
            assert np.allclose(grad(div(g3d))-curlop.adjoint(curlop(g3d)) ,  Lap3D(g3d),atol=1e-6)