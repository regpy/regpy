import numpy as np
from numpy import pi 
from numpy.linalg import norm
import pytest

from regpy.vecsps.numpy import *
from regpy.operators.convolution import *
from regpy.util import Errors

from .base_operator import op_basics_wrapper,op_evaluation_and_ot
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)

class TestPaddingOperator():

    @pytest.mark.parametrize("vs, pad_amount",[ 
        (UniformGridFcts(3,3),None),
        (UniformGridFcts(3,3),[2,3]),
        (UniformGridFcts(2,3,dtype=complex),2)
    ])
    def test_op_basic(self,vs,pad_amount):
        op_basics_wrapper(PaddingOperator,vs,test_methods=True, pad_amount = pad_amount)

    @pytest.mark.parametrize("vs, pad_amount, x, res",[ 
        (UniformGridFcts(3,3),None, np.ones((3,3)),np.ones((3,3))),
        (UniformGridFcts(2,2),[1,2], np.ones((2,2)), np.asarray([[0,0,0,0,0,0],[0,0,1.,1.,0,0],[0,0,1.,1.,0,0],[0,0,0,0,0,0]])),
        (UniformGridFcts(2,3,dtype=complex),2,np.ones((2,3))*1j, np.asarray([[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,1j,1j,1j,0,0],[0,0,1j,1j,1j,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]))
    ])
    def test_ot_eval(self,vs,pad_amount,x,res):
        op=PaddingOperator(vs,pad_amount=pad_amount)
        op_evaluation_and_ot(op,x=x,res=res)

@pytest.mark.parametrize("vs, pad_amount, kernel",[ 
        (UniformGridFcts(8,12),2,np.arange(12*9).reshape(12,9)),
        (UniformGridFcts(8,12),2,lambda a,b:a*b*1j),
        (UniformGridFcts(10,10,dtype=np.complex128),2,np.arange(14*14).reshape(14,14)),
        (UniformGridFcts(10,10,dtype=np.complex128),2,lambda a,b:a*np.conj(b)),
    ])
class TestConvolutionOperator():
    def test_op_basic(self,vs,pad_amount,kernel):
        op_basics_wrapper(ConvolutionOperator,vs,test_methods=True,rel_tol_norm = 1e-3,
                          fourier_multiplier=kernel, pad_amount = pad_amount)

    def test_ot_eval(self,vs,pad_amount,kernel):
        op=ConvolutionOperator(vs,fourier_multiplier=kernel,pad_amount=2)
        op_evaluation_and_ot(op)

@pytest.mark.parametrize("vs, sigma, pad_amount, convolution_axes",[ 
        (UniformGridFcts(10,10),5,2,[1]),
        (UniformGridFcts(10,10,dtype=np.complex128),5,2,[1]),
    ])
class TestGaussianBlur():
    def test_op_basic(self,vs,sigma,pad_amount,convolution_axes):
        op_basics_wrapper(GaussianBlur,vs,test_methods=True, sigma = sigma, pad_amount=pad_amount,convolution_axes=convolution_axes)

    def test_ot_eval(self,vs,sigma,pad_amount,convolution_axes):
        op=GaussianBlur(vs,sigma=sigma,pad_amount=pad_amount,convolution_axes=convolution_axes)
        op_evaluation_and_ot(op)

@pytest.mark.parametrize("vs, a, pad_amount, convolution_axes",[ 
        (UniformGridFcts(10,10),0.5,2,[0]),
        (UniformGridFcts(10,10,dtype=np.complex128),0.5,2,None),
    ])
class TestExponentialConvolution():
    def test_op_basic(self,vs,a,pad_amount,convolution_axes):
        op_basics_wrapper(ExponentialConvolution,vs,test_methods=True, a = a, pad_amount=pad_amount,convolution_axes=convolution_axes)

    def test_ot_eval(self,vs,a,pad_amount,convolution_axes):
        op=ExponentialConvolution(vs,a=a,pad_amount=pad_amount,convolution_axes=convolution_axes)
        op_evaluation_and_ot(op)

@pytest.mark.parametrize("vs, fresnel_number, pad_amount, convolution_axes",[ 
        (UniformGridFcts(10,10,dtype=np.complex128),2.5,2,None),
    ])
class TestFresnelPropagator():
    def test_op_basic(self,vs,fresnel_number, pad_amount, convolution_axes):
        op_basics_wrapper(FresnelPropagator,vs,fresnel_number,test_methods=True, rel_tol_norm=1e-3, pad_amount=pad_amount,convolution_axes=convolution_axes)
    
    def test_ot_eval(self,vs,fresnel_number, pad_amount, convolution_axes):
        op=FresnelPropagator(vs,fresnel_number,pad_amount=pad_amount, convolution_axes=convolution_axes)
        op_evaluation_and_ot(op)


class TestDifferentialOperators():
    errors = []

    grid = UniformGridFcts((-pi,pi,10), (-pi,pi,9),dtype=float,shape_codomain=(1,))
    pad_amount = [2,0]

    @pytest.mark.parametrize("op, vs, pad_amount",[ 
        (gradient,grid.vector_valued_space(1),[2,0]),
        (divergence,grid.vector_valued_space(2),[2,0]),
        (Laplacian,grid.scalar_space(),[2,0]),
    ])
    def test_op_basic(self, op, vs, pad_amount):
        op_basics_wrapper(op,vs,pad_amount=pad_amount)

    def test_compatibility(self):
        grad =  gradient(self.grid.vector_valued_space(1),pad_amount=self.pad_amount)
        div = divergence(self.grid.vector_valued_space(2),pad_amount=self.pad_amount)  
        Lap = Laplacian(self.grid,pad_amount=self.pad_amount)
        div_grad = div.composition(grad)
        assert np.allclose(Lap.fourier_multiplier,div_grad.fourier_multiplier), Errors.failed_test(f"Comparing the Fourier multiplier of Laplace to div_grad composition is not close!",meth="Differential Operators")

    @pytest.mark.parametrize("op, vs, pad_amount",[ 
        (gradient,grid.vector_valued_space(1),[2,0]),
        (divergence,grid.vector_valued_space(2),[2,0]),
        (Laplacian,grid.scalar_space(),[2,0]),
    ])
    def test_op_eval(self,op,vs,pad_amount):
        op_evaluation_and_ot(op(vs,pad_amount=pad_amount))

    # test identities curl grad = 0,  div curl = 0, and \Delta = grad div - curl curl
    # different implementations of convolution operators with and without padding 
    @pytest.mark.parametrize("type,pad_amount",[ 
        (float,None),
        (float,[2,0,3]),
        (complex,None),
        (complex,[2,0,3]),
    ])
    def test_identies(self,type,pad_amount):
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

class TestShiftConvolutionCalculus():
    @pytest.mark.parametrize("vs,shift,pad_amount,convolution_axes", [ 
        (UniformGridFcts((-pi,pi,50),dtype=np.complex128),[1.],2,None)
    ])
    def test_op_basic(self,vs,shift,pad_amount,convolution_axes):
        op_basics_wrapper(PeriodicShift,vs,shift,test_methods=True, rel_tol_norm=1e-3, pad_amount=pad_amount,convolution_axes=convolution_axes)
    
    @pytest.mark.parametrize("vs,shift,pad_amount,convolution_axes", [ 
        (UniformGridFcts((-pi,pi,50),dtype=np.complex128),[1.],2,None)
    ])
    def test_ot_eval(self,vs,shift,pad_amount,convolution_axes):
        op=PeriodicShift(vs,shift,pad_amount=pad_amount,convolution_axes=convolution_axes)
        op_evaluation_and_ot(op)

    def test_periodic_shift(self):
        grid =  UniformGridFcts((-pi,pi,300),(-pi,pi,50),periodic=True,dtype=complex)
        X,Y = grid.coords
        f = np.exp(-100*(X**2+Y**2))
        g = np.exp(-100*((X+1)**2+Y**2))
        shift1 = PeriodicShift(grid,[1,0])
        fshift = shift1(f)
        assert norm(fshift-g)<=1e-10

        blur = GaussianBlur(grid,(0.1)**2)
        double_peak = 0.5*shift1.composition(blur) + blur
        assert norm(double_peak(f)-blur(f)-0.5*shift1(blur(f)))<1e-12