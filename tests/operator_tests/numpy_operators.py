import numpy as np
from scipy.sparse import csc_array
import pytest

from regpy.vecsps import TupleVector
from regpy.vecsps.numpy import *
from regpy.operators.numpy import *

from .base_operator import op_basics_wrapper,op_evaluation_and_ot
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)

class TestMatrixMultiplication():
    @pytest.mark.parametrize("matrix",[ 
        np.random.rand(3,5),
        np.random.rand(20,21),
        np.random.rand(20,21)+1j*np.random.rand(20,21)
    ])
    def test_on_random_matrix(self,matrix):
        op_basics_wrapper(MatrixMultiplication, matrix, test_methods=True)
        op = MatrixMultiplication(matrix)
        op_evaluation_and_ot(op)

    def test_eval(self):
        matrix = np.arange(4*7).reshape(4,7)
        vec = np.sum(matrix,axis=1)
        op = MatrixMultiplication(matrix=matrix)
        assert vec == pytest.approx(op(op.domain.ones()))
    
class TestCholeskyInverse():
    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),domain= UniformGridFcts(4),codomain=UniformGridFcts(4))
    
    def test_op_basic(self):
        op_basics_wrapper(CholeskyInverse, self.op_mat, test_methods=True, inv_tol=1e-14)
    
    def test_ot_eval(self):
        op = CholeskyInverse(self.op_mat)
        op_evaluation_and_ot(op)
    
class TestSuperLUInverse():
    mat =  csc_array([[1,2,0,4], [1,0,0,1], [1,0,2,1], [2,2,1,0.]])
    op_mat = MatrixMultiplication(mat,domain= UniformGridFcts(4),codomain=UniformGridFcts(4))
    
    def test_op_basic(self):
        op_basics_wrapper(SuperLUInverse, self.op_mat, test_methods=True, inv_tol=1e-14)

    def test_ot_eval(self):
        op = SuperLUInverse(self.op_mat)
        op_evaluation_and_ot(op)
    
class TestPower():
    vs = NumPyVectorSpace((2,4),dtype=complex)
    
    @pytest.mark.parametrize("power, kwargs", [(3,{"integer":True}),(1.0,{"integer":True}),(1.5,{}),(-1.5,{})])
    def test_op_basic(self,power,kwargs):
        op_basics_wrapper(Power,power,self.vs,test_methods=True,**kwargs)

    @pytest.mark.parametrize("power, kwargs, x, res", [
        (1.5,{},(np.arange(8).reshape(2,4) + 1j*np.arange(8).reshape(2,4))**2,np.array([(-1+1j)*(2*i)*i**2 for i in range(8)]).reshape(2,4)),
        (3.0,{"integer":True},np.arange(8).reshape(2,4) +1j*np.arange(8).reshape(2,4),np.array([(-1+1j)*(2*i)*i**2 for i in range(8)]).reshape(2,4))])
    def test_ot_eval(self,power,kwargs,x,res):
        op = Power(power,self.vs,**kwargs)
        op_evaluation_and_ot(op,x=x,res=res)

@pytest.mark.parametrize("vs",[ 
        NumPyVectorSpace((2,4),dtype=complex),
        NumPyVectorSpace((3,5))
    ])
class TestExponential():

    def test_op_basic(self,vs):
        op_basics_wrapper(Exponential,vs,test_methods=True)

    def test_ot_eval(self,vs):
        op=Exponential(domain=vs)
        x=vs.randn()
        res = np.exp(x)

        op_evaluation_and_ot(op,x=x,res=res)

@pytest.mark.parametrize("vs",[ 
        (NumPyVectorSpace((2,4),dtype=complex),NumPyVectorSpace(3,dtype=complex),NumPyVectorSpace((5,2),dtype=complex)),
        (NumPyVectorSpace((2,4)),NumPyVectorSpace(3),NumPyVectorSpace((5,2)))
    ])
class TestOuterProduct():

    def test_op_basic(self,vs):
        op_basics_wrapper(OuterProduct,*vs,test_methods=True)

    def test_ot_eval(self,vs):
        op=OuterProduct(*vs)
        x=op.domain.randn()
        res = op.codomain.product(*x)
        op_evaluation_and_ot(op,x=x,res=res)
    
@pytest.mark.parametrize("s,spaces",[
    ("ijki,ji,il->ik",(UniformGridFcts(2,3,3,2,dtype=np.complex128),UniformGridFcts(3,2)) ), #non-linear
    ("ijki,il->ji",(UniformGridFcts(2,3,3,2,dtype=np.complex128),)) #linear
 ])
class TestEinSum():
    errors = []
    tensors=(np.arange(12).reshape(2,6),)

    def test_op_basic(self,s,spaces):
        op_basics_wrapper(EinSum,s,*spaces,test_methods=True,tensors=self.tensors)

    def test_ot_eval(self,s, spaces):
        op=EinSum(s,*spaces,tensors=self.tensors)
        x=op.domain.randn()
        if isinstance(x,TupleVector):
            res = np.einsum(s,*x,*self.tensors)
        else:
            res = np.einsum(s,x,*self.tensors)
        op_evaluation_and_ot(op,x=x,res=res)
