import numpy as np
from scipy.sparse import csc_array

from regpy.vecsps.numpy import *
from regpy.operators.numpy import *

from .base_operator import op_basics_wrapper,op_evaluation_and_ot,collect_errors



def test_MatrixMultiplication():
    errors = []
    errors += op_basics_wrapper(MatrixMultiplication, np.random.rand(3,5), test_methods=True,domain= UniformGridFcts(5),codomain=UniformGridFcts(3))
    
    op = MatrixMultiplication(np.random.rand(20,21),domain= UniformGridFcts(21),codomain=UniformGridFcts(20))

    errors += op_evaluation_and_ot(op)

    op = MatrixMultiplication(np.random.rand(20,21)+1j*np.random.rand(20,21))

    errors += op_evaluation_and_ot(op)

    collect_errors(MatrixMultiplication,errors)
    
def test_CholeskyInverse():
    errors = []
    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),domain= UniformGridFcts(4),codomain=UniformGridFcts(4))
    errors += op_basics_wrapper(CholeskyInverse, op_mat, test_methods=True, inv_tol=1e-14)
    
    op = CholeskyInverse(op_mat)

    errors += op_evaluation_and_ot(op)

    collect_errors(CholeskyInverse,errors)
    
def test_SuperLUInverse():
    errors = []
    mat =  csc_array([[1,2,0,4], [1,0,0,1], [1,0,2,1], [2,2,1,0.]])
    op_mat = MatrixMultiplication(mat,domain= UniformGridFcts(4),codomain=UniformGridFcts(4))
    errors += op_basics_wrapper(SuperLUInverse, op_mat, test_methods=True, inv_tol=1e-14)

    op = SuperLUInverse(op_mat)

    errors += op_evaluation_and_ot(op)

    collect_errors(CholeskyInverse,errors)
    
def test_Power():
    errors = []
    vs = NumPyVectorSpace((2,4),dtype=complex)
    errors += op_basics_wrapper(Power,3,vs,test_methods=True,integer=True)
    errors += op_basics_wrapper(Power,1.0,vs,test_methods=True,integer=True)
    errors += op_basics_wrapper(Power,1.5,vs,test_methods=True)
    errors += op_basics_wrapper(Power,-1.5,vs,test_methods=True)

    op = Power(1.5,vs)
    x = (np.arange(8).reshape(2,4) + 1j*np.arange(8).reshape(2,4))**2
    res = np.array([(-1+1j)*(2*i)*i**2 for i in range(8)]).reshape(2,4)
    
    errors += op_evaluation_and_ot(op,x=x,res=res)

    op = Power(3.0,vs,integer=True)
    x = np.arange(8).reshape(2,4) +1j*np.arange(8).reshape(2,4)
    res = np.array([(-1+1j)*(2*i)*i**2 for i in range(8)]).reshape(2,4)
    
    errors += op_evaluation_and_ot(op,x=x,res=res)

    collect_errors(Power,errors)
    
def test_Exponential():
    errors = []
    vs = NumPyVectorSpace((2,4),dtype=complex)
    errors += op_basics_wrapper(Exponential,vs,test_methods=True)

    op=Exponential(domain=vs)
    x=vs.randn()
    res = np.exp(x)

    errors += op_evaluation_and_ot(op,x=x,res=res)

    collect_errors(Exponential,errors)