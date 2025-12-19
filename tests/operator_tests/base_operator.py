from math import  sqrt
from copy import deepcopy

import pytest
import numpy as np
from scipy.sparse.linalg import LinearOperator

from regpy.vecsps import NumPyVectorSpace, TupleVector
import regpy.operators.base as op_base
from regpy.operators.graph_operator import OperatorGraph
import regpy.util.operator_tests as ot
from regpy.util import Errors, set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)

def op_basics(op,*args,test_methods = False, test_norm = True, rel_tol_norm = 1e-3, inv_tol = 1e-15,**kwargs):
    """Initializes an object of `vs` with `kwargs` and tests it basic functionality. If `test_methods` is true it test the standard methods that should be available. 

    Parameters
    ----------
    vs : object
        The object to initialize
    test_methods : bool, optional
        Flag if to test methods, by default False
    kwargs : dict
        The keywords to pass to the initialization of the `vs` instance.
        
    Raises
    ------
    AssertionError
        Should any of the tests fail.
    """
    errors = []
    if hasattr(op,"full_domain"):
        full_dom = op.full_domain
        dom = op.domain
    else:
        full_dom = op.domain
        dom = op.domain

    assert op.domain is not None, Errors.failed_test(f"The Operator {op} being initiated with {args} and {kwargs} has no domain specified.",obj = op)
    assert op.codomain is not None, Errors.failed_test(f"The Operator {op} being initiated with {args} and {kwargs} has no codomain specified.", obj=op)
        
    if test_methods:
        assert op(full_dom.rand()) in op.codomain, Errors.failed_test(f"The Operator {op} being initiated with {args} and {kwargs} returned from evaluation something not in the codomain",obj=op,meth="_eval")

        tup = op.linearize(dom.rand())
        assert tup is not None and len(tup) == 2 and tup[0] in op.codomain and ((op.linear and tup[1] is op) or (not op.linear and isinstance(tup[1],op_base.Derivative))), Errors.failed_test(f"The Operator {op} returned from linearize not exactly 2 results of type [array in codomain, Derivative]",obj=op,meth="linearize")

        tup = op.linearize(dom.rand(), return_adjoint_eval = True)
        assert tup is not None and len(tup) == 2 and tup[0] in dom and ((op.linear and (tup[1] is op or isinstance(tup[1],op_base.AdjointEval))) or (not op.linear and isinstance(tup[1],op_base.Derivative))), Errors.failed_test(f"The Operator {op} returned from linearize with return_adjoint_eval = True not exactly 2 results of type [array in full_domain, self or AdjointEval or Derivative]. It returned {tup}",obj=op,meth="linearize")
        if op.linear:
            assert isinstance(op.as_linear_operator(),LinearOperator)
            if isinstance(op,op_base.Zero) or not test_norm:
                pass
            else:
                norm_power = op.norm(method="power")
                assert isinstance(norm_power,float), Errors.failed_test(f"The norm of Operator {op} being initiated with {args} and {kwargs} computed by power method is not of float type of norm_power = {type(norm_power)}",obj=op,meth="norm")
                norm_lanczos = op.norm(method="lanczos")
                assert isinstance(norm_lanczos,float), Errors.failed_test(f"The norm of Operator {op} being initiated with {args} and {kwargs} computed by lanczos is not of float type of norm_lanczos = {type(norm_lanczos)}",obj=op,meth="norm")
                #assert norm_power == pytest.approx(norm_lanczos,rel=rel_tol_norm), Errors.failed_test(f"The Operator {op} being initiated with {args} and {kwargs} computed the norm with power and lanczos method resulted in not close values norm_power = {norm_power} and norm_lanczos = {norm_lanczos}.",obj=op,meth="norm")

    op_alt = deepcopy(op)

    try: 
        inv = op.inverse
        x = dom.rand()
        x_alt = inv(op(x))
        diff = dom.norm(x-x_alt)
        assert diff == pytest.approx(0., abs = inv_tol), Errors.failed_test(f"The inverse of Operator {op} does not return the identity wrt to the norm! diff = {diff} and x = {x} and x_inv = {x_alt}",obj=op,meth="inverse")
    except NotImplementedError:
        pass

    _ = op + op_alt
    op_alt += op
    _ = op + 0
    _ = op + 1.0
    _ = op + op.codomain.rand()
    _ = op - op_alt
    op_alt -= op
    _ = op - 0
    _ = op - 1.0
    _ = op - op.codomain.rand()
    id = op.domain.identity
    _ = op * id
    _ = op * 4
    _ = op * op.domain.rand()
    _ = 6 * op 
    _ = op.codomain.rand() * op
    if dom == op.codomain and op.linear:
        _ = op**4

def op_basics_wrapper(OP,*args,test_methods = False, rel_tol_norm = 1e-3, inv_tol = 1e-15,**kwargs):
    op_basics(OP(*args,**kwargs),*args,test_methods=test_methods,rel_tol_norm = rel_tol_norm, inv_tol=inv_tol,**kwargs)

def op_evaluation_and_ot(op,x=None,res=None,rel_tol=1e-6, tol=1e-10,**kwargs):
    print(kwargs)
    ot.test_operator(op,**kwargs)
    if x is not None or res is not None:
        diff = op.codomain.norm(op(x)-res)
        assert diff == pytest.approx(0.,rel=rel_tol,abs = tol), f"Testing the application of {type(op)} at {x} against given result {res} is different from computed result {op(x)} norm difference {diff}"

@pytest.mark.parametrize("vs",[NumPyVectorSpace((4,3)),NumPyVectorSpace((2,2),np.complex128)])
class TestIdentity():

    def test_op_basic(self,vs):
        op_basics_wrapper(op_base.Identity, vs, test_methods=True)

    def test_ot_eval(self,vs):
        op=op_base.Identity(domain=vs)
        x=vs.randn()

        op_evaluation_and_ot(op,x=x,res=x)


class TestSquaredModulus():
    vs = NumPyVectorSpace((2,2),dtype=complex)

    def test_op_basic(self):
        op_basics_wrapper(op_base.SquaredModulus, self.vs, test_methods=True)

    def test_ot_eval(self):
        op = op_base.SquaredModulus(domain=self.vs)
        x=self.vs.ones()*1j
        x[0,0]=2+1j
        res = np.abs(x)**2

        op_evaluation_and_ot(op,x=x,res=res)

class TestPow():
    @pytest.mark.parametrize("vs",[
        NumPyVectorSpace((4,3)),NumPyVectorSpace((4,3),np.complex128)
    ])
    def test_op_basic(self,vs):
        op_basics_wrapper(op_base.Pow, vs.identity, 3, test_methods=True)
    
    @pytest.mark.parametrize("dom, factor, x, res",[
        (NumPyVectorSpace((2,2)), 2, np.array([[2,1],[1,1]]), np.array([[16,8],[8,8]])),
        (NumPyVectorSpace((2,2),np.complex128), 1j, np.array([[2+1j,1.],[1.,1.]]) ,np.array([[1-2j,-1j],[-1j,-1j]]))
    ])
    def test_ot_eval(self, dom, factor, x, res):
        mult_op=op_base.PtwMultiplication(dom,factor=factor)
        op=op_base.Pow(mult_op,3)
        op_evaluation_and_ot(op,x=x,res=res)
    
class TestPtwMultiplication():
    @pytest.mark.parametrize("vs",[
        NumPyVectorSpace((4,3)),
        NumPyVectorSpace((4,3),np.complex128)
    ])
    def test_op_basic(self,vs):
        op_basics_wrapper(op_base.PtwMultiplication, vs, vs.randn(), test_methods=True,rel_tol_norm=1e-3)
    
    @pytest.mark.parametrize("dom, factor, x, res",[
        (NumPyVectorSpace((2,3)), np.array([[2,2,1],[4,1,1]]), np.array([[3,1,-1],[1,-4,1]]), np.array([[6,2,-1],[4,-4,1]])),
        (NumPyVectorSpace((2,3),np.complex128), np.array([[1j,1.,2j],[-1j,1.,-3.]]), np.array([[1j,4j,1.],[3.,1j,1.]]) ,np.array([[-1,4j,2j],[-3j,1j,-3]]))
    ])
    def test_ot_eval(self, dom, factor, x, res):
        op = op_base.PtwMultiplication(dom,factor)
        op_evaluation_and_ot(op,x,res)
 
class TestComposition():
    def test_basics(self):
        vs = NumPyVectorSpace((4,3))
        op_basics_wrapper(op_base.Composition, vs.identity, vs.identity*4, test_methods=True)

    def test_evaluation(self):
        vs_c = NumPyVectorSpace((2,2),dtype=complex)
        vs_r = vs_c.real_space()
        factor = np.array([[2,2],[1,2]])
        x = np.array([[-1+1j,2],[1-1j,3+4j]])
        res = np.array([[4,8],[2    ,50]])
        op = op_base.PtwMultiplication(vs_r,factor=factor) * op_base.SquaredModulus(vs_c)
        op_evaluation_and_ot(op,x=x,res=res)

def test_LinearCombination():
    vs = NumPyVectorSpace((4,3))
    op_basics_wrapper(op_base.LinearCombination, (3.0,vs.identity), (-3,vs.identity*3), test_methods=True)
    
class TestOuterShift():
    @pytest.mark.parametrize("vs",[
        NumPyVectorSpace((4,3)),
        NumPyVectorSpace((4,3),np.complex128)
    ])
    def test_op_basic(self,vs):
        op_basics_wrapper(op_base.OuterShift, vs.identity, vs.randn(), test_methods=True)

    @pytest.mark.parametrize("dom, shift, x, res",[
        (NumPyVectorSpace((2,3)), np.array([[2,2,1],[4,1,1]]), np.array([[3,1,-1],[1,-4,1]]), np.array([[11,3,2],[5,17,2]])),
        (NumPyVectorSpace((2,3),np.complex128), np.array([[1.,1.,2.],[-1.,1.,-3.]]), np.array([[1j,4j,1.],[3.,1j,1.]]) ,np.array([[2,17,3],[8,2,-2]]))
    ])
    def test_ot_eval(self, dom, shift, x, res):
        op_unshifted = op_base.SquaredModulus(domain=dom)
        op_shifted = op_base.OuterShift(op_unshifted,shift)
        op_evaluation_and_ot(op_shifted,x,res)

class TestInnerShift():
    @pytest.mark.parametrize("vs",[
        NumPyVectorSpace((4,3)),
        NumPyVectorSpace((4,3),np.complex128)
    ])
    def test_op_basic(self,vs):
        op_basics_wrapper(op_base.InnerShift, vs.identity, vs.randn(), test_methods=True)

    @pytest.mark.parametrize("dom, shift, x, res",[
        (NumPyVectorSpace((2,3)), np.array([[2,2,1],[4,1,1]]), np.array([[3,1,-1],[1,-4,1]]), np.array([[1.,1.,4.],[9.,25.,0.]])),
        (NumPyVectorSpace((2,3),np.complex128), np.array([[1.,1j,2.],[-1j,1.,-3j]]), np.array([[1j,4j,1.],[3.,1j,1.]]) ,np.array([[2,9,1],[10.,2.,10.]]))
    ])
    def test_ot_eval(self, dom, shift, x, res):
        op_unshifted = op_base.SquaredModulus(domain=dom)
        op_shifted = op_base.InnerShift(op_unshifted,shift)
        op_evaluation_and_ot(op_shifted,x,res)

class TestCoordinateProjection():
    @pytest.mark.parametrize("vs, mask",[
        (NumPyVectorSpace((4,3)),(np.random.rand(12)>0.5).reshape((4,3))),
        (NumPyVectorSpace((4,3),np.complex128), (np.random.rand(12)>0.5).reshape((4,3)))
    ])
    def test_op_basic(self,vs,mask):
        op_basics_wrapper(op_base.CoordinateProjection, vs, mask, test_methods=True)

    @pytest.mark.parametrize("dom, mask, x, res",[
        (NumPyVectorSpace((2,2)), np.array([[1,0],[0,1]],dtype=bool), np.array([[2.,1.],[1.,1.]]), np.array([2,1])),
        (NumPyVectorSpace((2,2),np.complex128), np.array([[1,0],[0,1]],dtype=bool), np.array([[2+1j,1j],[1j,1j]]) ,np.array([2+1j,1j]))
    ])
    def test_ot_eval(self, dom, mask, x, res):
        op=op_base.CoordinateProjection(dom,mask)
        op_evaluation_and_ot(op,x=x,res=res)
    
class TestCoordinateMask():
    @pytest.mark.parametrize("vs, mask",[
        (NumPyVectorSpace((4,3)),(np.random.rand(12)>0.5).reshape((4,3))),
        (NumPyVectorSpace((4,3),np.complex128), (np.random.rand(12)>0.5).reshape((4,3)))
    ])
    def test_op_basic(self,vs,mask):
        op_basics_wrapper(op_base.CoordinateMask, vs, mask, test_methods=True)

    @pytest.mark.parametrize("dom, mask, x, res",[
        (NumPyVectorSpace((2,2)), np.array([[1,0],[0,1]],dtype=bool), np.array([[2.,1.],[1.,1.]]), np.array([[2,0],[0,1]])),
        (NumPyVectorSpace((2,2),np.complex128), np.array([[1,0],[0,1]],dtype=bool), np.array([[2+1j,1j],[1j,1j]]) ,np.array([[2+1j,0],[0,1j]]))
    ])
    def test_ot_eval(self, dom, mask, x, res):
        op=op_base.CoordinateMask(dom,mask)
        op_evaluation_and_ot(op,x=x,res=res)

class TestDirectSum():
    vs = NumPyVectorSpace((4,3))
    mask = (np.random.rand(12)>0.5).reshape((4,3))

    def test_op_basic(self):
        op_basics_wrapper(op_base.DirectSum, self.vs.identity, op_base.CoordinateMask(self.vs,self.mask), op_base.CoordinateProjection(self.vs,self.mask), test_methods=True)
        op_basics_wrapper(op_base.DirectSum, self.vs.identity,op_base.DirectSum(op_base.CoordinateMask(self.vs,self.mask), op_base.CoordinateProjection(self.vs,self.mask)), test_methods=True, flatten = True)

    def test_set_del_constants(self):
        op_with_const = op_base.DirectSum(self.vs.identity, op_base.CoordinateMask(self.vs,self.mask), op_base.CoordinateProjection(self.vs,self.mask))
        #add a constant
        op_with_const.set_constant(self.vs.randn(),0)
        op_basics(op_with_const,test_methods=True)
        # add additional constant
        op_with_const.set_constant(self.vs.randn(),2)
        op_basics(op_with_const,test_methods=True)
        # delete constants
        op_with_const.reset_constants()
        op_basics(op_with_const,test_methods=True)

    def test_ot_eval(self):
        shape1 = (2,2)
        shape2 = (2,3,4)
        dom1=NumPyVectorSpace(shape1,dtype=complex)
        dom2=NumPyVectorSpace(shape2)
        op1=op_base.SquaredModulus(dom1)
        op2=op_base.PtwMultiplication(dom2,3)
        op=op_base.DirectSum(op1,op2)
        x = op.domain.zeros()
        x[0] = np.arange(4).reshape(shape1) + 1j*np.arange(4).reshape(shape1)
        x[1] = np.arange(4,28).reshape(shape2)
        res = op.codomain.zeros()
        res[0] = 2*np.arange(4).reshape(shape1)**2
        res[1] = np.array([3*i for i in range(4,28)]).reshape(shape2)

        op_evaluation_and_ot(op,x=x,res=res)

class TestPartOfOperator():
    @pytest.mark.parametrize("vs, index, mask,",
            [(NumPyVectorSpace((4,3)),             2,(np.random.rand(12)>0.5).reshape((4,3))),
             (NumPyVectorSpace((4,3)),         (0,2),(np.random.rand(12)>0.5).reshape((4,3))),
             (NumPyVectorSpace((4,3)),  slice(0,2,1),(np.random.rand(12)>0.5).reshape((4,3))),
             (NumPyVectorSpace((4,3),dtype=complex),             2,(np.random.rand(12)>0.5).reshape((4,3))),
             (NumPyVectorSpace((4,3),dtype=complex),         (0,2),(np.random.rand(12)>0.5).reshape((4,3))),
             (NumPyVectorSpace((4,3),dtype=complex),  slice(0,2,1),(np.random.rand(12)>0.5).reshape((4,3)))
             ])    
    def test_op_basic(self,vs ,index,mask):
        op_basics_wrapper(op_base.PartOfOperator,op_base.DirectSum(vs.identity, op_base.CoordinateMask(vs,mask), op_base.CoordinateProjection(vs,mask)), index, test_methods=True)

    @pytest.mark.parametrize("index, mask, x, res",
            [(2,            np.asarray([[1,0],[0,1]],dtype=bool),TupleVector([np.ones((2,2)),np.ones((2,2)),3*np.ones((2,2))]), np.asarray([3,3])),
             ((0,2),        np.asarray([[1,0],[0,1]],dtype=bool),TupleVector([np.ones((2,2)),5*np.ones((2,2)),np.ones((2,2))]), TupleVector([np.ones((2,2)),np.asarray([1,1])])),
             (slice(0,2,1), np.asarray([[1,0],[0,1]],dtype=bool),TupleVector([np.ones((2,2)),6*np.ones((2,2)),np.ones((2,2))]), TupleVector([np.ones((2,2)),np.asarray([[6,0],[0,6]])]))])
    def test_ot_eval(self,index,mask,x,res):
        vs = NumPyVectorSpace((2,2))
        op = op_base.PartOfOperator(op_base.DirectSum(vs.identity, op_base.CoordinateMask(vs,mask), op_base.CoordinateProjection(vs,mask)), index)
        op_evaluation_and_ot(op,x,res)


class TestRealPart():
    vs = NumPyVectorSpace((4,3),dtype=complex)
    
    def test_op_basic(self):
        op_basics_wrapper(op_base.RealPart, self.vs, test_methods=True)

    def test_ot_eval(self):
        op=op_base.RealPart(domain=self.vs)
        x=self.vs.randn()
        res = np.real(x)
        op_evaluation_and_ot(op,x,res)
    
class TestImagPart():
    vs = NumPyVectorSpace((4,3),dtype=complex)
    
    def test_op_basic(self):
        op_basics_wrapper(op_base.ImaginaryPart, self.vs, test_methods=True)
    
    def test_ot_eval(self):
        op=op_base.ImaginaryPart(domain=self.vs)
        x=self.vs.randn()
        res = np.imag(x)
        op_evaluation_and_ot(op,x,res)
    
class TestSplitRealImagPart():
    vs = NumPyVectorSpace((4,3),dtype=complex)

    def test_op_basic(self):
        op_basics_wrapper(op_base.SplitRealImag, self.vs, test_methods=True)

    def test_ot_eval(self):
        op=op_base.SplitRealImag(domain=self.vs)
        x=self.vs.randn()
        res = op.codomain.join(np.real(x),np.imag(x))
        op_evaluation_and_ot(op,x,res)

@pytest.mark.parametrize("vs",[
    NumPyVectorSpace((4,3),dtype=complex),
    NumPyVectorSpace((4,3))
    ])    
class TestZero():
    def test_op_basic(self,vs):
        op_basics_wrapper(op_base.Zero, vs, test_methods=True)

    def test_ot_eval(self,vs):
        op=op_base.Zero(domain=vs)
        x=vs.randn()
        res = np.real(op.domain.zeros())
        op_evaluation_and_ot(op,x,res)
    
def test_MatrixOfOperators():
    shape = (2,2)
    dom=NumPyVectorSpace(shape)
    op1=op_base.RealPart(dom)
    op2=op_base.PtwMultiplication(dom,2)
    op3=op_base.PtwMultiplication(dom,3)
    op4=op_base.PtwMultiplication(dom,4)
    ops=[[op1,op2,None],[None,op3,op4]]
    
    op_basics_wrapper(op_base.MatrixOfOperators, ops, test_methods=True)

    op=op_base.MatrixOfOperators(ops)
    x = op.domain.zeros()
    x[0] = np.arange(4).reshape(shape)
    x[1] = np.arange(4,8).reshape(shape)
    res = op.codomain.zeros()
    res[0] = np.arange(4).reshape(shape)
    res[1] = np.array([12,17,22,27]).reshape(shape)
    res[2] = np.array([16,20,24,28]).reshape(shape)

    op_evaluation_and_ot(op,x,res)

@pytest.mark.parametrize("dtype, first_op, factor_1,factor_2, x, res",[
    (float, op_base.Identity, 2, 3, np.arange(4).reshape((2,2)), 
     TupleVector([np.arange(4).reshape((2,2)),np.array([2*i for i in range(4)]).reshape((2,2)),np.array([3*i for i in range(4)]).reshape((2,2))])),
    (complex, op_base.SquaredModulus, 2j, 3, np.arange(4).reshape((2,2)) + 1j*np.arange(4).reshape((2,2)), 
     TupleVector([2*np.arange(4).reshape((2,2))**2,np.array([(-2+2j)*i for i in range(4)]).reshape((2,2)),np.array([(3+3j)*i for i in range(4)]).reshape((2,2))])),
])
class TestVectorOfOperators():

    def test_op_basic(self,dtype,first_op,factor_1,factor_2, x, res  ):
        dom=NumPyVectorSpace((2,2), dtype= dtype)
        op1=first_op(dom)
        op2=op_base.PtwMultiplication(dom,factor_1)
        op3=op_base.PtwMultiplication(dom,factor_2)
        ops = [op1,op2,op3]
        op_basics_wrapper(op_base.VectorOfOperators, ops, test_methods=True)

    def test_ot_eval(self, first_op, dtype, factor_1,factor_2, x, res):
        dom=NumPyVectorSpace((2,2), dtype= dtype)
        op1=first_op(dom)
        op2=op_base.PtwMultiplication(dom,factor_1)
        op3=op_base.PtwMultiplication(dom,factor_2)
        ops = [op1,op2,op3]    
        op=op_base.VectorOfOperators(ops)
        
        op_evaluation_and_ot(op,x,res)

@pytest.mark.parametrize("vs, x, res",[
    (NumPyVectorSpace((4,3),dtype=np.complex128)+NumPyVectorSpace((4,3),dtype=np.complex128)+NumPyVectorSpace((4,3),dtype=np.complex128),
     TupleVector([(i+1)*1j**i*np.ones((4,3)) for i in range(3)]),
     -2*np.ones((4,3))+2j*np.ones((4,3))),
    (NumPyVectorSpace((4,3))+NumPyVectorSpace((4,3))+NumPyVectorSpace((4,3)), 
     TupleVector([(i+1)*np.ones((4,3)) for i in range(3)]),
     6*np.ones((4,3)))
])
class TestSum():
    def test_op_basic(self, vs, x, res):
        op_basics_wrapper(op_base.Sum,vs,test_methods=True)
    
    def test_ot_eval(self,vs,x,res):
        op=op_base.Sum(vs)
        op_evaluation_and_ot(op,x=x,res=res)

@pytest.mark.parametrize("vs, x, res",[
    (NumPyVectorSpace((4,3),dtype=np.complex128)+NumPyVectorSpace((4,3),dtype=np.complex128)+NumPyVectorSpace((4,3),dtype=np.complex128),
     TupleVector([(i+1)*1j**i*np.ones((4,3)) for i in range(3)]),
     -6j*np.ones((4,3))),
    (NumPyVectorSpace((4,3))+NumPyVectorSpace((4,3))+NumPyVectorSpace((4,3)), 
     TupleVector([(i+1)*np.ones((4,3)) for i in range(3)]),
     6*np.ones((4,3)))
])
class TestProduct():
    def test_op_basic(self, vs, x, res):
        op_basics_wrapper(op_base.Product,vs,test_methods=True)
    
    def test_ot_eval(self,vs,x,res):
        op=op_base.Product(vs)
        op_evaluation_and_ot(op,x=x,res=res)

class TestOperatorGraph():
    u=NumPyVectorSpace((2,3))
    v=NumPyVectorSpace((2,3))
    w=NumPyVectorSpace((2,3))
    A=op_base.Product(u+v+w)
    B=op_base.Product(v+w)
    C=op_base.SquaredModulus(u)
    D=op_base.Product(u+v)
    edges=[
        ((None,[0]),(C,0)), # IN-->C
        ((None,[1]),(B,0)), # IN-->B
        ((None,[0]),(D,0)), # IN-->D
        ((None,[2]),(D,1)), # IN-->D     
        ((D,[0]),(B,1)),    # D-->B
        ((C,[0]),(A,0)),    # C-->A
        ((B,[0]),(A,1)),    # B-->A
        ((D,[0]),(A,2)),    # D-->A
        ((A,[0]),(None,0))  # A-->Out
    ]

    operators=[A,B,C,D]

    def test_op_basic(self):
        op_basics_wrapper(OperatorGraph,TestOperatorGraph.operators,TestOperatorGraph.edges,test_methods=True)

    def test_ot_eval(self):
        op=OperatorGraph(TestOperatorGraph.operators,TestOperatorGraph.edges)
        x=op.domain.ones()
        x[0]*=2
        op_evaluation_and_ot(op,x=x,res=16*op.codomain.ones())
