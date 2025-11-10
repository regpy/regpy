from math import isclose
from copy import deepcopy
import traceback

import numpy as np

from regpy.vecsps import NumPyVectorSpace
import regpy.operators.base as op_base
import regpy.util.operator_tests as ot

def collect_errors(cls,errors):
    if errors:
        sep = "\n" +"-"*125 +"\n"
        title = f"\t\tDuring the testing of {cls} were the following errors collected"
        massage = sep+title+sep+ sep.join(errors)
        raise AssertionError(massage)
    else:
        return None

def call_safe(obj, method_name, error_log, *args, **kwargs):
    """
    Calls a method of an object safely.
    
    Returns a tuple:
        result_or_none
    
    result_or_none: the return value or none
    """
    method = getattr(obj, method_name)
    try:
        result = method(*args, **kwargs)
        return result
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        error_log.append(f"The method {method_name} of {obj} with arguments {args} and keyword arguments {kwargs} could not construct an object resulting in exception {e}.\n Resulting from {tb} \n")
        return None
    

def op_basics(op,*args,test_methods = False, rel_tol_norm = 1e-3, inv_tol = 1e-15,**kwargs):
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

    if op.domain is None:
        errors.append(f"The Operator {op} being initiated with {args} and {kwargs} has no domain specified.")
    if op.codomain is None:
        errors.append(f"The Operator {op} being initiated with {args} and {kwargs} has no codomain specified.")
        
    if test_methods:
        _ = call_safe(op,"_eval",errors,full_dom.rand())
        tup = call_safe(op,"linearize",errors, dom.rand())
        if tup is None:
            errors.append(f"The Operator {op} being initiated with {args} and {kwargs} returned from linearize None.")
        elif len(tup) !=2:
            errors.append(f"The Operator {op} being initiated with {args} and {kwargs} returned from linearize not exactly 2 results.")
        tup = call_safe(op,"linearize",errors, dom.rand(), return_adjoint_eval = True)
        if tup is None:
            errors.append(f"The Operator {op} being initiated with {args} and {kwargs} returned from linearize None.")
        elif len(tup) !=2:
            errors.append(f"The Operator {op} being initiated with {args} and {kwargs} returned from linearize with return_adjoint_eval = True not exactly 2 results.")
        if op.linear:
            _ = call_safe(op,"as_linear_operator",errors)
            if isinstance(op,op_base.Zero):
                pass
            else:
                norm_power = call_safe(op,"norm",errors,method="power")
                norm_lanczos = call_safe(op,"norm",errors,method="lanczos")
                
                if not (isinstance(norm_power,float) and isinstance(norm_lanczos,float)):
                    errors.append(f"The Operator {op} being initiated with {args} and {kwargs} computed the norm with power and lanczos method resulted in objects not of float type by norm_power = {type(norm_power)} and norm_lanczos = {type(norm_lanczos)}.")
                elif not isclose(norm_power,norm_lanczos,rel_tol=rel_tol_norm):
                    errors.append(f"The Operator {op} being initiated with {args} and {kwargs} computed the norm with power and lanczos method resulted in not close values norm_power = {norm_power} and norm_lanczos = {norm_lanczos}.")
    op_alt = deepcopy(op)
    
    try: 
        inv = op.inverse
        x = dom.rand()
        x_alt = inv(op(x))
        diff = dom.norm(x-x_alt)
        if not isclose(diff,0,abs_tol=inv_tol):
            errors.append(f"Testing the inverse of operator {op} on a random vector {x} did not return the almost same vector but {x_alt} with a domain norm difference {diff} ")
    except NotImplementedError:
        pass
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        errors.append(f"The inverse implementation for {op} does not properly work. Throwing and exception {e}.\n Resulting from {tb} \n")
    try:
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
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        errors.append(f"The addition and power implementations for {op} do not properly work. Throwing and exception {e}.\n Resulting from {tb} \n")

    return errors

def op_basics_wrapper(OP,*args,test_methods = False, rel_tol_norm = 1e-3, inv_tol = 1e-15,**kwargs):
    op = OP(*args,**kwargs)
    return op_basics(op,*args,test_methods=test_methods,rel_tol_norm = rel_tol_norm, inv_tol=inv_tol,**kwargs)

def op_evaluation_and_ot(op,x=None,res=None,tol=1e-10,**kwargs):
    errors = []
    if x is None or res is None:
        diff = None
    else:
        diff = op.codomain.norm(op(x)-res)
    if diff and diff>1e-10:
        errors.append(f"Testing the application of {type(op)} at {x} against given result {res} computed result is {op(x)} is not masking properly norm difference {diff}")
    try:
        ot.test_operator(op,**kwargs)
    except AssertionError as e:
        errors.append(f"Running the standard test resulted in an error {e}.")
    return errors

def test_Identity():
    vs = NumPyVectorSpace((4,3))
    errors = []
    errors += op_basics_wrapper(op_base.Identity, vs, test_methods=True)

    op=op_base.Identity(domain=vs)
    x=vs.randn()

    errors += op_evaluation_and_ot(op,x=x,res=x)

    #complex
    dom=NumPyVectorSpace((2,2),np.complex128)
    op=op_base.Identity(domain=dom)
    x=dom.randn()
    
    errors += op_evaluation_and_ot(op,x=x,res=x)
    
    collect_errors(op_base.Identity,errors)

def test_SquaredModulus():
    vs = NumPyVectorSpace((2,2),dtype=complex)
    errors = []
    errors += op_basics_wrapper(op_base.SquaredModulus, vs, test_methods=True)

    op=op_base.SquaredModulus(domain=vs)
    x=vs.ones()*1j
    x[0,0]=2+1j
    res = np.abs(x)**2

    errors += op_evaluation_and_ot(op,x=x,res=res)

    collect_errors(op_base.SquaredModulus,errors)

def test_Pow():
    vs = NumPyVectorSpace((4,3))
    errors = []
    errors += op_basics_wrapper(op_base.Pow, vs.identity, 3, test_methods=True)
    #real
    dom=NumPyVectorSpace((2,2))
    mult_op=op_base.PtwMultiplication(dom,factor=2)
    op=op_base.Pow(mult_op,3)
    x=dom.ones()
    x[0,0]=2
    res = np.array([[16,8],[8,8]])

    errors += op_evaluation_and_ot(op,x=x,res=res)

    #complex
    dom=NumPyVectorSpace((2,2),np.complex128)
    mult_op=op_base.PtwMultiplication(dom,factor=1j)
    op=op_base.Pow(mult_op,3)
    x=dom.ones()
    x[0,0]=2+1j
    res = np.array([[1-2j,-1j],[-1j,-1j]])

    errors += op_evaluation_and_ot(op,x=x,res=res)

    collect_errors(op_base.Pow,errors)
    
def test_PtwMultiplication():
    vs = NumPyVectorSpace((4,3))
    errors = []
    factor = vs.randn()
    errors += op_basics_wrapper(op_base.PtwMultiplication, vs, factor, test_methods=True,rel_tol_norm=1e-3)

    dom = NumPyVectorSpace((3,2))
    factor = dom.rand()  
    op = op_base.PtwMultiplication(dom,factor)
    
    errors += op_evaluation_and_ot(op)
    
    dom = NumPyVectorSpace((3,2),dtype=np.complex128)
    factor = dom.rand()  
    op = op_base.PtwMultiplication(dom,factor)

    errors += op_evaluation_and_ot(op)

    collect_errors(op_base.PtwMultiplication,errors)
    
def test_Composition():
    vs = NumPyVectorSpace((4,3))
    errors = []
    errors += op_basics_wrapper(op_base.Composition, vs.identity, vs.identity*4, test_methods=True)
    collect_errors(op_base.Composition,errors)

def test_LinearCombination():
    vs = NumPyVectorSpace((4,3))
    errors = []
    errors += op_basics_wrapper(op_base.LinearCombination, (3.0,vs.identity), (-3,vs.identity*3), test_methods=True)
    collect_errors(op_base.LinearCombination,errors)
    
def test_OuterShift():
    vs = NumPyVectorSpace((4,3))
    errors = []
    errors += op_basics_wrapper(op_base.OuterShift, vs.identity, vs.randn(), test_methods=True)

    dom=NumPyVectorSpace((2,5),np.complex128)
    offset = dom.real_space().rand()
    op_unshifted = op_base.SquaredModulus(domain=dom)
    op_shifted = op_base.OuterShift(op_unshifted,offset)
    x = dom.zeros()
    res = offset

    errors += op_evaluation_and_ot(op_shifted,x,res)

    collect_errors(op_base.OuterShift,errors)
    
def test_InnerShift():
    vs = NumPyVectorSpace((4,3))
    errors = []
    errors += op_basics_wrapper(op_base.InnerShift, vs.identity, vs.randn(), test_methods=True)

    dom=NumPyVectorSpace((2,5),np.complex128)
    offset = dom.rand()
    op_unshifted = op_base.SquaredModulus(domain=dom)
    op = op_base.InnerShift(op_unshifted,offset)
    x = dom.zeros()
    res = np.abs(offset)**2

    errors += op_evaluation_and_ot(op,x=x,res=res)

    collect_errors(op_base.InnerShift,errors)

def test_CoordinateProjection():
    vs = NumPyVectorSpace((4,3))
    errors = []
    mask = (np.random.rand(12)>0.5).reshape((4,3))
    errors += op_basics_wrapper(op_base.CoordinateProjection, vs, mask, test_methods=True)
    #real
    dom=NumPyVectorSpace((2,2))
    mask=np.array([[1,0],[0,1]],dtype=bool)
    op=op_base.CoordinateProjection(dom,mask)
    x=dom.ones()
    x[0,0]=2
    res = np.array([2,1])

    errors += op_evaluation_and_ot(op,x=x,res=res)
    
    #complex
    dom=NumPyVectorSpace((2,2),np.complex128)
    mask=np.array([[1,0],[0,1]],dtype=bool)
    op=op_base.CoordinateProjection(dom,mask)
    x=1j*dom.ones()
    x[0,0]=2+1j
    res = np.array([2+1j,1j])
    
    errors += op_evaluation_and_ot(op,x=x,res=res)

    collect_errors(op_base.CoordinateProjection,errors)

def test_CoordinateMask():
    vs = NumPyVectorSpace((4,3))
    errors = []
    mask = (np.random.rand(12)>0.5).reshape((4,3))
    errors += op_basics_wrapper(op_base.CoordinateMask, vs, mask, test_methods=True)

    #real
    dom=NumPyVectorSpace((2,2))
    mask=np.array([[1,0],[0,1]],dtype=bool)
    op=op_base.CoordinateMask(dom,mask)
    x=dom.ones()
    x[0,0]=2
    res = np.array([[2,0],[0,1]])

    errors += op_evaluation_and_ot(op,x=x,res=res)
    
    #complex
    dom=NumPyVectorSpace((2,2),np.complex128)
    mask=np.array([[1,0],[0,0]],dtype=bool)
    op=op_base.CoordinateMask(dom,mask)
    x=1j*dom.ones()
    x[0,0]=2+1j
    res = np.array([[2+1j,0],[0,0]])

    errors += op_evaluation_and_ot(op,x=x,res=res)

    collect_errors(op_base.CoordinateMask,errors)

def test_DirectSum():
    vs = NumPyVectorSpace((4,3))
    errors = []
    mask = (np.random.rand(12)>0.5).reshape((4,3))
    errors += op_basics_wrapper(op_base.DirectSum, vs.identity, op_base.CoordinateMask(vs,mask), op_base.CoordinateProjection(vs,mask), test_methods=True)
    errors += op_basics_wrapper(op_base.DirectSum, vs.identity,op_base.DirectSum(op_base.CoordinateMask(vs,mask), op_base.CoordinateProjection(vs,mask)), test_methods=True, flatten = True)

    op_with_const = op_base.DirectSum(vs.identity, op_base.CoordinateMask(vs,mask), op_base.CoordinateProjection(vs,mask))
    op_with_const.set_constant(vs.randn(),0)
    print(op_with_const(op_with_const.domain.rand()))
    print("Constants",op_with_const._constants)
    errors += op_basics(op_with_const,test_methods=True)
    op_with_const.set_constant(vs.randn(),2)
    errors += op_basics(op_with_const,test_methods=True)
    op_with_const.reset_constants()
    errors += op_basics(op_with_const,test_methods=True)

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

    errors += op_evaluation_and_ot(op,x=x,res=res)

    collect_errors(op_base.DirectSum,errors)

def test_PartOfOperator():
    vs = NumPyVectorSpace((4,3))
    errors = []
    mask = (np.random.rand(12)>0.5).reshape((4,3))
    errors += op_basics_wrapper(op_base.PartOfOperator,op_base.DirectSum(vs.identity, op_base.CoordinateMask(vs,mask), op_base.CoordinateProjection(vs,mask)), 2, test_methods=True)
    errors += op_basics_wrapper(op_base.PartOfOperator,op_base.DirectSum(vs.identity, op_base.CoordinateMask(vs,mask), op_base.CoordinateProjection(vs,mask)), (0,1), test_methods=True)
    errors += op_basics_wrapper(op_base.PartOfOperator,op_base.DirectSum(vs.identity, op_base.CoordinateMask(vs,mask), op_base.CoordinateProjection(vs,mask)), slice(0,2,1), test_methods=True)
    collect_errors(op_base.PartOfOperator,errors)

def test_RealPart():
    vs = NumPyVectorSpace((4,3),dtype=complex)
    errors = []
    errors += op_basics_wrapper(op_base.RealPart, vs, test_methods=True)

    op=op_base.RealPart(domain=vs)
    x=vs.randn()
    res = np.real(x)

    errors += op_evaluation_and_ot(op,x,res)

    collect_errors(op_base.RealPart,errors)
    
def test_RealPart():
    vs = NumPyVectorSpace((4,3),dtype=complex)
    errors = []
    errors += op_basics_wrapper(op_base.ImaginaryPart, vs, test_methods=True)

    op=op_base.ImaginaryPart(domain=vs)
    x=vs.randn()
    res = np.imag(x)

    errors += op_evaluation_and_ot(op,x,res)

    collect_errors(op_base.ImaginaryPart,errors)
    
def test_Zero():
    vs = NumPyVectorSpace((4,3),dtype=complex)
    errors = []
    errors += op_basics_wrapper(op_base.Zero, vs, test_methods=True)

    op=op_base.Zero(domain=vs)
    x=vs.randn()
    res = np.real(op.domain.zeros())

    errors += op_evaluation_and_ot(op,x,res)

    collect_errors(op_base.Zero,errors)
    
def test_MatrixOfOperators():
    shape = (2,2)
    dom=NumPyVectorSpace(shape)
    op1=op_base.RealPart(dom)
    op2=op_base.PtwMultiplication(dom,2)
    op3=op_base.PtwMultiplication(dom,3)
    op4=op_base.PtwMultiplication(dom,4)
    ops=[[op1,op2,None],[None,op3,op4]]
    
    errors = []
    errors += op_basics_wrapper(op_base.MatrixOfOperators, ops, test_methods=True)

    op=op_base.MatrixOfOperators(ops)
    x = op.domain.zeros()
    x[0] = np.arange(4).reshape(shape)
    x[1] = np.arange(4,8).reshape(shape)
    res = op.codomain.zeros()
    res[0] = np.arange(4).reshape(shape)
    res[1] = np.array([12,17,22,27]).reshape(shape)
    res[2] = np.array([16,20,24,28]).reshape(shape)

    errors += op_evaluation_and_ot(op,x,res)

    collect_errors(op_base.MatrixOfOperators,errors)

def test_VectorOfOperators():
    shape = (2,2)
    #real
    dom=NumPyVectorSpace(shape)
    op1=op_base.Identity(dom)
    op2=op_base.PtwMultiplication(dom,2)
    op3=op_base.PtwMultiplication(dom,3)
    ops = [op1,op2,op3]

    errors = []
    errors += op_basics_wrapper(op_base.VectorOfOperators, ops, test_methods=True)

    
    op=op_base.VectorOfOperators(ops)
    x = np.arange(4).reshape(shape)
    res = op.codomain.zeros()
    res[0] = np.arange(4).reshape(shape)
    res[1] = np.array([2*i for i in range(4)]).reshape(shape)
    res[2] = np.array([3*i for i in range(4)]).reshape(shape)

    errors += op_evaluation_and_ot(op,x,res)

    #complex
    dom=NumPyVectorSpace(shape,dtype=np.complex128)
    op1=op_base.SquaredModulus(dom)
    op2=op_base.PtwMultiplication(dom,2j)
    op3=op_base.PtwMultiplication(dom,3)
    ops = [op1,op2,op3]

    errors += op_basics_wrapper(op_base.VectorOfOperators, ops, test_methods=True)
    
    op=op_base.VectorOfOperators(ops)
    print(type(op))
    x = np.arange(4).reshape(shape) + 1j*np.arange(4).reshape(shape)
    res = op.codomain.zeros()
    res[0] = 2*np.arange(4).reshape(shape)**2
    res[1] = np.array([(-2+2j)*i for i in range(4)]).reshape(shape)
    res[2] = np.array([(3+3j)*i for i in range(4)]).reshape(shape)

    errors += op_evaluation_and_ot(op,x,res)

    collect_errors(op_base.VectorOfOperators,errors)
