import numpy as np
import pytest

from regpy.vecsps.numpy import *
from regpy.operators.bases_transform import *

from .base_operator import op_basics,op_evaluation_and_ot
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)

class TestChebyshevBasis():
    @pytest.mark.parametrize("coef_nr, eval_domain", [
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(1)))),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(2)))),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(3)))),
        ((10,5,8), Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(3)))),
        (10, Prod(*(GridFcts(np.sign(np.arange(-10,10,2))*np.arange(-10,10,2)**2) for _ in range(2)))),
    ])
    def test_op_basic(self,coef_nr,eval_domain):
        op_basics(chebyshev_basis(coef_nr,eval_domain), test_methods=True, test_norm=False)

    x_2 = np.zeros((10,10))
    x_2[1,1] = 1
    x_3 = np.zeros((10,10,10))
    x_3[1,1,2] = 1
    x = np.linspace(-1,1,10)

    @pytest.mark.parametrize("coef_nr, eval_domain,x,res", [
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(1))), np.array([1,0,0,0,0,0,0,0,0,0]),np.ones(10)),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(1))), np.array([0,1,0,0,0,0,0,0,0,0]),np.linspace(-1,1,10)),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(1))), np.array([0,0,0,1,0,0,0,0,0,0]),4*np.linspace(-1,1,10)**3-3*np.linspace(-1,1,10)),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(2))), x_2,np.outer(x,x)),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(3))), x_3,np.outer(np.outer(x,x),2*x**2 -1).reshape(10,10,10)),
    ])
    def test_ot_eval(self,coef_nr,eval_domain, x, res):
        op_evaluation_and_ot(chebyshev_basis(coef_nr,eval_domain), x, res)

    

class TestLegendre():
    @pytest.mark.parametrize("coef_nr, eval_domain", [
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(1)))),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(2)))),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(3)))),
        ((10,5,8), Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(3)))),
        (10, Prod(*(GridFcts(np.sign(np.arange(-10,10,2))*np.arange(-10,10,2)**2) for _ in range(2)))),
    ])
    def test_op_basic(self,coef_nr,eval_domain):
        op_basics(legendre_basis(coef_nr,eval_domain), test_methods=True, test_norm=False)

    x_2 = np.zeros((10,10))
    x_2[1,1] = 1
    x_3 = np.zeros((10,10,10))
    x_3[1,4,2] = 1
    x = np.linspace(-1,1,10)

    @pytest.mark.parametrize("coef_nr, eval_domain,x,res", [
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(1))), np.array([1,0,0,0,0,0,0,0,0,0]),np.ones(10)),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(1))), np.array([0,0,1,0,0,0,0,0,0,0]),(3*x**2-1)/2),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(1))), np.array([0,0,0,0,1,0,0,0,0,0]),(35*x**4-30*x**2+3)/8),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(2))), x_2,np.outer(x,x)),
        (10, Prod(*(UniformGridFcts(np.linspace(-1,1,10)) for _ in range(3))), x_3,np.outer(np.outer(x,(35*x**4-30*x**2+3)/8),(3*x**2-1)/2).reshape(10,10,10)),
    ])
    def test_ot_eval(self,coef_nr,eval_domain, x, res):
        op_evaluation_and_ot(legendre_basis(coef_nr,eval_domain), x, res)

class TestBSpline():

    @pytest.mark.parametrize("k,t,dim,add_points", [
        (0, np.linspace(-1,1,10), 1,5),
        (1, np.linspace(-1,1,10), 1,5),
        (1, np.linspace(-1,1,6), 2,5),
        (1, np.arange(0,10,2)**2/50, 2,5),
        (1, np.arange(0,10,2)**2/50, 3,3),
    ])
    def test_op_basic(self,k,t,dim,add_points):
        op_basics(bspline_basis(k,t,dim = dim,add_points=add_points), test_methods=True, test_norm=False)

    @pytest.mark.parametrize("k,t,dim,add_points,x,res", [
        (0, np.linspace(-1,1,10), 1,5,np.ones(9),np.ones(50)),
        (1, np.linspace(-1,1,10), 1,5,None,None),
        (1, np.linspace(-1,1,6), 2,5,None,None),
        (1, np.arange(0,10,2)**2/50, 3,5,None,None),
    ])
    def test_ot_eval(self,k,t,dim,add_points, x, res):
        op_evaluation_and_ot(bspline_basis(k,t,dim = dim,add_points=add_points), x, res)