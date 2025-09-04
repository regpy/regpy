import numpy as np

from regpy.vecsps import NumPyVectorSpace, MeasureSpaceFcts,UniformGridFcts
from regpy.functionals import *
from regpy.hilbert import L2
from regpy.util import functional_tests as ft

def test_L1():
    dom = NumPyVectorSpace((2,10)) 
    x = np.linspace(-5,4.5,20).reshape(2,10)
    func = L1(dom)
    assert (func(x) == 50.0)
    for tau in [0.1,1,2]:
        assert (func.proximal(x,tau) == np.maximum(0, np.abs(x)-tau)*np.sign(x)).all()
    ft.test_functional(func)
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    x = dom.ones()
    func = L1(dom)
    assert (func(x) == 21.0)
    ft.test_functional(func,u_s=[func.domain.rand() for _ in range(5)],u_stars=[func.domain.rand() for _ in range(5)])

# def test_TV():
#     ugf = UniformGridFcts((-1,1,10),(-1,1,10))
#     func = TV(ugf)
#     ft.test_functional(func,u_s=[func.domain.rand() for _ in range(5)],u_stars=[func.domain.rand() for _ in range(5)])

def test_kullback_leibler():
    dom=UniformGridFcts(2,2)
    F=KL(dom,w=dom.ones())
    u_s=[i*dom.rand() for i in range(1,11)]
    u_stars=[dom.rand() for i in range(10)]
    ft.test_functional(F,u_s=u_s,u_stars=u_stars)

def test_relative_entropy():
    dom=UniformGridFcts(2,2)
    F=RE(dom,w=dom.ones())
    u_s=[i*dom.rand() for i in range(1,11)]
    ft.test_functional(F,u_s=u_s)

def test_huber():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.complex128)
    sigma=np.real(dom.ones())
    sigma[0,0]=4
    F=Hub(dom,sigma=sigma,eps=1e-10)
    #essential domain of conjugate functional is |u_i|<=sigma
    u_stars=[dom.rand() for _ in range(5)]
    for i in range(len(u_stars)):
        scales=np.random.uniform(0,1,dom.shape)
        u_stars[i]*=scales/np.abs(u_stars[i])
    assert F(2*dom.ones())==32.0
    ft.test_functional(F,u_stars=u_stars)

def test_quadratic_intv():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.complex128)
    sigma=np.real(dom.ones())
    sigma[0,0]=4
    F=QuadIntv(dom,sigma=sigma,eps=1e-10)
    #essential domain of functional is |u_i|<=sigma
    u_s=[dom.rand() for _ in range(5)]
    for i in range(len(u_s)):
        scales=np.random.uniform(0,1,dom.shape)
        u_s[i]*=scales/np.abs(u_s[i])
    assert F(2*dom.ones())==np.inf
    ft.test_functional(F,u_s=u_s)

def test_quadnonneg():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    func = QuadNonneg(dom)
    ft.test_functional(func,u_s=[func.domain.rand() for _ in range(5)])

def test_quadbil():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    func = QuadBil(dom,lb = dom.zeros(), ub = dom.ones())
    ft.test_functional(func,u_s=[func.domain.rand() for _ in range(5)])

def test_quadlow():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    func = QuadLow(dom)
    ft.test_functional(func,u_s=[func.domain.rand() for _ in range(5)])

def test_quadratic_positive_semidef():
    N=5
    dom=UniformGridFcts(N,N,dtype=np.complex128)
    #without trace constraint
    F=QuadPosSemi(dom,tol=1e-10)
    samples=10
    orthos=np.random.randn(samples,2,N,N)
    orthos=orthos[:,0]+1j*orthos[:,1]
    orthos=np.linalg.qr(orthos)[0]
    diags=np.random.uniform(0,20,size=(samples,N))
    u_s=[orthos[i]@np.diag(diags[i])@np.conj(orthos[i].T) for i in range(samples)]
    assert np.abs(0.5*np.sum(diags[0]**2)-F(u_s[0]))<1e-10
    ft.test_functional(F,u_s=u_s,test_conj=False)
    #with trace constraint
    F_tr=QuadPosSemi(dom,trace_val=2,tol=1e-10)
    u_s_tr=[2*u/np.trace(u) for u in u_s]
    assert np.abs(0.5*np.sum((2*diags[0]/np.sum(diags[0]))**2)-F_tr(u_s_tr[0]))<1e-10
    ft.test_functional(F_tr,u_s_tr)

def test_hilbertnorm():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.complex128)
    l2 = L2(dom)
    func = HilbertNorm(l2)
    ft.test_functional(func)
