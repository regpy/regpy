import numpy as np

from regpy.vecsps import NumPyVectorSpace, MeasureSpaceFcts,UniformGridFcts
from regpy.functionals import *
from regpy.functionals.base import HorizontalShiftDilation, LinearFunctional
from regpy.hilbert import L2
from regpy.util import functional_tests as ft


def test_Lpp():
    dom = UniformGridFcts((-3.,3.,100))
    x = np.linspace(-3.,3.,100)
    #Numerical prox does not yet work together with options quad_taylor_x and lin_taylor_x!
    #for p in [1.5,2.,2.5]:
    #    for (l,u) in [(0.1,2.3),(-2.1,-1.2),(-1.2,1.)]:
    #        #print('p=',p,'l=',l,'u=',u)
    #        ft.test_functional(Lpp(dom,p=p,constr_l=l,quad_taylor_u=u))
    #        ft.test_functional(Lpp(dom,p=p,quad_taylor_l=l,lin_taylor_u=u))
    #        ft.test_functional(Lpp(dom,p=p,lin_taylor_l=l,constr_u=u))
    #        ft.test_functional(Lpp(dom,p=p,quad_taylor_l=l,quad_taylor_u=u))
    ft.test_functional(Lpp(dom,p=1.5,constr_l=-2,quad_taylor_u=3.),test_second_deriv_conj=False)
    ft.test_functional(Lpp(dom,p=2.5,quad_taylor_l=1.1,quad_taylor_u=2.),test_second_deriv=False)

def test_L1():
    dom = NumPyVectorSpace((2,10)) 
    x = np.linspace(-5,4.5,20).reshape(2,10)
    func = L1(dom)
    assert (func(x) == 50.0)
    for tau in [0.1,1,2]:
        assert (func.proximal(x,tau) == np.maximum(0, np.abs(x)-tau)*np.sign(x)).all()
    ft.test_functional(func,test_second_deriv=False)
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    x = dom.ones()
    func = L1(dom)
    assert (func(x) == 21.0)
    ft.test_functional(func) #u_s=[func.domain.rand() for _ in range(5)],
                       #u_stars=[func.domain.rand() for _ in range(5)],
                       #test_second_deriv=False)

# def test_TV():
#     ugf = UniformGridFcts((-1,1,10),(-1,1,10))
#     func = TV(ugf)
#     ft.test_functional(func,u_s=[func.domain.rand() for _ in range(5)],u_stars=[func.domain.rand() for _ in range(5)])

def test_kullback_leibler():
    dom=UniformGridFcts((-1,1,10),(-2,3,3))
    F=KL(dom,w=dom.ones())
    ft.test_functional(F)
    
    ft.test_functional(HorizontalShiftDilation(F,dilation=3.,shift=F.domain.ones()))
    ft.test_functional(F-2.)

    F2 = KL(dom,w=dom.ones(),quad_taylor_l=0.5,constr_u=5.)
    ft.test_functional(F2,test_second_deriv_conj=False)
    ft.test_functional(4.*F2+LinearFunctional(F2.domain.ones(),domain=F2.domain),test_second_deriv_conj=False)
    ft.test_functional(HorizontalShiftDilation(F2,dilation=3.,shift=-F2.domain.ones()),test_second_deriv_conj=False)

    w=1.+0.5*np.sin(dom.coords[0]*dom.coords[1])
    F3 = KL(dom,w=5*dom.ones(),lin_taylor_l=0.1,quad_taylor_u=2.5)
    ft.test_functional(F3)

    F4 = KL(dom,w=dom.ones(),constr_l=0.3,lin_taylor_u=2.5)
    ft.test_functional(F4)

def test_relative_entropy():
    dom=UniformGridFcts((-5,7,4),(100,200,3))
    F=RE(dom,w=dom.ones())
    ft.test_functional(F)

    F2 = RE(dom,w=dom.ones(),lin_taylor_l=0.2)
    ft.test_functional(2.*F2,test_second_deriv=False)
    ft.test_functional(F2+LinearFunctional(F2.domain.ones(),domain=F2.domain),test_second_deriv=False)
    ft.test_functional(HorizontalShiftDilation(F2,dilation=3.,shift=F2.domain.ones()),test_second_deriv=False)

    F3 = RE(dom,w=dom.ones(),constr_u=3.)
    ft.test_functional(F3)    

def test_huber():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.complex128)
    sigma=np.real(dom.ones())
    sigma[0,0]=4
    F=Hub(dom,sigma=sigma,eps=1e-10)
    ft.test_functional(F)

    ft.test_functional(HorizontalShiftDilation(F,dilation=3,shift=F.domain.ones()))
    ft.test_functional(F-2.)
    ft.test_functional(F+LinearFunctional(0.5*F.domain.ones(),domain=F.domain),
                       test_second_deriv=False,test_second_deriv_conj=False
                       )    

def test_quadratic_intv():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.complex128)
    sigma=np.real(dom.ones())
    sigma[0,0]=4
    F=QuadIntv(dom,sigma=sigma,eps=1e-10)
    assert F(2*dom.ones())==np.inf
    ft.test_functional(F)

def test_quadnonneg():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    func = QuadNonneg(dom)
    ft.test_functional(func)

def test_quadbil():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    func = QuadBil(dom,lb = dom.zeros(), ub = dom.ones())
    ft.test_functional(func)

def test_quadlow():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64))
    func = QuadLow(dom)
    ft.test_functional(func)

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
    ft.test_functional(F_tr,u_s_tr,test_second_deriv=False)

def test_hilbertnorm():
    dom=MeasureSpaceFcts(measure=np.array([[1,2,3],[4,5,6]],dtype=np.float64),dtype=np.complex128)
    l2 = L2(dom)
    func = HilbertNorm(l2)
    ft.test_functional(func)
