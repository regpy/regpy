import numpy as np
import logging

from regpy.vecsps import NumPyVectorSpace, MeasureSpaceFcts,UniformGridFcts
from regpy.functionals import *
from regpy.functionals.base import HorizontalShiftDilation, LinearFunctional, FunctionalOnDirectSum
from regpy.functionals.numpy import VectorIntegralFunctional, LppL2, L1L2, HuberL2
from regpy.hilbert import L2
from regpy.util import functional_tests as ft


def test_Lpp():
    dom = UniformGridFcts((-3.,3.,100))
    for p in [1.5,2.,2.5]:
        for (l,u) in [(0.1,2.3),(-2.1,-1.2),(-1.2,1.)]:
            msg = f" Testing Lpp functionals for p={p}, l={l}, u={u}"
            logging.info(msg)
            print(msg)
            ft.test_functional(Lpp(dom,p=p,constr_l=l,quad_taylor_u=u),
                               test_second_deriv=(p>=2),test_second_deriv_conj=False,msg=msg
                               )
            ft.test_functional(Lpp(dom,p=p,quad_taylor_l=l,lin_taylor_u=u),
                               test_second_deriv=(p>=2),test_second_deriv_conj=(p<=2),msg=msg
                               )
            ft.test_functional(Lpp(dom,p=p,lin_taylor_l=l,constr_u=u),
                               test_second_deriv=False,test_second_deriv_conj=False,msg=msg
                               )
            ft.test_functional(Lpp(dom,p=p,quad_taylor_l=l,quad_taylor_u=u),
                               test_second_deriv=(p>=2),test_second_deriv_conj=(p<=2),msg=msg
                               )

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

def test_TV():
     ugf = UniformGridFcts((-1,1,10),(-1,1,10))
     func = TV(ugf)
     func(ugf.rand())
     func.proximal(ugf.rand(),1.)

def test_kullback_leibler():
    dom=UniformGridFcts((-1,1,10),(-2,3,5))
    F=KL(dom,w=dom.ones())
    ft.test_functional(F)
    
    ft.test_functional(HorizontalShiftDilation(F,dilation=3.,shift=F.domain.ones()))
    ft.test_functional(F-2.)


    w=1.+0.5*np.sin(dom.coords[0]*dom.coords[1])   
    F2 = KL(dom,w=w,quad_taylor_l=0.5,quad_taylor_u=5.)
    ft.test_functional(F2)
    ft.test_functional(4.*F2+LinearFunctional(F2.domain.ones(),domain=F2.domain))
    ft.test_functional(HorizontalShiftDilation(F2,dilation=3.,shift=-F2.domain.ones()))

    F3 = KL(dom,w=4.*dom.ones(),lin_taylor_l=0.1,quad_taylor_u=2.5)
    ft.test_functional(F3)

    F4 = KL(dom,w=dom.ones(),constr_l=0.3,lin_taylor_u=2.5)
    ft.test_functional(F4,test_second_deriv=False,test_second_deriv_conj=False)

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

def test_Composed():
    dom = UniformGridFcts((0,1,10))
    func = Lpp(dom,p=2.5) * np.arange(1,11)
    ft.test_functional(func,test_second_deriv_conj=False)
    func2 = Lpp(dom,p=1.5) * np.arange(1,11)
    ft.test_functional(func2,test_second_deriv=False)

def test_VectorIntegralFunctional():
    grid = UniformGridFcts((-1,1,10))
    N_v = 5
    vgrid = grid.vector_valued_space(N_v)
    for p in  [1.5,2,4]:
        ft.test_functional(LppL2(vgrid,p=p))
    ft.test_functional(L1L2(vgrid)) 
    for sigma in [1e-2,1e-1,1,10.]:
        HuberL2 = VFunc(vgrid,scalar_func=Hub(sigma = sigma))
        u_s = [ft.sample_vector_in_domain(HuberL2) for _ in range(5)]
        u_stars = [ft.sample_vector_in_domain(HuberL2.conj) for _ in range(5)]
        ft.test_functional(HuberL2, u_s = u_s, u_stars= u_stars,
                          test_second_deriv=False, test_second_deriv_conj=False)

def test_L1_dist_subdiff():
    # further tests of dist_subdiff in  test_subgradient_conj_subgradient_dist_subdiff in functional_tests
    grid = UniformGridFcts((-0.5,0.5,10),periodic=True)
    ran = grid.rand()
    G = L1(grid).shift(ran)
    if not np.isclose(G.dist_subdiff(2*grid.measure*grid.ones(),ran),1.):
        raise RuntimeError('Distance to subdifferential should be 1.')
    H = 3.+L1(grid).dilation(3.)
    x = grid.zeros()
    grad = 3*np.sign(np.linspace(-1,1,10))*grid.measure
    if not np.isclose(H.dist_subdiff(grad,x),0.):
        raise RuntimeError('3*sgn should be in the subdifferential of int |3t| dt.')

def test_FunctionalOnDirectSum():
    grid1 = UniformGridFcts((0,1,10))
    grid2 = UniformGridFcts((0,1,12))
    f1 = KL(grid1,w=grid1.ones())
    f2 = L1(grid2)

    f = FunctionalOnDirectSum((f1,f2))
    ft.test_functional(f)
    ft.test_functional(f+2.)
    ft.test_functional(f.shift(f.domain.ones()))

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
