import numpy as np
import regpy.functionals as fct
from regpy.functionals import QuadraticIntv, LinearFunctional, HorizontalShiftDilation
import regpy.vecsps as vecsps
from regpy.vecsps import UniformGridFcts


def check_prox(F,u=None,tau=1):
    if(u is None):
        u=F.domain.rand()
    Fs = F.conj
    prox = F.proximal(u,tau)
    gram = F.h_domain.gram
    proxstar = Fs.proximal(gram(u/tau),1/tau)
    assert np.linalg.norm(u-prox-tau*gram.inverse(proxstar))<10e-15

def check_conj_and_subgradient(F,u=None,w=None):
    if(u is None):
        u=F.domain.rand()
    if(w is None):
        w=F.domain.rand()
    Fs = F.conj
    assert not F(w)==np.inf
    grad = F.subgradient(u)
    assert not Fs(grad)==np.inf
    u2 = Fs.subgradient(grad)
    assert F.is_subgradient(grad,u)#Check if functionals are subgradients of each other
    assert Fs.is_subgradient(u2,grad)
    assert np.sum(u2*grad)-F(u2)-Fs(grad)<10e-15#Young equality

def test_conj_and_subgradient_huber():
    grid = UniformGridFcts((-1,1,5))
    w = np.random.randint(10,size=(5,))+1.2
    F0 = fct.Huber(grid,sigma=1./w)
    F = fct.HorizontalShiftDilation(F0,dilation=0.5,shift=0.01*grid.ones())
    check_conj_and_subgradient(F)

def test_prox_huber():
    grid = UniformGridFcts((-1,1,5))
    w = np.random.randint(10,size=(5,))+1.2
    F0 = fct.Huber(grid,sigma=1./w)
    F = fct.HorizontalShiftDilation(F0,dilation=0.5,shift=0.01*grid.ones())
    check_prox(F)

