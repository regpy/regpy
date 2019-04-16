# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:53:58 2019

@author: Hendrik Müller
"""

from itreg.operators import NonlinearOperator, OperatorImplementation, Params
from itreg.util import instantiate

import numpy as np
import math as mt


class ReactionCoefficient(NonlinearOperator):
    

    def __init__(self, domain, rhs, boundary_values=None, range=None, spacing=1):
        range = range or domain
        if boundary_values is None:
            boundary_values=np.zeros(2)
            boundary_values[0]=rhs[0]
            boundary_values[-1]=rhs[-1]
        #assert len(domain.shape) == 1
        #assert domain.shape == range.shape
        super().__init__(Params(domain, range, rhs=rhs, boundary_values=boundary_values, spacing=spacing))
        
    @instantiate
    class operator(OperatorImplementation):
        def eval(self, params, c, data, differentiate, **kwargs):
            B=B_builder(params, c)
            r_c=rc(params, c)
            rhs=rhs_builder(params, r_c)
            coeff= np.linalg.solve(B, rhs)
            res=np.zeros(params.domain.size_support)
            for i in range(0, params.domain.size_support):
                res=res+coeff[i]*basisfunc(params, i)
            if differentiate:
                data.B = B
                data.u=res+tilde_g_builder(params, params.boundary_values)
            return res+tilde_g_builder(params, params.boundary_values)

    @instantiate
    class derivative(OperatorImplementation):
        def eval(self, params, x, data, **kwargs):
            rhs=rhs_builder(params, -data.u*x)
            coeff=np.linalg.solve(data.B, rhs)
            res=np.zeros(params.domain.size_support)
            for i in range(0, params.domain.size_support):
                res=res+coeff[i]*basisfunc(params, i)
            return res

        def adjoint(self, params, y, data, **kwargs):
            rhs=rhs_builder(params, y)
            coeff=np.linalg.solve(data.B, rhs)
            res=np.zeros(params.domain.size_support)
            for i in range(0, params.domain.size_support):
                res=res+coeff[i]*basisfunc(params, i)
            return -data.u*res
            
            
            
def tilde_g_builder(params, boundary_values):
    tilde_g=np.interp(params.domain.coords, np.asarray([params.domain.coords[0], params.domain.coords[-1]]), boundary_values)
    v_star=tilde_g
    return v_star    

def rc(params, c):
    res=mylaplace(params, tilde_g_builder(params, params.boundary_values))
    return params.rhs+res-c*tilde_g_builder(params, params.boundary_values)    

def L(params, c, u):
    res=mylaplace(params, -u)
    return res+c*u

def basisfunc(params, i):
    return np.sin((i+1)*mt.pi*params.domain.coords)  

def basisfunc_der(params, i):
    return (i+1)*mt.pi*np.cos((i+1)*mt.pi*params.domain.coords)
     
def B_builder(params, c):
    B=np.zeros((params.domain.size_support, params.domain.size_support))
    for i in range(0, params.domain.size_support):
        for j in range(0, params.domain.size_support):
            B[i,j]=np.trapz(basisfunc(params, i)*basisfunc(params, j)*c, params.domain.coords)+np.trapz(basisfunc_der(params, i)*basisfunc_der(params, j), params.domain.coords)
    return B    

def rhs_builder(params, r_c):

    res=np.zeros(params.domain.size_support)
    for i in range(0, params.domain.size_support):
        res[i]=np.trapz(r_c*basisfunc(params, i), params.domain.coords)
    return res

def mylaplace(params, func):
    N=params.domain.size_support
    der=np.zeros(N)
    for i in range(1, N-1):
        der[i]=(func[i+1]+func[i-1]-2*func[i])/(1/N**2)
    der[0]=der[1]
    der[-1]=der[-2]
    return der  
            
            
            
            
            
            
            
            
            
            
            