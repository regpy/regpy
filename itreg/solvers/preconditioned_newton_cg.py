# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:51:50 2019

@author: Björn Müller
"""

"""Newton_CG solver using Lanczos as preconditioner"""

import logging
import numpy as np

from . import Solver

__all__ = ['Newton_CG']


class Newton_CG(Solver): 

    """The Newton-CG method.
    
    Solves the potentially non-linear, ill-posed equation:

        T(x) = y,

    where T is a Frechet-differentiable operator. The number of iterations is
    effectively the regularization parameter and needs to be picked carefully.

    The Newton equations are solved by the conjugate gradient method applied to
    the normal equation (CGNE) using the regularizing properties of CGNE with 
    early stopping (see Hanke 1997).
    The "outer iteration" and the "inner iteration" are referred to as the 
    Newton iteration and the CG iteration, respectively. The CG method with all
    its iterations is run in each Newton iteration. We use Lanczos method to compute
    a preconditioner.

    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    cgmaxit : int, optional
        Maximum iterations for the inner iteration (where the CG method is run).
    rho : float, optional
        A factor considered for stopping the inner iteration (which is the 
        CG method).
        
    Attributes
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    cgmaxit : int, optional
        Maximum iterations for the inner iteration (which is the CG method).
    rho : float, optional
        A factor considered for stopping the inner iteration (which is the 
        CG method).
    x : array
        The current point.
    y : array
        The value at the current point.
    """
    
    def __init__(self, op, data, init, cgmaxit=50, rho=0.8):
        """Initialization of parameters"""
        
        super().__init__()
        self.op = op
        self.data = data
        self.x = init
        
        # 
        self._n=0
        self.precondtioner_update=True
        self.eigval_num=3
        self.orthonormal=np.zeros((self.eigval_num, self.data.shape[0]))
        self.outer_update()
        
        # parameters for exiting the inner iteration (CG method)
        self.rho = rho
        self.cgmaxit = cgmaxit
        
        
        
    def outer_update(self):
        """Initialize and update variables in the Newton iteration."""
        if int(self._n/10)*10==self._n:
            _, self.deriv=self.op.linearize(self.x)
            self.precondtioner_update=True
        self._x_k = np.zeros(np.shape(self.x))       
        self.y = self.op(self.x)                   
        self._residual = self.data - self.y
        _, self.deriv=self.op.linearize(self.x)
        self._s = self._residual - self.deriv(self._x_k)
        self._s2 = self.op.range.gram(self._s)
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.op.domain.gram_inv(self._rtilde)
        self._d = self._r
        self._innerProd = self.op.domain.inner(self._r,self._rtilde)
        self._norms0 = np.sqrt(np.real(self.op.domain.inner(self._s2,self._s)))
        self._k = 1
        self._n+=1
        self.lanzcos_update()
        
        
        
     
    def inner_update(self):
        """Compute variables in each CG iteration. The CG iteration is performed
        in solving the normal equation"""
        
        self._aux = self.pre_cond_deriv.dot(self._d)
        self._aux2 = self.op.domain.gram(self._d)
        self._alpha = (self._innerProd
                       / np.real(self.op.range.inner(self._aux,self._aux2)))
        self._s2 += -self._alpha*self._aux2
        self._rtilde = self.pre_cond_deriv.dot(self._s2)
        self._r = self.op.domain.gram_inv(self._rtilde)
        self._beta = (np.real(self.op.range.inner(self._r,self._rtilde))
                      / self._innerProd)  
        if self.precondtioner_update==True and self._k<=self.eigval_num:
            self.orthonormal[self._k, :]=self._s2/self.op.range.norm(self._s2)
        
#        self.z=self.deriv(self._d)

    def next(self):
        """Run a single Newton_CG iteration.

        Returns
        -------
        bool
            Always True, as the Newton_CG method never stops on its own.

        """
        while (np.sqrt(self.op.domain.inner(self._s2,self._s)) 
               > self.rho*self._norms0 and
               self._k <= self.cgmaxit):
            self.inner_update()
            self._x_k += self._alpha*self._d       
            self._d = self._r + self._beta*self._d
            self._k += 1
        
        # Updating ``self.x``
        self.x += self._x_k
        
        if self.precondtioner_update==True:
            self.lanzcos_update()
            
        self.precondtioner_update=False       
                
        self.outer_update()
        return True
    
    
    
    def lanzcos_update(self):
        """perform lanzcos method to calculate the preconditioner"""
        self.deriv_mat=np.zeros((self.op.domain.shape[0], self.op.domain.shape[0]))
        self.L=np.zeros((self.eigval_num, self.eigval_num))
        _, self.deriv=self.op.linearize(self.x)
        for i in range(0, self.eigval_num):
            self.L[i, :]=np.dot(self.orthonormal, self.deriv.adjoint(self.op.range.gram(self.deriv((self.orthonormal[i, :])))))
            self.deriv_mat[i, :]=self.deriv.adjoint(self.deriv(self.x))
        self.lamb, self.U=np.linalg.eig(self.L)
        self.lanczos=np.dot(self.orthonormal.transpose(), self.U)
#        self.M=self._alpha*np.identity(self.data.shape[0])
        self.M=np.zeros((self.data.shape[0], self.data.shape[0]))
        for  i in range(0, self.eigval_num):
            self.M[i, :]+=self.lanczos[:, i]*self.lamb[i]
        self.pre_cond_deriv=np.dot(self.M.transpose(), np.dot(self.deriv_mat, self.M))
        self.C=np.dot(self.pre_cond_deriv, self.pre_cond_deriv.transpose())