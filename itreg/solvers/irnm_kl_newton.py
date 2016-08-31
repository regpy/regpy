import logging
import numpy as np
import numpy.linalg as LA

import setpath 
from itreg.util.cg_methods import CG

from . import Solver


__all__ = ['IRNM_KL_Newton']


class IRNM_KL_Newton(Solver):

    def __init__(self, op, data, init, alpha0 = 2e-6, alpha_step = 2/3., intensity = 1, scaling = 1, offset = 1e-4, offset_step = 0.8, inner_res = 1e-10, inner_it = 10, cgmaxit = 50):
        super().__init__(logging.getLogger(__name__))
        self.op = op
        self.data = data
        self.init = init
        self.x = self.init
        self.y = self.op(self.x)
        
        # Parameter for the outer iteration (Newton method)
        self.k = 0
        self.alpha_step = alpha_step
        self.intensity = intensity
        self.data = self.data / self.intensity
        self.scaling = scaling/np.abs(self.intensity)
        self.alpha = alpha0
        self.offset = offset
        self.offset_step = offset_step
        self.inner_res = inner_res
        self.inner_it = inner_it
        self.cgmaxit = cgmaxit
            
    def next(self):
        """Run a single IRGNM_CG iteration.

        Returns
        -------
        bool
            Always True, as the IRGNM_CG method never stops on its own.

        """
#        print(self.x[1:10])
        self.k += 1
        self.h_n = np.zeros(np.shape(self.x))
        self.eta = np.zeros(np.shape(self.x))
        self.rhs = -self.grad(self.h_n)
#        print("rhs hier")
#        print(self.rhs[1:10])
        self.n = 1
        self.res = self.op.domx.norm(self.rhs)
        while self.res > self.inner_res and self.n <= self.inner_it:
            self.eta = CG(self.Ax, self.rhs/self.res, np.zeros(np.shape(self.eta)), 1e-2, self.cgmaxit)
            self.eta = self.res * self.eta
#            print("eta innerhalb while")
#            print(self.eta[1:10])
            self.h_n += self.eta
            self.rhs = -self.grad(self.h_n)
#            print("rhs innerhalb while nach fehlerschritt")
#            print(self.rhs[1:10])
            self.res = LA.norm(self.rhs)
            
            self.n += 1
        self.h = self.h_n
        
        self.x += self.h
        self.y = self.op(self.x)
        self.alpha = self.alpha * self.alpha_step
        self.offset = self.offset * self.offset_step
        return True

    def frakF(self, x):
        return np.log(self.op(x) + self.offset)
    
    def A(self, h):
        return self.op.derivative()(h)/(self.y + self.offset)
    
    def Ast(self, h):
        return self.op.adjoint(h/(self.y + self.offset))

    def grad(self, h):
#        print("argument das an Ast uebergeben wird")
#        print(((self.y + self.offset) * np.exp(self.A(h)) - self.data - self.offset)[1:10])
#        print(self.Ast((self.y + self.offset) * np.exp(self.A(h)) - self.data - self.offset)[1:10])
        return self.Ast((self.y + self.offset) * np.exp(self.A(h)) - self.data - self.offset) + 2 * self.alpha * self.op.domx.gram(self.x + h - self.init)
        
    def Dgrad(self, h, eta):
        return self.Ast((self.y + self.offset) * np.exp(self.A(h)) * self.A(eta)) + 2 * self.alpha * self.op.domx.gram(eta)
    def Ax(self, eta):
        return self.Dgrad(self.h_n, eta)
        