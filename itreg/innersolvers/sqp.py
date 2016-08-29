import logging
import numpy as np
import scipy
import setpath
from itreg.util import CGNE_reg
from itreg.operators import WeightedOp

from . import Inner_Solver

__all__ = ['SQP']


class SQP(Inner_Solver):


    def __init__(self, op, data, init, x_input, y_input, alpha, it, intensity = 1):
        super().__init__(logging.getLogger(__name__))
        self.op = op
        self.data = data
        self.init = init
        self.x = x_input + 0j
        self.y = y_input
        self.alpha = alpha    
        self.it = it
        self.intensity = intensity
        
        #some parameters
        # maximum number of CG iterations
        self.N_CG = 50
        # replace KL(a,b) by KL(a+offset, b+offset)
        self.offset0 =2e-6
        # offset is reduced in each Newton step by a factor offset_step
        self.offset_step = 0.8
        # relative tolerance value for termination of inner iteration
        self.update_ratio = 0.01
        # max number of inner iterations to minimize the KL-functional
        self.inner_kl_it = 10
        
        self.preparations()
        
    def preparations(self):
        self.offset = self.offset0 * self.intensity * self.offset_step **(self.it - 1) 
        self.chi = self.data != -self.offset
        self.smb = scipy.logical_not(self.chi)
        self.h = np.zeros(len(self.x)) + 0j
        self.y_kl = self.y +0j
        self.first = 1
        self.norm_update = self.update_ratio + 1
        self.l = 1
        self.mu = 1
        
    def next(self):
        self.cont = (self.norm_update > self.update_ratio * self.first and self.l <= self.inner_kl_it and self.mu > 0)
        self.til_y_kl = self.y_kl
        self.weight = (self.chi * (self.data + self.offset + 0j)**0.5/(np.sqrt(2.)*(self.til_y_kl + self.offset)))
        self.b = (1/np.sqrt(2))*self.chi/np.sqrt(self.data+self.offset + 0j) * (self.data - self.til_y_kl) - 0.5*self.smb
        self.opw = WeightedOp(self.op, self.weight)
        
        self.hl = CGNE_reg(op = self.opw, y = self.b, xref = self.init - self.x - self.h, regpar = self.alpha, cgmaxit = self.N_CG)
        
        self.y_kl_update = self.op.derivative()(self.hl)
        self.mask = self.y_kl_update < 0
        self.tmp1 = -0.9 * self.offset - self.y_kl
        self.tmp2 = self.tmp1[self.mask]
        self.tmp3 = self.y_kl_update[self.mask]
        #stepsize control
        if not np.any(self.mask):
            self.mu = 1
        else:
            self.mu = min(np.min(self.tmp2/self.tmp3),1)
        self.h += self.mu * self.hl
        self.y_kl += self.mu * self.y_kl_update
        self.norm_update = self.mu * self.op.domx.norm(self.hl)
        if self.l == 1:
            self.first = self.norm_update
        self.l += 1
        
        if not self.cont:
            self.x += self.h
            self.it += 1
        return self.cont    
        
      