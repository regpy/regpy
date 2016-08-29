import logging
import numpy as np
import setpath
from .operators import WeightedOp

from . import Inner_Solver

__all__ = ['SQP']


class SQP(Inner_Solver):


    def __init__(self, op, data, init, x, y, alpha, it):
        super().__init__(logging.getLogger(__name__))
        self.op = op
        self.data = data
        self.init = init
        self.x = x
        self.y = y
        self.alpha = alpha    
        self.it = it
        
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
        self.smb = not self.chi
        self.h = np.zeros(len(self.x))
        self.y_kl = self.y
        self.first = 1
        self.norm_update = self.update_ratio + 1
        self.l = 1
        self.mu = 1
        
    def next(self):
        self.til_y_kl = self.y_kl
        self.weight = (self.chi * (self.data + self.offset)**0.5/np.sqrt(2)*(self.til_y_kl + self.offset))
        self.b = (1/np.sqrt(2))*self.chi/np.sqrt(self.data+self.offset) * (self.data - self.til_y_kl) - 0.5*self.smb
        self.T = WeightedOp(self.op)
        
        self.hl = CGNE_reg(blablablablablablablablaaaa)
        
        self.y_kl_update = self.op.derivative()(self.hl)
        self.l = np.find(self.y_kl_update < 0)
        self.tmp1 = -0.9 * self.offset - self.y_kl
        self.tmp2 = self.tmp[self.l]
        self.tmp3 = self.y_kl_update[self.l]
        #stepsize control
        if np.size(self.l):
            self.mu = 1
        else:
            self.mu = np.min(np.min(self.tmp2/self.tmp3),1)
        self.h += self.mu * self.hl
        self.y_kl += self.mu * self.y_kl_update
        self.norm_update = self.mu * self.op.domx.norm(self.hl)
        if self.l == 1:
            self.first = self.norm_update
        self.l += 1
        
        return self.norm_update > self.update_ratio * self.first and self.l <= self.inner_kl_it and self.mu > 0

    def run(self, stoprule=None):
        """Run the solver with the given stopping rule.

        This is convenience method that implements a simple loop running the
        solver until it either converges or the stopping rule triggers.

        Parameters
        ----------
        stoprule : :class:`StopRule <itreg.stoprules.StopRule>`, optional
            The stopping rule to be used. If omitted, stopping will only be
            based on the return value of :meth:`next`.

        """
        for x, y in self:
            self.x += self.h
            self.it += 1
            if stoprule is not None and stoprule.stop(x, y):
                self.log.info('Stopping rule triggered.')
                return stoprule.x
        self.log.info('Solver converged.')
        return x
        
        
            
            
            
            