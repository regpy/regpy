import logging
import numpy as np
import scipy

from . import Solver
#impoort inner solver sqp

__all__ = ['IRNM_KL']


class IRNM_KL(Solver):

    def __init__(self, op, data, init, alpha0 = 5e-6, alpha_step = 2/3., intensity = 1):
        super().__init__(logging.getLogger(__name__))
        self.op = op
        self.data = data
        self.init = init
        self.x = self.init
        self.y = self.op(self.x)
        
        # Parameter for the outer iteration (Newton method)
        self.k = 0   
        self.alpha_step = alpha_step
        self.alpha = alpha0 * intensity
            
    def next(self):
        """Run a single IRGNM_CG iteration.

        Returns
        -------
        bool
            Always True, as the IRGNM_CG method never stops on its own.

        """
        self.x = sqp(self.op, self.data, self.init, self.x, self.y, self.alpha, self.k)
        self.k += 1
        self.y = self.op(self.x)
        self.alpha *= self.alpha_step
        
        return True
