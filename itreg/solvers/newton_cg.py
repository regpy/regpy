import logging
import numpy as np

from . import Solver

__all__ = ['Newton_CG']


class Newton_CG(Solver):
    """The Newton-CG method. #fragen
    
    Solves the potentially non-linear, ill-posed equation ::

        T(x) = y,
        #hier weiter machen
        
    
    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    rho : int
        A factor considered for stopping the inner iteration (which is the CG method).
    cgmaxit : int
        Maximum iterations for the inner iteration (which is the CG method).
    
    """
    
    def __init__(self, op, data, init, cgmaxit = 50, rho = 0.8):
        super().__init__(logging.getLogger(__name__)) # noch verstehen
        self.op = op
        self.data = data
        self.x = init
        self.outer_update()                        # initialize certain variables
        
        # parameters for exiting the inner iteration (CG method)
        self.rho = rho
        self.cgmaxit = cgmaxit
        
    def outer_update(self):
        """
        This function does two things:
            1. Initialization of the needed variables in the outer iteration with the input init.
            2. Straight forward computations for the outer iteration. The input for this purpose is
               actually not needed.
        
        """
        self.x_k = np.zeros(np.shape(self.x))       # x_k = 0
        self.y = self.op(self.x)                    # y = T(x)
        self._residual = self.data - self.y
        self.s = self._residual - self.op.derivative()(self.x_k)
        self.s2 = self.op.domy.gram(self.s)
        self.rtilde = self.op.adjoint(self.s2)
        self.r = self.op.domx.gram_inv(self.rtilde)
        self.d = self.r
        self.innerProd = self.op.domx.inner(self.r,self.rtilde)
        self.norms0 = np.sqrt(np.real(self.op.domx.inner(self.s2,self.s)))
        self.k = 1
     
    def inner_update(self):
        """
        This function does straight forward computations of variables only.
        Its whole purpose is to increase readability.
        
        """
        self.aux = self.op.derivative()(self.d)
        self.aux2 = self.op.domy.gram(self.aux)
        self.alpha = self.innerProd / np.real(self.op.domy.inner(self.aux,self.aux2))
        self.s2 += -self.alpha*self.aux2
        self.rtilde = self.op.adjoint(self.s2)
        self.r = self.op.domx.gram_inv(self.rtilde)
        self.beta = np.real(self.op.domy.inner(self.r,self.rtilde))/self.innerProd        

    def next(self):
        while np.sqrt(self.op.domx.inner(self.s2,self.s)) > self.rho * self.norms0 and self.k <= self.cgmaxit:
            self.inner_update()
            
            self.x_k += self.alpha * self.d
            
            self.d = self.r + self.beta * self.d
            self.k += 1
        self.x += self.x_k
        self.outer_update()
        return True