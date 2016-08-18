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
    stepsize : float, optional#
    
    """
    
    def __init__(self, op, data, init, rho = 0.8, cgmaxit = 50):
        super().__init__(logging.getLogger(__name__)) # noch verstehen
        self.op = op
        self.data = data
        self.x_k = np.zeros(len(init))
        self.setx(init)
        self.rho = rho
        self.cgmaxit = cgmaxit
        
    def setx(self, x):
        self.x = x + self.x_k
        self.x_k = np.zeros(len(x))
        self.y = self.op(self.x)
        self._residual = self.data - self.y
        
        self.s = self._residual - self.op.derivative()(self.x_k)
        self.s2 = self.op.domy.gram(self.s)
        self.rtilde = self.op.adjoint(self.s2)
        self.r = self.op.domx.gram_inv(self.rtilde)
        self.d = self.r
        self.innerProd = self.op.domx.inner(self.r,self.rtilde)
        self.norms0 = np.sqrt(np.real(self.op.domx.inner(self.s2,self.s)))
        
        self.k = 1
     
    def next(self):
        while np.sqrt(self.op.domx.inner(self.s2,self.s)) > self.rho * self.norms0 and self.k <= self.cgmaxit:
            self.aux = self.op.derivative()(self.d)
            self.aux2 = self.op.domy.gram(self.aux)
            self.alpha = self.innerProd / np.real(self.op.domy.inner(self.aux,self.aux2))
            self.x_k = self.x_k + self.alpha * self.d
            self.s2 = self.s2 - self.alpha*self.aux2
            self.rtilde = self.op.adjoint(self.s2)
            self.r = self.op.domx.gram_inv(self.rtilde)
            self.beta = np.real(self.op.domy.inner(self.r,self.rtilde))/self.innerProd
            self.d = self.r + self.beta*self.d
            self.k += 1
        self.k -= 1
        self.setx(self.x)
        return True
            
                                        

