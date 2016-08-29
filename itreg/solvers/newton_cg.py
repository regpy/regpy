import logging
import numpy as np

from . import Solver

__all__ = ['Newton_CG']


class Newton_CG(Solver):
    """The Newton-CG method. 
    
    
    Solves the potentially non-linear, ill-posed equation ::

        T(x) = y,

    where `T` is a Frechet-differentiable operator. The number of iterations is
    effectively the regularization parameter and needs to be picked carefully.

    The Newton equations are solved by the conjugate gradient
    method applied to the normal equation (CGNE)
    using the regularizing properties of CGNE with early stopping
    (see Hanke 1997).

        
    
    Parameters
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
        A factor considered for stopping the inner iteration (which is the CG method).
        
    Attributes
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    x : array
        The current point.
    y : array
        The value at the current point.
    deriv : :class:`LinearOperator <itreg.operators.LinearOperator>`
        The derivative of the operator at the current point.
    stepsize : float
        The step length to be used in the next step.
    
    
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
        self._x_k = np.zeros(np.shape(self.x))       # x_k = 0
        self.y = self.op(self.x)                    # y = T(x)
        self._residual = self.data - self.y
        self._s = self._residual - self.op.derivative()(self._x_k)
        self._s2 = self.op.domy.gram(self._s)
        self._rtilde = self.op.adjoint(self._s2)
        self._r = self.op.domx.gram_inv(self._rtilde)
        self._d = self._r
        self._innerProd = self.op.domx.inner(self._r,self._rtilde)
        self._norms0 = np.sqrt(np.real(self.op.domx.inner(self._s2,self._s)))
        self._k = 1
     
    def inner_update(self):
        """
        This function does straight forward computations of variables only.
        Its whole purpose is to increase readability.
        
        """
        self._aux = self.op.derivative()(self._d)
        self._aux2 = self.op.domy.gram(self._aux)
        self._alpha = self._innerProd / np.real(self.op.domy.inner(self._aux,self._aux2))
        self._s2 += -self._alpha*self._aux2
        self._rtilde = self.op.adjoint(self._s2)
        self._r = self.op.domx.gram_inv(self._rtilde)
        self._beta = np.real(self.op.domy.inner(self._r,self._rtilde))/self._innerProd        

    def next(self):
        """Run a single Newton_CG iteration.

        Returns
        -------
        bool
            Always True, as the Newton_CG method never stops on its own.

        """
        while np.sqrt(self.op.domx.inner(self._s2,self._s)) > self.rho * self._norms0 and self._k <= self.cgmaxit:
            self.inner_update()
            
            self._x_k += self._alpha * self._d
            
            self._d = self._r + self._beta * self._d
            self._k += 1
        self.x += self._x_k
        self.outer_update()
        return True