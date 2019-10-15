"""Newton_CG solver """

import logging
import numpy as np

from regpy.solvers import Solver

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
    its iterations is run in each Newton iteration.

    Parameters
    ----------
    op : :class:`Operator <regpy.operators.Operator>`
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
    op : :class:`Operator <regpy.operators.Operator>`
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
        self.outer_update()

        # parameters for exiting the inner iteration (CG method)
        self.rho = rho
        self.cgmaxit = cgmaxit

    def outer_update(self):
        """Initialize and update variables in the Newton iteration."""

        self._x_k = np.zeros(np.shape(self.x))
        self.y = self.op(self.x)
        self._residual = self.data - self.y
        _, self.deriv=self.op.linearize(self.x)
        self._s = self._residual - self.deriv(self._x_k)
        self._s2 = self.op.codomain.gram(self._s)
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.op.domain.gram_inv(self._rtilde)
        self._d = self._r
        self._innerProd = self.op.domain.inner(self._r,self._rtilde)
        self._norms0 = np.sqrt(np.real(self.op.domain.inner(self._s2,self._s)))
        self._k = 1

    def inner_update(self):
        """Compute variables in each CG iteration."""
        _, self.deriv=self.op.linearize(self.x)
        self._aux = self.deriv(self._d)
        self._aux2 = self.op.codomain.gram(self._aux)
        self._alpha = (self._innerProd
                       / np.real(self.op.codomain.inner(self._aux,self._aux2)))
        self._s2 += -self._alpha*self._aux2
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.op.domain.gram_inv(self._rtilde)
        self._beta = (np.real(self.op.codomain.inner(self._r,self._rtilde))
                      / self._innerProd)

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
        self.outer_update()
        return True
