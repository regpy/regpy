"""IRGNM_L1_fid solver """

import logging
import numpy as np
import scipy
import scipy.optimize

from regpy.solvers import Solver

__all__ = ['IRGNM_L1_fid']


class IRGNM_L1_fid(Solver):
    """The IRGNM_L1_fid method.

    Solves the potentially non-linear, ill-posed equation:

      .. math::  T(x) = y,

    where :math:`T` is a Frechet-differentiable operator. The number of
    iterations is effectively the regularization parameter and needs to be
    picked carefully.

    IRGNM stands for Iteratively Regularized Gauss Newton Method. The L1 means
    that this algorithm includes some kind of :math:`L^1` regularization term.
    The fid stands for fidelity, as we include some kind of fidelity term.

    In this algorithm the function ``scipy.optimize.minimize`` is used to solve
    a nonlinear functional. In our case this is ``self._func``. For more
    information about this method, see the scipy documentation for
    scipy.optimize.minimize .

    Parameters
    ----------
    op : :class:`Operator <regpy.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    alpha0 : float, optional
    alpha_step : float, optional
        With these (alpha0, alpha_step) we compute the regulization parameter
        for the k-th Newton step by alpha0*alpha_step^k.
    alpha_l1 : float, optional
        Parameter for the boundaries for the solution of
        ``scipy.optimize.minimize``.

    Attributes
    ----------
    op : :class:`Operator <regpy.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    k : int
        The k-th iteration.
    x : array
        The current point.
    y : array
        The value at the current point.
    alpha0 : float
    alpha_step : float
        Needed for the computation of the regulization parameter for the k-th
        iteration.
    alpha_l1 : float
        Parameter for the boundaries for the solution of
        ``scipy.optimize.minimize``.
    """

    def __init__(self, op, data, init,
                 alpha0=1, alpha_step=0.9, alpha_l1=1e-4):
        """Initialize parameters."""

        #super().__init__(logging.getLogger(__name__))
        super().__init__()
        self.op = op
        self.data = data
        self.init = init
        self.x = self.init
        self.y = self.op(self.x)

        # Initialization of some parameters for the iteration
        self.k = 0
        self.alpha0 = alpha0
        self.alpha_step = alpha_step
        self.alpha_l1 = alpha_l1

        # Computation of some parameters for the iteration
        self._GramX = self.op.domain.gram(np.eye(len(self.x)))
        self._GramX = 0.5 * (self._GramX+self._GramX.T)
        self._GramY = self.op.codomain.gram(np.eye(len(self.y)))
        self._GramY = 0.5 * (self._GramY+self._GramY.T)
        self._maxiter = 10000

    def update(self):
        """Compute and update variables for each iteration.

        Uses the scipy function ``scipy.optimize.minimize`` to minimize the
        functional ``self._func`` with ``self._iter`` number of iterations. If
        successfull, ``self._iter`` is doubled and the procedure is repeated
        until it is not successful. Then the while loop will be exited.
        """
        self.k += 1

        # Preparations for the minimization procedure
        self._regpar = self.alpha0 * self.alpha_step**self.k
        self._DF = np.eye(len(self.y), len(self.x))
        _, deriv=self.op.linearize(self.x)
        for i in range(len(self.x)):
            self._DF[:,i] = deriv(self._DF[:,i])
        self._Hess = (np.dot(self._DF,np.linalg.solve(self._GramX, self._DF.T))
                      + self._regpar*np.linalg.inv(self._GramY))
        self._Hess = 0.5 * (self._Hess+self._Hess.T)
        self._rhs = self.data - self.y - np.dot(self._DF, self.init - self.x)
#        self.alpha_l1 = np.max(self._regpar, self.alpha_l1)
        self.alpha_l1=np.asarray([self._regpar, self.alpha_l1]).max()

        # Parameters for the minimization procedure
        self._iter = 100
        self._exitflag = True

        # Minimization procedure
        while self._iter < self._maxiter and self._exitflag:
            self._bounds = (-(1/self.alpha_l1) * np.ones(len(self.y)),
                            (1/self.alpha_l1) * np.ones(len(self.y)))
            self._options = {'maxiter':self._iter, 'disp':False}
            self._quadprogsol = scipy.optimize.minimize(
                                self._func,
                                bounds = np.transpose(self._bounds).tolist(),
                                options=self._options,
                                x0 = np.zeros(len(self.x))
                                )
            self._updateY = self._quadprogsol.x
            self._exitflag = self._quadprogsol.success
            self._iter *= 2

    def _func(self, x):
        """Define the functional for ``scipy.optimize.minimize``."""

        return 0.5*np.dot(x.T,np.dot(self._Hess,x)) - np.dot(self._rhs.T,x)

    def next(self):
        """Run a single IRGNM_L1_fid iteration.

        The actual computation happens in ``self.update``. In the end,
        ``self.x`` is updated as well as ``self.y``.

        Returns
        -------
        bool
            Always True, as the IRGNM_L1_fid method never stops on its own.

        """
        self.update()
        self.x = (self.init
                  + self.op.domain.gram_inv(np.dot(self._DF.T,self._updateY)))
        self.y = self.op(self.x)
        return True
