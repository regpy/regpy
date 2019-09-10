"""IRGNM_CG solver """

import logging
import numpy as np

from . import Solver

__all__ = ['IRGNM_CG']


class IRGNM_CG(Solver):
    """The IRGNM_CG method.

    Solves the potentially non-linear, ill-posed equation:

       .. math:: T(x) = y,

    where   :math:`T` is a Frechet-differentiable operator. The number of 
    iterations is effectively the regularization parameter and needs to be
    picked carefully.
    
    IRGNM stands for Iteratively Regularized Gauss Newton Method. CG stands for
    the Conjugate Gradient method. The regularized Newton equations are solved
    by the conjugate gradient method applied to the normal equation. The "outer
    iteration" and the "inner iteration" are referred to as the Newton
    iteration and the CG iteration, respectively. The CG method with all its
    iterations is run in each Newton iteration.
    
    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    cgmaxit : int, optional
        Maximum number of CG iterations.
    cgtol : list of float, optional
        Contains three tolerances:
        The first entry controls the relative accuracy of the Newton update in
        preimage (space of "x").
        The second entry controls the relative accuracy of the Newton update in
        data space.
        The third entry controls the reduction of the residual.
    alpha0 : float, optional
    alpha_step : float, optional
        With these (alpha0, alpha_step) we compute the regulization parameter
        for the k-th Newton step by alpha0*alpha_step^k.

    Attributes
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    cgmaxit : int, optional
        Maximum number of CG iterations.
    alpha0 : float
    alpha_step : float
        Needed for the computation of the regulization parameter for the k-th
        Newton step.
    k : int
        Is the k-th Newton step.
    cgtol : list of float
        Contains three tolerances.
    x : array
        The current point.
    y : array
        The value at the current point.
    """

    def __init__(self, setting, data, init, cgmaxit=50, alpha0=1, alpha_step=2/3., 
                 cgtol=[0.3, 0.3, 1e-6]):
        """Initialization of parameters."""
        
        #super().__init__(logging.getLogger(__name__))
        super().__init__()
        self.setting=setting
        self.op = setting.op
        self.data = data
        self.init = init
        self.x = self.init
        
        # Parameter for the outer iteration (Newton method)
        self.k = 0
        
        # Parameters for the inner iteration (CG method)
        self.cgmaxit = cgmaxit
        self.alpha0 = alpha0
        self.alpha_step = alpha_step
        self.cgtol = cgtol

        # Update of the variables in the Newton iteration and preparation of
        # the first CG step.
        self.outer_update()

    def outer_update(self):
        """Updates and computes variables for the Newton iteration.
        
        In this function all variables of the current Newton iteration are 
        updated, after the CG method is used. Furthermore some variables for
        the next time the CG method is used (in the next Newton iteration) are
        prepared.
        """
        
        self.y, deriv = self.op.linearize(self.x)
        self._residual = self.data - self.y
        self._xref = self.init - self.x
        self.k += 1
        self._regpar = self.alpha0 * self.alpha_step**self.k
        self._cgstep = 0
        self._kappa = 1
        
        # Preparations for the CG method
        self._ztilde = self.setting.Hcodomain.gram(self._residual)
        self._stilde = (deriv.adjoint(self._ztilde) 
                        + self._regpar*self.setting.Hdomain.gram(self._xref))
        self._s = self.setting.Hdomain.gram_inv(self._stilde)
        self._d = self._s
        self._dtilde = self._stilde
        self._norm_s = np.real(self._stilde @ self._s)
        self._norm_s0 = self._norm_s
        self._norm_h = 0
        
        self._h = np.zeros(np.shape(self._s))
        self._Th = np.zeros(np.shape(self._residual))
        self._Thtilde = self._Th
        
    def inner_update(self):
        """Updates and computes variables for the CG iteration.
        
        In this function all variables in each CG iteration , after ``self.x``
        was updated, are updated. Its only purpose is to improve tidiness.
        """
        self._Th = self._Th + self._gamma*self._z
        self._Thtilde = self._Thtilde + self._gamma*self._ztilde
        _, deriv=self.op.linearize(self.x)
        self._stilde += (- self._gamma*(deriv.adjoint(self._ztilde) 
                         + self._regpar*self._dtilde)).real
        self._s = self.setting.Hdomain.gram_inv(self._stilde)
        self._norm_s_old = self._norm_s
        self._norm_s = np.real(self._stilde @ self._s)
        self._beta = self._norm_s / self._norm_s_old
        self._d = self._s + self._beta*self._d
        self._dtilde = self._stilde + self._beta*self._dtilde
        self._norm_h = self.setting.Hdomain.inner(self._h, self._h)
        self._kappa = 1 + self._beta*self._kappa
        self._cgstep += 1

    def _next(self):
        """Run a single IRGNM_CG iteration.

        The while loop is the CG method, it has four conditions to stop. The
        first three work with the tolerances given in ``self.cgtol``. The last
        condition checks if the maximum number of CG iterations 
        (``self.cgmaxit``) is reached.
        
        The CG method solves by CGNE
        

        .. math:: A h = b,
        
        with
        
        .. math:: A := G_X^{-1} F^{' *} G_Y F' + \mbox{regpar} ~I
        .. math:: b := G_X^{-1} F^{' *} G_Y y + \mbox{regpar}~ \mbox{xref}
        
        where
        
        +--------------------+-------------------------------------+ 
        | :math:`F`          | self.op                             | 
        +--------------------+-------------------------------------+ 
        | :math:`G_X,~ G_Y`  | self.op.domain.gram, self.op.codomain.gram | 
        +--------------------+-------------------------------------+
        | :math:`G_X^{-1}`   | self.op.domain.gram_inv               |
        +--------------------+-------------------------------------+                  
        | :math:`F'`         | self.op.derivative()                |
        +--------------------+-------------------------------------+ 
        | :math:`F'*`        | self.op.derivative().adjoint        | 
        +--------------------+-------------------------------------+


        Returns
        -------
        bool
            Always True, as the IRGNM_CG method never stops on its own.

        """
        while (
              # First condition
              np.sqrt(np.float64(self._norm_s)/self._norm_h/self._kappa)
              /self._regpar > self.cgtol[0] / (1+self.cgtol[0]) and
              # Second condition
              np.sqrt(np.float64(self._norm_s)
              /np.real(self.setting.Hdomain.inner(self._Thtilde,self._Th))
              /self._kappa/self._regpar)
              > self.cgtol[1] / (1+self.cgtol[1]) and
              # Third condition
              np.sqrt(np.float64(self._norm_s)/self._norm_s0/self._kappa) 
              > self.cgtol[2] and 
              # Fourth condition
              self._cgstep <= self.cgmaxit):

              
                  
            # Computations and updates of variables
            _, deriv=self.op.linearize(self.x)
            self._z = deriv(self._d)
            self._ztilde = self.setting.Hcodomain.gram(self._z)
            self._gamma = (self._norm_s
                           / np.real(self._regpar
                                     *self._dtilde @ self._d
                                     + self._ztilde @ self._z
                                     )
                           )
            self._h = self._h + self._gamma*self._d
#            print(self._gamma)
#            print(np.mean(self._norm_s)/np.mean(self._norm_s0))
            # Updating ``self.x`` 
 #           self.x += self._h
            
            self.inner_update()
        # End of the CG method. ``self.outer_update()`` does all computations
        # of the current Newton iteration.
        _, deriv=self.op.linearize(self.x)
        self.err=self._regpar*self._h+deriv.adjoint(deriv(self._h))-self._stilde
#        print(self._h)
        self.x+=self._h
        self.outer_update()
        return True

"""while (np.sqrt(np.float64(self._norm_s)/self._kappa/self._regpar**2)>self.cgtol[0] and
self._cgstep<=self.cgmaxit):"""
