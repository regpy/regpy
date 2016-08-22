import logging
import numpy as np

from . import Solver

__all__ = ['IRGNM_CG']


class IRGNM_CG(Solver):
    """The IRGNM_CG method.

    noch ausfüllen -------------------------

    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array_
        The initial guess.
    cgmaxit : int, optional
        Maximum number of CG iterations.
    alpha0, alpha_step : float, optional
        With these we compute the regulization parameter for the k-th Newton step
        by alpha0*alpha_step^k.
    cgtol : list of float, optional
        Contains three tolerances:
        The first entry controls the relative accuracy of the Newton update in preimage,
        the second entry controls the relative accuracy of the Newton update in data space,
        the third entry controls the reduction of the residual.
    

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
    """

    def __init__(self, op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 2/3., cgtol = [0.3, 0.3, 1e-6]):
        super().__init__(logging.getLogger(__name__))
        self.op = op
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

        # Initialization of the first step
        self.outer_update()
        

    def outer_update(self):
        """
        hier beschreibung einfuegen
        
        """
        
        self.y = self.op(self.x)
        self._residual = self.data - self.y
        self.xref = self.init - self.x
        self.k += 1
        self.regpar = self.alpha0 * self.alpha_step**self.k
        self.cgstep = 0
        self.kappa = 1
        self.ztilde = self.op.domy.gram(self._residual)
        self.stilde = self.op.adjoint(self.ztilde) + self.regpar * self.op.domx.gram(self.xref)
        self.s = self.op.domx.gram_inv(self.stilde)
        self.d = self.s
        self.dtilde = self.stilde
        self.norm_s = np.real(self.op.domx.inner(self.stilde, self.s))
        self.norm_s0 = self.norm_s
        self.norm_h = 0
        
        self.h = np.zeros(np.shape(self.s))
        self.Th = np.zeros(np.shape(self._residual))
        self.Thtilde = self.Th
        
        #prepare the parameters for the first inner iteration (CG method)
        self.z = self.op.derivative()(self.d)
        self.ztilde = self.op.domy.gram(self.z)
        self.gamma = self.norm_s / \
            np.real(self.regpar*self.op.domx.inner(self.dtilde,self.d)+self.op.domx.inner(self.ztilde,self.z))
        
        # prepare the stopping parameters for the inner iteration (CG method),
        # since we might divide by zero.
        
        # First condition
        if self.norm_h == 0 or self.kappa == 0 or self.regpar == 0:
            self.stop1 = self.cgtol[0]/(1 + self.cgtol[0]) + 1
        else:
            self.stop1 = np.sqrt(self.norm_s/self.norm_h/self.kappa)/self.regpar

        # Second condition
        if np.real(self.op.domx.inner(self.Thtilde,self.Th)) == 0 or self.kappa == 0 or self.regpar == 0:
            self.stop2 = self.cgtol[1]/(1 + self.cgtol[1]) + 1
        else:
            self.stop2 = np.sqrt(self.norm_s/np.real(self.op.domx.inner(self.Thtilde,self.Th))/self.kappa/self.regpar)

        # Third condition
        if self.norm_s0 == 0 or self.kappa == 0:
            self.stop3 = self.cgtol[2] + 1
        else:
            self.stop3 = np.sqrt(self.norm_s/self.norm_s0/self.kappa)

    def inner_update(self):
        self.Th = self.Th + self.gamma * self.z
        self.Thtilde = self.Thtilde + self.gamma * self.ztilde
        self.stilde += -self.gamma*(self.op.adjoint(self.ztilde) + self.regpar * self.dtilde)
        self.s = self.op.domx.gram_inv(self.stilde)
        self.norm_s_old = self.norm_s
        self.norm_s = np.real(self.op.domx.inner(self.stilde, self.s))
        self.beta = self.norm_s / self.norm_s_old
        self.d = self.s + self.beta * self.d
        self.dtilde = self.stilde + self.beta * self.dtilde
        self.norm_h = self.op.domx.inner(self.h, self.op.domx.gram(self.h))
        self.kappa = 1 + self.beta * self.kappa
        self.cgstep += 1
        
        self.z = self.op.derivative()(self.d)
        self.ztilde = self.op.domy.gram(self.z)
        self.gamma = self.norm_s / \
            np.real(self.regpar*self.op.domx.inner(self.dtilde,self.d)+self.op.domx.inner(self.ztilde,self.z))
            
        # prepare the stopping parameters for the inner iteration (CG method),
        # since we might divide by zero.
        
        # First condition
        if self.norm_h == 0 or self.kappa == 0 or self.regpar == 0:
            self.stop1 = self.cgtol[0]/(1 + self.cgtol[0]) + 1
        else:
            self.stop1 = np.sqrt(self.norm_s/self.norm_h/self.kappa)/self.regpar

        # Second condition
        if np.real(self.op.domx.inner(self.Thtilde,self.Th)) == 0 or self.kappa == 0 or self.regpar == 0:
            self.stop2 = self.cgtol[1]/(1 + self.cgtol[1]) + 1
        else:
            self.stop2 = np.sqrt(self.norm_s/np.real(self.op.domx.inner(self.Thtilde,self.Th))/self.kappa/self.regpar)

        # Third condition
        if self.norm_s0 == 0 or self.kappa == 0:
            self.stop3 = self.cgtol[2] + 1
        else:
            self.stop3 = np.sqrt(self.norm_s/self.norm_s0/self.kappa)
        
    def next(self):
        """Run a single IRGNM_CG iteration.

        Returns
        -------
        bool
            Always True, as the IRGNM_CG method never stops on its own.

        """
        while self.stop1 > self.cgtol[0]/(1 + self.cgtol[0]) and \
              self.stop2 > self.cgtol[1]/(1 + self.cgtol[1]) and \
              self.stop3 > self.cgtol[2] and self.cgstep <= self.cgmaxit:
            self.h = self.h + self.gamma * self.d
            self.x += self.h
            self.inner_update()        
        self.outer_update()        
        return True
