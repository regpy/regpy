import logging
import numpy as np
import scipy

from . import Solver

__all__ = ['IRGNM_L1_fid']


class IRGNM_L1_fid(Solver):
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

    def __init__(self, op, data, init, alpha0 = 1, alpha_step = 0.9, alpha_l1 = 1e-4):
        super().__init__(logging.getLogger(__name__))
        self.op = op
        self.data = data
        self.init = init
        self.x = self.init
        self.y = self.op(self.x)
        
        # Parameter for the outer iteration (Newton method)
        self.k = 0   
        self.alpha0 = alpha0
        self.alpha_step = alpha_step
        self.alpha_l1 = alpha_l1

        self.GramX = self.op.domx.gram(np.eye(len(self.x)))
        self.GramX = 0.5 * (self.GramX + self.GramX.T)
        self.GramY = self.op.domy.gram(np.eye(len(self.y)))
        self.GramY = 0.5 * (self.GramY + self.GramY.T)     
        self.maxiter = 10000
        self.x = self.init
        
    def outer_update(self):
        """
        hier beschreibung einfuegen
        
        """
        self.k += 1
        self.regpar = self.alpha0 * self.alpha_step ** self.k
        self.DF = np.eye(len(self.y),len(self.x))
        for i in range(len(self.x)):
            self.DF[:,i] = self.op.derivative()(self.DF[:,i])
        self.Hess = np.dot(self.DF,np.linalg.solve(self.GramX,self.DF.T)) + self.regpar * np.linalg.inv(self.GramY)
        self.Hess = 0.5 * (self.Hess + self.Hess.T)
        self.rhs = self.data - self.y - np.dot(self.DF,self.init - self.x)
        self.alpha_l1 = np.max(self.regpar,self.alpha_l1)
        self.iter = 100
        self.exitflag = True
        while self.iter < self.maxiter and self.exitflag:
            self.bounds = (-(1/self.alpha_l1)*np.ones(len(self.y)), (1/self.alpha_l1)*np.ones(len(self.y)))
            self.options = {'maxiter':self.iter, 'disp':False}
            self.quadprogsol = scipy.optimize.minimize(self.func, bounds = np.transpose(self.bounds).tolist(),options=self.options, x0 = np.zeros(len(self.x)))
            self.updateY = self.quadprogsol.x
            #noch nicht 1 zu 1 uebersetzt,
            self.exitflag = self.quadprogsol.success
            self.iter *= 2
        
    def func(self, x):
        return 0.5 * np.dot(x.T,np.dot(self.Hess,x)) - np.dot(self.rhs.T,x)            
            
    def next(self):
        """Run a single IRGNM_CG iteration.

        Returns
        -------
        bool
            Always True, as the IRGNM_CG method never stops on its own.

        """
        self.outer_update()
        self.x = self.init + self.op.domx.gram_inv(np.dot(self.DF.T,self.updateY))
        self.y = self.op(self.x)
        return True
