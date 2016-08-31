import logging
import numpy as np

from . import Solver

__all__ = ['IRGNM_CG']


class IRGNM_CG(Solver):
    """The IRGNM_CG method.

    Solves the potentially non-linear, ill-posed equation ::

        T(x) = y,

    where `T` is a Frechet-differentiable operator. The number of iterations is
    effectively the regularization parameter and needs to be picked carefully.
    

    The regularized Newton equations are solved by the conjugate gradient
    method applied to the normal equation.

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
        init : array_
        The initial guess.
    k : int
        Parameter for the outer iteration (Newton method)
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
        
        """
        solves A h = b by CGNE with
        A := G_X^{-1} F'* G_Y F' + regpar I
        b := G_X^{-1}F'^* G_Y y + regpar xref
        A is self-adjoint with respect to the inner product <u,v> = u'G_X v

        G_X, G_Y -> F.applyGramX, F.applyGramY
        G_X^{-1} -> F.applyGramX_inv
        F'       -> F.derivative
        F'*      -> F.adjoint
        
        ########SO WIRD CGNE_reg IN MATLAB BESCHRIEBEN!!!! UEBERNEHMEN?!?!?
        
        """      
        
        
        
    def outer_update(self):
        self.y = self.op(self.x)
        self._residual = self.data - self.y
        self._xref = self.init - self.x
        self.k += 1
        self._regpar = self.alpha0 * self.alpha_step**self.k
        self._cgstep = 0
        self._kappa = 1
        self._ztilde = self.op.domy.gram(self._residual)
        self._stilde = self.op.adjoint(self._ztilde) + self._regpar * self.op.domx.gram(self._xref)
        self._s = self.op.domx.gram_inv(self._stilde)
        self._d = self._s
        self._dtilde = self._stilde
        self._norm_s = np.real(self.op.domx.inner(self._stilde, self._s))
        self._norm_s0 = self._norm_s
        self._norm_h = 0
        
        self._h = np.zeros(np.shape(self._s))
        self._Th = np.zeros(np.shape(self._residual))
        self._Thtilde = self._Th
        
        #prepare the parameters for the first inner iteration (CG method)
        self._z = self.op.derivative()(self._d)
        self._ztilde = self.op.domy.gram(self._z)
        self._gamma = self._norm_s / \
            np.real(self._regpar*self.op.domx.inner(self._dtilde,self._d)+self.op.domx.inner(self._ztilde,self._z))
        
    def inner_update(self):
        self._Th = self._Th + self._gamma * self._z
        self._Thtilde = self._Thtilde + self._gamma * self._ztilde
        self._stilde += -self._gamma*(self.op.adjoint(self._ztilde) + self._regpar * self._dtilde)
        self._s = self.op.domx.gram_inv(self._stilde)
        self._norm_s_old = self._norm_s
        self._norm_s = np.real(self.op.domx.inner(self._stilde, self._s))
        self._beta = self._norm_s / self._norm_s_old
        self._d = self._s + self._beta * self._d
        self._dtilde = self._stilde + self._beta * self._dtilde
        self._norm_h = self.op.domx.inner(self._h, self.op.domx.gram(self._h))
        self._kappa = 1 + self._beta * self._kappa
        self._cgstep += 1
        
        self._z = self.op.derivative()(self._d)
        self._ztilde = self.op.domy.gram(self._z)
        self._gamma = self._norm_s / \
            np.real(self._regpar*self.op.domx.inner(self._dtilde,self._d)+self.op.domx.inner(self._ztilde,self._z))
        
    def next(self):
        """Run a single IRGNM_CG iteration.

        Returns
        -------
        bool
            Always True, as the IRGNM_CG method never stops on its own.

        """
        while np.sqrt(np.float64(self._norm_s)/self._norm_h/self._kappa)/self._regpar > self.cgtol[0]/(1 + self.cgtol[0]) and \
              np.sqrt(np.float64(self._norm_s)/np.real(self.op.domx.inner(self._Thtilde,self._Th))/self._kappa/self._regpar) > self.cgtol[1]/(1 + self.cgtol[1]) and \
              np.sqrt(np.float64(self._norm_s)/self._norm_s0/self._kappa) > self.cgtol[2] and self._cgstep <= self.cgmaxit:
            self._h = self._h + self._gamma * self._d
            self.x += self._h
            self.inner_update()        
        self.outer_update()        
        return True
