import logging
import numpy as np
import scipy.optimize

from . import Solver

__all__ = ['IRGNM_CG']


class IRGNM_CG_preconditioned(Solver):
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

    def __init__(self, op, data, init, cgmaxit=50, alpha0=1, alpha_step=2/3., 
                 cgtol=[0.3, 0.3, 1e-6]):
        """Initialization of parameters."""
        
        #super().__init__(logging.getLogger(__name__))
        super().__init__()
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
        
        self.eigval_num=3
        self.orthonormal=np.zeros((self.eigval_num, self.data.shape[0]))
        self.need_prec_update=True

        # Update of the variables in the Newton iteration and preparation of
        # the first CG step.
        self.outer_update()
        self.lanzcos_update()
        self.outer_update_precond()

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
        self._ztilde = self.op.range.gram(self._residual)
        self._stilde = (deriv.adjoint(self._ztilde) 
                        + self._regpar*self.op.domain.gram(self._xref))
        self._s = self.op.domain.gram_inv(self._stilde)
        self._d = self._s
        self._dtilde = self._stilde
        self._norm_s = np.real(self.op.domain.inner(self._stilde, self._s))
        self._norm_s0 = self._norm_s
        self._norm_h = 0
        
        self._h = np.zeros(np.shape(self._s))
        self._Th = np.zeros(np.shape(self._residual))
        self._Thtilde = self._Th
        self.inner_num=0
        
    def outer_update_precond(self):
        
        """Updates and computes variables for the Newton iteration.
        
        In this function all variables of the current Newton iteration are 
        updated, after the CG method is used. Furthermore some variables for
        the next time the CG method is used (in the next Newton iteration) are
        prepared.
        """
    
        self.y, deriv = self.op.linearize(self.x)
        self._residual_precond = self.data - self.y
        self._xref = self.init - self.x
        self.k += 1
        self._regpar = self.alpha0 * self.alpha_step**self.k
        self._cgstep = 0
        self._kappa_precond = 1
        
        # Preparations for the CG method
        self._ztilde_precond = self.op.range.gram(self._residual_precond)
        self._stilde_precond = np.dot(self.M.transpose(),(deriv.adjoint(self._ztilde_precond) 
                        + self._regpar*self.op.domain.gram(self._xref)))
        self._s_precond = self.op.domain.gram_inv(self._stilde_precond)
        self._d_precond = self._s_precond
        self._dtilde_precond = self._stilde_precond
        self._norm_s_precond = np.real(self.op.domain.inner(self._stilde_precond, self._s_precond))
        self._norm_s0_precond = self._norm_s
        self._norm_h_precond = 0
        
        self._h_precond = np.zeros(np.shape(self._s_precond))
        self._Th_precond = np.zeros(np.shape(self._residual_precond))
        self._Thtilde_precond = self._Th_precond
            
        
    def inner_update(self):
        """Updates and computes variables for the CG iteration.
        
        In this function all variables in each CG iteration , after ``self.x``
        was updated, are updated. Its only purpose is to improve tidiness.
        """
        self._Th = self._Th + self._gamma*self._z
        self._Thtilde = self._Thtilde + self._gamma*self._ztilde
        _, deriv=self.op.linearize(self.x)
        self._stilde += (- self._gamma*(deriv(self._ztilde) 
                         + self._regpar*self._dtilde)).real
        self._s = self.op.domain.gram_inv(self._stilde)
        self._norm_s_old = self._norm_s
        self._norm_s = np.real(self.op.domain.inner(self._stilde, self._s))
        self._beta = self._norm_s / self._norm_s_old
        self._d = self._s + self._beta*self._d
        self._dtilde = self._stilde + self._beta*self._dtilde
        self._norm_h = self.op.domain.inner(self._h, self.op.domain.gram(self._h))
        self._kappa = 1 + self._beta*self._kappa
        self._cgstep += 1
        self.inner_num+=1
        
        
    def inner_update_precond(self):
        """Inner update for the preconditioned problem"""
        self._Th_precond = self._Th_precond + self._gamma_precond*self._z_precond
        self._Thtilde_precond = self._Thtilde_precond + self._gamma_precond*self._ztilde_precond
        _, deriv=self.op.linearize(self.x)
        self._stilde_precond += (- self._gamma_precond*np.dot(self.M.transpose(), (deriv(self._ztilde_precond) 
                         + self._regpar*self._dtilde_precond)).real)
        self._s_precond = self.op.domain.gram_inv(self._stilde_precond)
        self._norm_s_old_precond = self._norm_s_precond
        self._norm_s_precond = np.real(self.op.domain.inner(self._stilde_precond, self._s_precond))
        self._beta_precond = self._norm_s_precond / self._norm_s_old_precond
        self._d_precond = self._s_precond + self._beta_precond*self._d_precond
        self._dtilde_precond = self._stilde_precond + self._beta_precond*self._dtilde_precond
        self._norm_h_precond = self.op.domain.inner(self._h_precond, self.op.domain.gram(self._h_precond))
        self._kappa_precond = 1 + self._beta_precond*self._kappa_precond
        self._cgstep += 1
        
        

    def next(self):
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
        | :math:`G_X,~ G_Y`  | self.op.domain.gram, self.op.range.gram | 
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
        
            
        if self.need_prec_update is True:
            while (
              # First condition
              np.sqrt(np.float64(self._norm_s)/self._norm_h/self._kappa)
              /self._regpar > self.cgtol[0] / (1+self.cgtol[0]) and
              # Second condition
              np.sqrt(np.float64(self._norm_s)
              /np.real(self.op.domain.inner(self._Thtilde,self._Th))
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
                self._ztilde = self.op.range.gram(self._z)
                self._gamma = (self._norm_s
                               / np.real(self._regpar
                                         *self.op.domain.inner(self._dtilde,self._d)
                                         + self.op.domain.inner(self._ztilde,self._z)
                                         )
                               )
                self._h = self._h + self._gamma*self._d
    #            print(np.mean(self._norm_s)/np.mean(self._norm_s0))
                # Updating ``self.x`` 
     #           self.x += self._h
                
                self.inner_update()
                if self.inner_num<=self.eigval_num:
                    self.orthonormal[self.inner_num-1, :]=self._s/self._norm_s
                self.lanzcos_update()
            
        # End of the CG method. ``self.outer_update()`` does all computations
        # of the current Newton iteration.
        
        while (
              # First condition
              np.sqrt(np.float64(self._norm_s_precond)/self._norm_h_precond/self._kappa)
              /self._regpar > self.cgtol[0] / (1+self.cgtol[0]) and
              # Second condition
              np.sqrt(np.float64(self._norm_s_precond)
              /np.real(self.op.domain.inner(self._Thtilde_precond,self._Th_precond))
              /self._kappa/self._regpar)
              > self.cgtol[1] / (1+self.cgtol[1]) and
              # Third condition
              np.sqrt(np.float64(self._norm_s_precond)/self._norm_s0_precond/self._kappa) 
              > self.cgtol[2] and 
              # Fourth condition
              self._cgstep <= self.cgmaxit):
                  
            # Computations and updates of variables
            
            _, deriv=self.op.linearize(self.x)
            self._z_precond = deriv(self._d_precond)
            self._ztilde_precond = self.op.range.gram(self._z_precond)
            self._gamma_precond = (self._norm_s_precond
                           / np.real(self._regpar
                                     *self.op.domain.inner(self._dtilde_precond,self._d_precond)
                                     + self.op.domain.inner(self._ztilde_precond,self._z_precond)
                                     )
                           )
            self._h_precond = self._h_precond + self._gamma_precond*self._d_precond
#            print(np.mean(self._norm_s)/np.mean(self._norm_s0))
            # Updating ``self.x`` 
 #           self.x += self._h
            
            self.inner_update_precond()
            
#        self._h=scipy.optimize.minimize(self.M+self._regpar*np.identity(self.op.domain.shape[0]), self._h_precond)
        self._h=np.dot(np.linalg.inv(self.M), self._h_precond)
        
        
        
        self.x+=self._h
        self.outer_update()
        self.need_prec_update=False
        if int(np.sqrt(self.k))**2==self.k:
            self.need_prec_update=True
        return True
    
    
    
    
    
    
    
    
    def lanzcos_update(self):
        """perform lanzcos method to calculate the preconditioner"""
        self.deriv_mat=np.zeros((self.op.domain.shape[0], self.op.domain.shape[0]))
        self.L=np.zeros((self.eigval_num, self.eigval_num))
        _, self.deriv=self.op.linearize(self.x)
        for i in range(0, self.eigval_num):
            self.L[i, :]=np.dot(self.orthonormal, self.op.domain.gram_inv(self.deriv.adjoint(self.op.range.gram(self.deriv((self.orthonormal[i, :]))))))
            self.deriv_mat[i, :]=self.op.domain.gram_inv(self.deriv.adjoint(self.op.domain.gram(self.deriv(self.x))))
        self.lamb, self.U=np.linalg.eig(self.L)
        self.lanczos=np.dot(self.orthonormal.transpose(), self.U)
#        self.M=self._alpha*np.identity(self.data.shape[0])
#        self.M=np.zeros((self.data.shape[0], self.data.shape[0]))
        self.M=self._regpar*np.identity(self.data.shape[0])
        for  i in range(0, self.eigval_num):
            self.M[i, :]+=self.lanczos[:, i]*self.lamb[i]
        self.pre_cond_deriv=np.dot(self.M.transpose(), np.dot(self.deriv_mat+self._regpar*np.identity(self.op.domain.shape[0]), self.M))
        self.C=np.dot(self.pre_cond_deriv, self.pre_cond_deriv.transpose())
    
        
    