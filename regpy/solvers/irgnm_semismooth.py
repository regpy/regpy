from . import Solver

import logging
import numpy as np
import scipy.sparse.linalg as spla
import scipy.optimize as sco

class IRGNMSemiSmooth(Solver):
    """
    Semismooth Newton Method. In each iteration, solves

     x_{n+1} \in argmin_{psi_minus < x_* < psi_plus}   ||T(x_n) + T'[x_n] (x_*-x_n) - data||**2 + regpar_n * ||x_* - init||**2

    where `T` is a Frechet-differentiable operator, using `regpy.solvers.tikhonov.TikhonovCG`.
    `regpar_n` is a decreasing geometric sequence of regularization parameters.
    """

    def __init__(self, setting, data, psi_minus, psi_plus, regpar, regpar_step=2 / 3, init=None, cgpars=None):
        super().__init__()
        self.setting=setting
        """The problem setting"""
        self.data=data
        """The measured data"""
        if init is None:
            init = self.setting.op.domain.zeros()
        self.init = np.asarray(init)
        """The initial guess."""
        self.x=np.copy(self.init)
        self.regpar=regpar
        """The regularizaton parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        if cgpars is None:
            cgpars = {}
        self.cgpars = cgpars
        """The additional `regpy.solvers.tikhonov.TikhonovCG` parameters."""
        self.psi_minus=psi_minus
        self.psi_plus=psi_plus
        """The upper and the lower bound"""
        self.size=self.init.shape[0]

        """Prepare first iteration step"""
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.rhs=self.data-self.y+self.deriv(self.x)
        self.b=self.setting.Hdomain.gram_inv(self.deriv.adjoint(self.setting.Hcodomain.gram(self.rhs)))+self.regpar*self.init
        
        """Prepare newton-semismooth minimization"""
        self.lam_plus=np.maximum(np.zeros(self.size), self.b-self._A(self.x))
        self.lam_minus=-np.minimum(np.zeros(self.size), self.b-self._A(self.x))

        """sets where the upper constraint and the lower constarint are active"""
        self.active_plus=[self.lam_plus[j]+self.regpar*(self.x[j]-self.psi_plus)>0 for j in range(self.size)]
        self.active_minus=[self.lam_minus[j]-self.regpar*(self.x[j]-self.psi_minus)>0 for j in range(self.size)]
        
        """compute active and inactive sets, need to be computed in each step again"""
        self.active=np.zeros(self.size)
        self.inactive=np.zeros(self.size)
        
    def _next(self):
        for i in range(10):
            self.inner_update()
        
        self.y, self.deriv = self.setting.op.linearize(self.x)
        
        self.rhs=self.data-self.y+self.deriv(self.x)
        self.b=self.setting.Hdomain.gram_inv(self.deriv.adjoint(self.setting.Hcodomain.gram(self.rhs)))+self.regpar*self.init

        #Prepare newton-semismooth minimization
        self.lam_plus=np.maximum(np.zeros(self.size), self.b-self._A(self.x))
        self.lam_minus=-np.minimum(np.zeros(self.size), self.b-self._A(self.x))

        #sets where the upper constraint and the lower constarint are active
        self.active_plus=[self.lam_plus[j]+self.regpar*(self.x[j]-self.psi_plus)>0 for j in range(self.size)]
        self.active_minus=[self.lam_minus[j]-self.regpar*(self.x[j]-self.psi_minus)>0 for j in range(self.size)]

        self.regpar *= self.regpar_step
        
        
    def inner_update(self):
        self.active=[self.active_plus[j] or self.active_minus[j] for j in range(self.size)]
        self.inactive=[self.active[j]==False for j in range(self.size)]

        #On the active sets the solution takes the values of the constraints
        self.x[self.active_plus]=self.psi_plus
        self.x[self.active_minus]=self.psi_minus

        self.lam_plus[self.inactive]=0
        self.lam_plus[self.active_minus]=0
        self.lam_minus[self.inactive]=0
        self.lam_minus[self.active_plus]=0

        #A as spla.LinearOperator constrained to inactive set
        A_inactive=spla.LinearOperator(
                (np.count_nonzero(self.inactive), np.count_nonzero(self.inactive)),
                matvec=self._A_inactive,
                dtype=float)
        #Solve system on the different sets
        self.x[self.inactive]=spla.gmres(A_inactive, self.b[self.inactive], maxiter=15)[0]
        z=self._A(self.x)
        """
        self.log.info('Running Tikhonov solver.')
        f, _ = TikhonovCG(
            setting=HilbertSpaceSetting(self.deriv @ projection, self.setting.Hdomain, self.setting.Hcodomain),
            data=self.rhs, 
            regpar=self.regpar,
            xref=self.init,
            **self.cgpars
        ).run()
        """
        self.lam_plus[self.active_plus]=self.b[self.active_plus]+self.lam_minus[self.active_plus]-z[self.active_plus]
        self.lam_minus[self.active_minus]=-self.b[self.active_minus]+self.lam_plus[self.active_minus]+z[self.active_minus]

        #Update active and inactive sets
        self.active_plus=[self.lam_plus[j]+self.regpar*(self.x[j]-self.psi_plus)>0 for j in range(self.size)]
        self.active_minus=[self.lam_minus[j]-self.regpar*(self.x[j]-self.psi_minus)>0 for j in range(self.size)]
        
    def _A(self, u):
        return self.regpar*u+self.setting.Hdomain.gram_inv(self.deriv.adjoint(self.setting.Hcodomain.gram(self.deriv(u))))

    def _A_inactive(self, u):
        projection=np.zeros(self.size)
        projection[self.active_plus]=self.psi_plus
        projection[self.active_minus]=self.psi_minus
        projection[self.inactive]=u
        return self._A(projection)[self.inactive]