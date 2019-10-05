from itreg.solvers import Solver

import logging
import numpy as np
import scipy.sparse.linalg as spla

class NewtonSemiSmooth(Solver):
    def __init__(self, setting, rhs, init, alpha, psi_minus, psi_plus):
        super().__init__()
        self.setting=setting
        self.rhs=rhs
        self.x=init
        self.alpha=alpha
        #constraints
        self.psi_minus=psi_minus
        self.psi_plus=psi_plus

        self.size=init.shape[0]

        self.y = self.setting.op(self.x)
        #A=Id+T*T

        self.b=self.setting.op.adjoint(self.rhs)+self.alpha*init

        self.lam_plus=np.maximum(np.zeros(self.size), self.b-self._A(self.x))
        self.lam_minus=-np.minimum(np.zeros(self.size), self.b-self._A(self.x))

        #sets where the upper constraint and the lower constarint are active
        self.active_plus=[self.lam_plus[j]+self.alpha*(self.x[j]-self.psi_plus)>0 for j in range(self.size)]
        self.active_minus=[self.lam_minus[j]-self.alpha*(self.x[j]-self.psi_minus)>0 for j in range(self.size)]

        #complte active and inactive sets, need to be computed in each step again
        self.active=np.empty(self.size)
        self.inactive=np.empty(self.size)

    def _next(self):
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
        self.x[self.inactive]=self._gmres(A_inactive, self.b[self.inactive]+self.lam_minus[self.inactive]-self.lam_plus[self.inactive])
        z=self._A(self.x)
        self.lam_plus[self.active_plus]=self.b[self.active_plus]+self.lam_minus[self.active_plus]-z[self.active_plus]
        self.lam_minus[self.active_minus]=-self.b[self.active_minus]+self.lam_plus[self.active_minus]+z[self.active_minus]

        #Update active and inactive sets
        self.y=self.setting.op(self.x)
        self.active_plus=[self.lam_plus[j]+self.alpha*(self.x[j]-self.psi_plus)>0 for j in range(self.size)]
        self.active_minus=[self.lam_minus[j]-self.alpha*(self.x[j]-self.psi_minus)>0 for j in range(self.size)]

    def _gmres(self, op, rhs):
        result, info = spla.gmres(op, rhs.ravel())
        if info > 0:
            self.log.warn('Gmres failed to converge')
        elif info < 0:
            self.log.warn('Illegal Gmres input or breakdown')
        return result

    def _A(self, u):
        self.y = self.setting.op(u)
        return self.alpha*u+self.setting.op.adjoint(self.y)

    def _A_inactive(self, u):
        projection=np.zeros(self.size)
        projection[self.inactive]=u
        return self._A(projection)[self.inactive]
