import logging

from regpy.solvers import RegularizationSetting, RegSolver
from regpy.solvers.linear.tikhonov import TikhonovCG
from regpy.operators import CoordinateMask

class IrgnmSemiSmooth(RegSolver):
    r"""
    Semismooth Newton Method. In each iteration, solves
    \[
     x_{n+1} \in \textrm{argmin}_{\psi_- < x_\ast < \psi_+}   ||T(x_n) + T'[x_n] (x_\ast-x_n) - g_\text{data}||^2 + \alpha_n  ||x_\ast - x_\text{init}||^2
    \]
    where \(T\) is a Frechet-differentiable operator, using `regpy.solvers.linear.tikhonov.TikhonovCG`.
    \(\alpha_n\) is a decreasing geometric sequence of regularization parameters.
    
    Parameters
    ----------
    setting : RegularizationSetting
        Setting for regularization. 
    data : array-like
        Data for reconstruction. Must be in the operators codomain.
    psi_minus : np.number
        lower constraint of the minimization. Must be larger then `psi_plus`
    psi_plus : np.number
        upper constraint of the minimization. Must be smaller then `psi_minus`
    regpar : np.number
        Initial regularization parameter \(\alpha\) 
    regpar_step : np.number, optional
        Must be between 0 and 1. Multiplied to regularization parameter to construct the decreasing geometric sequence. (Default: 2/3)
    init : array-like, optional
        An element of operator domain that is an initial guess. (Default: None)
    cg_pars : dict
        Dictionary of parameter to be given to the inner `TikhonovCG` solver. (Default: None) 
    """
    def __init__(self, setting, data, psi_minus, psi_plus, regpar, regpar_step=2 / 3, init=None, cg_pars=None):
        assert isinstance(setting,RegularizationSetting)
        assert psi_minus < psi_plus
        super().__init__(setting)
        self.data=data
        """The measured data"""
        if init is None:
            init = self.op.domain.zeros()
        self.init = init.copy()
        """The initial guess."""
        self.x=self.init.copy()
        self.regpar=regpar
        """The regularizaton parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        if cg_pars is None:
            cg_pars = {}
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        self.psi_minus=psi_minus
        self.psi_plus=psi_plus
        """The upper and the lower bound"""
        self.size=self.init.shape[0]

        """Prepare first iteration step"""
        self.y, self.deriv = self.op.linearize(self.x)
        self.rhs=self.data-self.y+self.deriv(self.x)
        self.b=self.h_domain.gram_inv(self.deriv.adjoint(self.h_codomain.gram(self.rhs)))+self.regpar*self.init
        
        self.lam_plus = self.op.domain.zeros()
        self.lam_minus = self.op.domain.zeros()
        """Prepare newton-semismooth minimization"""
        z = self.b-self._A(self.x)
        self.lam_plus[z>0] = z[z>0]
        self.lam_minus[z<0] = -z[z<0]

        """sets where the upper constraint and the lower constarint are active"""
        self.active_plus=(self.lam_plus+self.regpar*(self.x-self.psi_plus))>0
        self.active_minus=(self.lam_minus-self.regpar*(self.x-self.psi_minus))>0

        self.active_plus_old=self.active_plus
        self.active_minus_old=self.active_minus
        
        """compute active and inactive sets, need to be computed in each step again"""
        self.active=self.op.domain.zeros()
        self.inactive=self.op.domain.zeros()
        
    def _next(self):
        iter_count = 0
        while iter_count<=20 and (iter_count==0 or (self.active_plus_old != self.active_plus).sum()>3 or (self.active_minus_old != self.active_minus).sum()>3):
            self.active_plus_old=self.active_plus
            self.active_minus_old=self.active_minus
            self.inner_update()
            iter_count += 1
        
        self.y, self.deriv = self.op.linearize(self.x)
        
        self.rhs=self.data-self.y+self.deriv(self.x)
        self.b=self.h_domain.gram_inv(self.deriv.adjoint(self.h_codomain.gram(self.rhs)))+self.regpar*self.init

        self.lam_plus = self.op.domain.zeros()
        self.lam_minus = self.op.domain.zeros()
        #Prepare newton-semismooth minimization
        z = self.b-self._A(self.x)
        self.lam_plus[z>0] = z[z>0]
        self.lam_minus[z<0] = -z[z<0]

        #sets where the upper constraint and the lower constarint are active
        self.active_plus=(self.lam_plus+self.regpar*(self.x-self.psi_plus))>0
        self.active_minus=(self.lam_minus-self.regpar*(self.x-self.psi_minus))>0

        self.active_plus_old=self.active_plus
        self.active_minus_old=self.active_minus

        self.regpar *= self.regpar_step
        
        
    def inner_update(self):
        self.active=self.op.domain.vec_type.logical_or(self.active_plus, self.active_minus)
        self.inactive=self.op.domain.vec_type.logical_not(self.active)

        #On the active sets the solution takes the values of the constraints
        self.x[self.active_plus]=self.psi_plus[self.active_plus]
        self.x[self.active_minus]=self.psi_minus[self.active_minus]

        self.lam_plus[self.inactive]=0
        self.lam_plus[self.active_minus]=0
        self.lam_minus[self.inactive]=0
        self.lam_minus[self.active_plus]=0

        project = CoordinateMask(self.h_domain.vecsp, self.inactive)
        self.log.info('Running inner Tikhonov solver.')
        f, _ = TikhonovCG(
            setting=RegularizationSetting(self.deriv * project, self.h_domain, self.h_codomain),
            data=self.rhs, 
            regpar=self.regpar,
            xref=self.init,
            logging_level="WARNING",
            **self.cg_pars
        ).run()
        self.x[self.inactive] = f[self.inactive]
        z = self._A(self.x)
        
        self.lam_plus[self.active_plus]=self.b[self.active_plus]+self.lam_minus[self.active_plus]-z[self.active_plus]
        self.lam_minus[self.active_minus]=-self.b[self.active_minus]+self.lam_plus[self.active_minus]+z[self.active_minus]

        #Update active and inactive sets
        self.active_plus=(self.lam_plus+self.regpar*(self.x-self.psi_plus))>0
        self.active_minus=(self.lam_minus-self.regpar*(self.x-self.psi_minus))>0
        
    def _A(self, u):
        return self.regpar*u+self.h_domain.gram_inv(self.deriv.adjoint(self.h_codomain.gram(self.deriv(u))))