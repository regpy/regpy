from regpy.operators import CoordinateMask
from regpy.util import Errors

from ..general import Setting, RegSolver
from ..linear.tikhonov import TikhonovCG

__all__ = ["IrgnmSemiSmooth"]

class IrgnmSemiSmooth(RegSolver):
    r"""
    Semismooth Newton Method. In each iteration, solves
    
    .. math::
        x_{n+1} \in \textrm{argmin}_{\psi_- < x_\ast < \psi_+}   ||T(x_n) + T'[x_n] (x_\ast-x_n) - g_\text{data}||^2 + \alpha_n  ||x_\ast - x_\text{init}||^2

    where :math:`T` is a Frechet-differentiable operator, using `regpy.solvers.linear.tikhonov.TikhonovCG`.
    :math:`\alpha_n` is a decreasing geometric sequence of regularization parameters.
    
    Parameters
    ----------
    setting : regpy.solvers.Setting
        Setting for regularization. 
    psi_minus : np.number
        lower constraint of the minimization. Must be larger then `psi_plus`
    psi_plus : np.number
        upper constraint of the minimization. Must be smaller then `psi_minus`
    data : array-like, default None
        The measured data. Must be in the operators codomain. If it is None it is taken from the setting.
    regpar : float, default None
        Initial regularization parameter :math:`\alpha`. Must be positive. If it is None it is taken from the setting.
    regpar_step : np.number, optional
        Must be between 0 and 1. Multiplied to regularization parameter to construct the decreasing geometric sequence. (Default: 2/3)
    init : array-like, optional
        An element of operator domain that is an initial guess. (Default: None)
    inner_it_count : int, optional
        Number of inner iterations for the `TikhonovCG` solver. (Default: 20)
    inner_active_change : int, optional
        NUmber of changes in active sets mask to continue the inner iterations. (Default: 3)
    cg_pars : dict
        Dictionary of parameter to be given to the inner `TikhonovCG` solver. (Default: None) 
    """
    def __init__(self, setting, psi_minus, psi_plus,data=None, regpar=None, regpar_step=2 / 3, init=None, inner_it_count = 20, inner_active_change = 3, cg_pars=None):
        super().__init__(setting)
        if (psi_minus >= psi_plus):
            raise ValueError(Errors.value_error("The upper constraint is less or equal the lower constraint in IrgnmSemiSmooth. Given: "+"\n\t "+f"psi_minus = {psi_minus} "+"\t\n "+f"psi_plus = {psi_plus}"))
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        if(regpar is None):
            if(not setting.is_tikhonov):
                raise ValueError(Errors.value_error("Regularization parameter has to be included in setting or given directly."))
            regpar=setting.regpar
        self.data=data
        """The measured data"""
        if init is None:
            init = self.op.domain.zeros()
        self.init = init.copy()
        """The initial guess."""
        self.x=self.init.copy()
        self.regpar=regpar
        """The regularization parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        if cg_pars is None:
            cg_pars = {"logging_level": "WARNING"}
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        self.psi_minus=psi_minus*self.op.domain.ones()
        self.psi_plus=psi_plus*self.op.domain.ones()
        """The upper and the lower bound"""
        self.inner_it_count = inner_it_count
        self.inner_active_change = inner_active_change

        """Prepare first iteration step"""
        self.y, self.deriv = self.op.linearize(self.x)
        self.rhs=self.data-self.y+self.deriv(self.x)
        self.b=self.h_domain.gram_inv(self.deriv.adjoint(self.h_codomain.gram(self.rhs)))+self.regpar*self.init
        
        self.lam_plus = self.op.domain.zeros()
        self.lam_minus = self.op.domain.zeros()
        """Prepare newton-semismooth minimization"""
        z = self.b-self._A(self.x)
        pos_mask = self.op.domain.IfPos(z)
        self.lam_plus[pos_mask] = z[pos_mask]
        self.lam_minus[~pos_mask] = -z[~pos_mask]

        """sets where the upper constraint and the lower constraint are active"""
        self.active_plus=self.op.domain.IfPos(self.lam_plus+self.regpar*(self.x-self.psi_plus))
        self.active_minus=self.op.domain.IfPos(self.lam_minus-self.regpar*(self.x-self.psi_minus))

        self.active_plus_old=self.active_plus
        self.active_minus_old=self.active_minus
        
        """compute active and inactive sets, need to be computed in each step again"""
        self.inactive=self.op.domain.IfPos(self.op.domain.zeros())
        
    def _next(self):
        iter_count = 0
        while iter_count<self.inner_it_count and (iter_count==0 or sum(self.active_plus_old & self.active_plus)>self.inner_active_change or sum(self.active_minus_old & self.active_minus)>self.inner_active_change):
            self.log.info(f'Running inner iteration {iter_count+1} of {self.inner_it_count}.')
            self.active_plus_old=self.active_plus
            self.active_minus_old=self.active_minus
            self.inner_update()
            self.log.debug(f"Active plus Compare: {sum(self.active_plus_old & self.active_plus)}, Active minus Compare: {sum(self.active_minus_old & self.active_minus)}")
            iter_count += 1

        
        self.y, self.deriv = self.op.linearize(self.x)
        
        self.rhs=self.data-self.y+self.deriv(self.x)
        self.b=self.h_domain.gram_inv(self.deriv.adjoint(self.h_codomain.gram(self.rhs)))+self.regpar*self.init

        self.lam_plus = self.op.domain.zeros()
        self.lam_minus = self.op.domain.zeros()
        #Prepare newton-semismooth minimization
        z = self.b-self._A(self.x)
        pos_mask = self.op.domain.IfPos(z)
        self.lam_plus[pos_mask] = z[pos_mask]
        self.lam_minus[~pos_mask] = -z[~pos_mask]

        #sets where the upper constraint and the lower constraint are active
        self.active_plus=self.op.domain.IfPos(self.lam_plus+self.regpar*(self.x-self.psi_plus))
        self.active_minus=self.op.domain.IfPos(self.lam_minus-self.regpar*(self.x-self.psi_minus))

        self.active_plus_old=self.active_plus
        self.active_minus_old=self.active_minus

        self.regpar *= self.regpar_step
        
        
    def inner_update(self):
        self.inactive= ~(self.active_plus | self.active_minus)
        #On the active sets the solution takes the values of the constraints
        self.x[self.active_plus]=self.psi_plus[self.active_plus]
        self.x[self.active_minus]=self.psi_minus[self.active_minus]

        self.lam_plus[self.inactive]=0
        self.lam_plus[self.active_minus]=0
        self.lam_minus[self.inactive]=0
        self.lam_minus[self.active_plus]=0

        self.log.debug(f"lam_plus: {self.lam_plus}")
        self.log.debug(f"lam_minus: {self.lam_minus}")
        self.log.debug(f"init: {self.init}")
        self.log.debug(f"rhs: {self.rhs}")
        self.log.debug(f"Inactive: {self.inactive}")

        project = CoordinateMask(self.h_domain.vecsp, self.inactive)
        tik = TikhonovCG(
            setting=Setting(self.deriv * project, self.h_domain, self.h_codomain),
            data=self.rhs, 
            regpar=self.regpar,
            xref=self.init,
            **self.cg_pars
        )
        f, _ = tik.run()
        self.log.info(f"Inner Tikhonov solver took {tik.iteration_step_nr}.")
        self.log.debug(f"Inner Tikhonov solver result {f} iterations.")
        self.x[self.inactive] = f[self.inactive]
        z = self._A(self.x)
        
        self.lam_plus[self.active_plus]=self.b[self.active_plus]+self.lam_minus[self.active_plus]-z[self.active_plus]
        self.lam_minus[self.active_minus]=-self.b[self.active_minus]+self.lam_plus[self.active_minus]+z[self.active_minus]

        #Update active and inactive sets
        self.active_plus=self.op.domain.IfPos(self.lam_plus+self.regpar*(self.x-self.psi_plus))
        self.active_minus=self.op.domain.IfPos(self.lam_minus-self.regpar*(self.x-self.psi_minus))
        
    def _A(self, u):
        return self.regpar*u+self.h_domain.gram_inv(self.deriv.adjoint(self.h_codomain.gram(self.deriv(u))))