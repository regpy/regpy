import math as ma

from regpy.util import Errors

from ..general import RegSolver, TikhonovRegularizationSetting

__all__ = ["ForwardBackwardSplitting","FISTA"]

class ForwardBackwardSplitting(RegSolver):
    r"""    Minimizes 

    .. math::
        \mathcal{S}(Tf)+\alpha*\mathcal{R}(f)

    by forward backward splitting. 

    Parameters
    ----------
    setting : regpy.solvers.TikhonovRegularizationSetting
        The setting of the forward problem. Includes both penalty :math:`\mathcal{R}` and data fidelity :math:`\mathcal{S}` functional. 
    init : setting.domain [default: None]
        The initial guess. (domain.zeros() in the default case)
    tau : float , optional
        The step size parameter. Must be positive. 
        Default is the reciprocal of the operator norm of :math:`T^*T` 
    proximal_pars: dict, optional
        Parameter dictionary passed to the computation of the prox-operator.
    logging_level: int [default: logging.INFO]
        logging level
    """

    def __init__(self, setting, init=None, tau = None, proximal_pars = {}, logging_level = "INFO"):
        if not isinstance(setting,TikhonovRegularizationSetting):
            raise TypeError(Errors.not_instance(setting,TikhonovRegularizationSetting,add_info="ForwardBackwardSplitting requires the Setting to be a Tikhonov setting!"))
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="ForwardBackwardSplitting requires the operator to be linear!"))
        if init is not None and init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        self.x = self.op.domain.zeros() if init is None else init

        self.tau = 1/setting.op.norm(setting.h_domain,setting.h_codomain)**2 if tau is None else tau
        """The step size parameter"""
        if self.tau<=0:
            raise ValueError(Errors.value_error("tau the step size needs to be positive!"))   
        self.proximal_pars = proximal_pars
        self.log.setLevel(logging_level)

        self.y = self.op(self.x)

        # try:
        #     self.gap=self.setting.dualityGap(primal = self.x)
        #     self.dualityGapWorks =True
        # except NotImplementedError:
        #     self.dualityGapWorks = False
        
    def _next(self):
        self.x -= self.tau*self.h_domain.gram_inv(self.op.adjoint(self.data_fid.subgradient(self.y)))
        self.x = self.penalty.proximal(self.x, self.regpar*self.tau, **self.proximal_pars)
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        self.y = self.op(self.x)
 
        # if self.dualityGapWorks:
        #     self.gap=self.setting.dualityGap(primal = self.x,dual=self.setting.primalToDual(self.y,argumentIsOperatorImage=True) )
            
class FISTA(RegSolver):
    r"""
    The generalized FISTA algorithm for minimization of Tikhonov functionals

    .. math::
        \mathcal{S}_{g^{\delta}}(F(f)) + \alpha \mathcal{R}(f).

    Gradient steps are performed on the first term, and proximal steps on the second term. 
    
    Parameters
    ----------
    setting : regpy.solvers.TikhonovRegularizationSetting
        The setting of the forward problem. Includes the penalty and data fidelity functionals. 
    init : setting.op.domain [defaul: setting.op.domain.zeros()]
        The initial guess
    tau : float [default: None]
        Step size of minimization procedure. In the default case the reciprocal of the operator norm of $T^*T$ is used.
    op_lower_bound : float [default: 0]
        lower bound of the operator: :math:`|op(f)|\geq op_lower_bound * |f|`\.
        Used to define convexity parameter of data functional.     
    proximal_pars : dict [default: {}]
        Parameter dictionary passed to the computation of the prox-operator for the penalty term. 
    logging_level: [default: logging.INFO]
        logging level
    compute_dual: boolean [False]
        sets if dual is computed, it is not directly necessary for the algorithm. The default is False
    """
    def __init__(self, setting, init= None, tau = None, op_lower_bound = 0, proximal_pars=None,logging_level= "INFO",compute_dual = False):
        if not isinstance(setting,TikhonovRegularizationSetting):
            raise TypeError(Errors.not_instance(setting,TikhonovRegularizationSetting,add_info="ForwardBackwardSplitting requires the Setting to be a Tikhonov setting!"))
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="ForwardBackwardSplitting requires the operator to be linear!"))
        if init is not None and init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        self.x = self.op.domain.zeros() if init is None else init

        self.log.setLevel(logging_level)

        self.y = self.op(self.x)

        self.mu_penalty  = self.regpar * self.penalty.convexity_param
        self.mu_data_fidelity = self.data_fid.convexity_param * op_lower_bound**2
        self.proximal_pars = proximal_pars
        """Proximal parameters that are passed to prox-operator of penalty term. """

        self.tau = 1./(setting.op.norm(setting.h_domain,setting.h_codomain)**2 * self.data_fid.Lipschitz) if tau is None else tau
        """The step size parameter"""
        if self.tau<=0:
            raise ValueError(Errors.value_error("tau the step size needs to be positive!"))  
        self.t = 0
        self.t_old = 0
        self.mu = self.mu_data_fidelity+self.mu_penalty

        self.x_old = self.x
        self.q = (self.tau * self.mu) / (1+self.tau*self.mu_penalty)
        if self.mu>0:
            self.log.info('Setting up FISTA with convexity parameters mu_R={:.3e}, mu_S={:.3e} and step length tau={:.3e}.\n Expected linear convergence rate: {:.3e}'.format(
                self.mu_penalty,self.mu_data_fidelity,self.tau,1.-ma.sqrt(self.q)))
            
        if not hasattr(self,"compute_dual") or not self.compute_dual:
            self.compute_dual = compute_dual
        if self.compute_dual:
            self._compute_dual()
        

    def _compute_dual(self):
        self.dual=self.setting.primalToDual(self.y,argumentIsOperatorImage=True,own=True)


    def _next(self):
        if self.mu == 0:
            self.t = (1 + ma.sqrt(1+4*self.t_old**2))/2
            beta = (self.t_old-1) / self.t
        else: 
            self.t = (1-self.q*self.t_old**2+ma.sqrt((1-self.q*self.t_old**2)**2+4*self.t_old**2))/2
            beta = (self.t_old-1)/self.t * (1+self.tau*self.mu_penalty-self.t*self.tau*self.mu)/(1-self.tau*self.mu_data_fidelity)

        h = self.x+beta*(self.x-self.x_old)

        self.x_old = self.x
        self.t_old = self.t

        grad = self.h_domain.gram_inv(self.op.adjoint(self.data_fid.subgradient(self.y) ))
        self.x = self.penalty.proximal(h-self.tau*grad, self.tau * self.regpar, self.proximal_pars)
        self.y = self.op(self.x)

        if self.compute_dual:
            self._compute_dual()