import numpy as np
import logging

from regpy.solvers import RegSolver, TikhonovRegularizationSetting

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

class FISTA(RegSolver):
    r"""
    The generalized FISTA algorithm for minimization of Tikhonov functionals
    \[ \mathcal{S}_{g^{\delta}}(F(f)) + \alpha \mathcal{R}(f).
    \] 
    Gradient steps are performed on the first term, and proximal steps on the second term. 
    
    Parameters:
    -----------
    setting : regpy.solvers.TikhonovRegularizationSetting
        The setting of the forward problem. Includes the penalty and data fidelity functionals. 
    init : setting.op.domain [defaul: setting.op.domain.zeros()]
        The initial guess
    tau : float [default: None]
        Step size of minimization procedure. In the default case the reciprocal of the operator norm of $T^*T$ is used.
    op_lower_bound : float [default: 0]
        lower bound of the operator: \(\|op(f)\|\geq op_lower_bound * \|f\| \).
        Used to define convexity parameter of data functional.     
    proximal_pars : dict [default: {}]
        Parameter dictionary passed to the computation of the prox-operator for the penalty term. 
    """
    def __init__(self, setting, init= None, tau = None, op_lower_bound = 0, proximal_pars=None,logging_level= logging.INFO):
        assert isinstance(setting,TikhonovRegularizationSetting)
        super().__init__(setting)
        self.setting = setting
        self.regpar = setting.regpar
        if init is None:
            self.x = self.op.domain.zeros()
        else:
            assert init in self.op.domain
            self.x = init
        self.y, self.deriv = self.op.linearize(self.x)
        self.log.setLevel(logging_level)

        self.mu_penalty  = self.regpar * self.penalty.convexity_param
        self.mu_data_fidelity = self.data_fid.convexity_param * op_lower_bound**2
        self.proximal_pars = proximal_pars
        """Proximal parameters that are passed to prox-operator of penalty term. """

        assert tau is None or tau>0
        if tau is None:
            self.tau = 1./(setting.op_norm(op=self.deriv)*self.data_fid.Lipschitz)
        else:
            self.tau = tau
            """The step size parameter"""
 

        self.t = 0
        self.t_old = 0
        self.mu = self.mu_data_fidelity+self.mu_penalty

        self.x_old = self.x
        self.q = (self.tau * self.mu) / (1+self.tau*self.mu_penalty)
        if self.mu>0:
            self.log.info('Set up FISTA with convexity parameters mu_R={:.3e}, mu_S={:.3e} and step length tau={:.3e}.\n Expected linear convergence rate: {:.3e}'.format(
                self.mu_penalty,self.mu_data_fidelity,self.tau,1.-np.sqrt(self.q)))
        try:
            gap=self.setting.dualityGap(primal = self.x)
            self.dualityGapWorks =True
            self.log.info('initial duality gap: {}'.format(gap))
        except NotImplementedError:
            self.dualityGapWorks = False


    def _next(self):
        if self.mu == 0:
            self.t = (1 + np.sqrt(1+4*self.t_old**2))/2
            beta = (self.t_old-1) / self.t
        else: 
            self.t = (1-self.q*self.t_old**2+np.sqrt((1-self.q*self.t_old**2)**2+4*self.t_old**2))/2
            beta = (self.t_old-1)/self.t * (1+self.tau*self.mu_penalty-self.t*self.tau*self.mu)/(1-self.tau*self.mu_data_fidelity)

        h = self.x+beta*(self.x-self.x_old)

        self.x_old = self.x
        self.t_old = self.t

        grad = self.h_domain.gram_inv(self.deriv.adjoint(self.data_fid.subgradient(self.y) ))
        self.x = self.penalty.proximal(h-self.tau*grad, self.tau * self.regpar, self.proximal_pars)
        self.y, self.deriv = self.op.linearize(self.x)

        if self.dualityGapWorks:
            gap=self.setting.dualityGap(primal = self.x,dual=self.setting.primalToDual(self.y,argumentIsOperatorImage=True) )
            self.log.debug('it.{}: duality gap={:.3e}'.format(self.iteration_step_nr,gap))