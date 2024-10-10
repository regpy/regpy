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
        lower bound of the operator: :math:`\|op(f)\|\geq op_lower_bound * \|f\|`\.
        Used to define convexity parameter of data functional.     
    proximal_pars : dict [default: {}]
        Parameter dictionary passed to the computation of the prox-operator for the penalty term. 
    logging_level: [default: logging.INFO]
        logging level
    """
    def __init__(self, setting, init= None, tau = None, op_lower_bound = 0, proximal_pars=None,logging_level= logging.INFO):
        assert isinstance(setting,TikhonovRegularizationSetting)
        super().__init__(setting)
        self.x = self.op.domain.zeros() if init is None else init
        assert init is None or init in self.op.domain
        
        self.y, self.deriv = self.op.linearize(self.x)
        self.log.setLevel(logging_level)

        self.mu_penalty  = self.regpar * self.penalty.convexity_param
        self.mu_data_fidelity = self.data_fid.convexity_param * op_lower_bound**2
        self.proximal_pars = proximal_pars
        """Proximal parameters that are passed to prox-operator of penalty term. """

        self.tau = 1./(setting.op_norm(op=self.deriv)**2 * self.data_fid.Lipschitz) if tau is None else tau
        """The step size parameter"""
        assert self.tau>0 

        self.t = 0
        self.t_old = 0
        self.mu = self.mu_data_fidelity+self.mu_penalty

        self.x_old = self.x
        self.q = (self.tau * self.mu) / (1+self.tau*self.mu_penalty)
        if self.mu>0:
            self.log.info('Setting up FISTA with convexity parameters mu_R={:.3e}, mu_S={:.3e} and step length tau={:.3e}.\n Expected linear convergence rate: {:.3e}'.format(
                self.mu_penalty,self.mu_data_fidelity,self.tau,1.-np.sqrt(self.q)))
        else: 
            self.log.info('Setting up FISTA with step length tau={:.3e}.'.format(self.tau))

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
