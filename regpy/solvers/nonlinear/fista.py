from math import sqrt,inf
import numpy as np

from regpy.util import Errors
from ..general import RegSolver, Setting

__all__ = ["FISTA"]

class FISTA(RegSolver):
    r"""
    The generalized FISTA algorithm for minimization of Tikhonov functionals
    
    .. math:: 
        \mathcal{S}_{g^{\delta}}(F(f)) + \alpha \mathcal{R}(f).

    Gradient steps are performed on the first term, and proximal steps on the second term. 
    The step sizes for the gradient steps are determined using a backtracking method introduced in
    A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding algorithm for
    linear inverse problems. SIAM J. Imaging Sci., 2(1):183–202, 2009.
    
    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem. Includes the penalty and data fidelity functionals. 
    init : setting.op.domain [default: setting.op.domain.zeros()]
        The initial guess
    tau : float [default: 10**16]
        Initial step size of minimization procedure. Has to be sufficiently large.
    eta : float [default 0.8]
        Step size reduction constant.
    op_lower_bound : float [default: 0]
        lower bound of the operator: :math:`\|op(f)\|\geq op_lower_bound * \|f\|`\.
        Used to define convexity parameter of data functional.     
    proximal_pars : dict [default: {}]
        Parameter dictionary passed to the computation of the prox-operator for the penalty term. 
    logging_level: [default: logging.INFO]
        logging level
    """
    def __init__(self, setting, init= None, tau = 10**16, eta = 0.8, op_lower_bound = 0, 
                 proximal_pars=None,logging_level= "INFO",
                 data=None, without_codomain_vectors=False):
        if not setting.is_tikhonov:
            raise ValueError(Errors.value_error("FISTA requires the setting to contain a regularization parameter!")) 
        super().__init__(setting)
        if self.op.linear:
            self.log.warning("Using non-linear FISTA with a linear Operator! Consider using the linear FISTA in the module solvers.linear")
        if init is not None and init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        self.x = self.op.domain.zeros() if init is None else init
        
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        self.data=data
        self.log.setLevel(logging_level)
        self.regpar = self.setting.regpar
        self.mu_penalty  = self.regpar * self.penalty.convexity_param
        self.mu_data_fidelity = self.data_fid.convexity_param * op_lower_bound**2
        self.proximal_pars = proximal_pars
        """Proximal parameters that are passed to prox-operator of penalty term. """

        self.without_codomain_vectors=without_codomain_vectors
        self.eta = eta
        if not (0<self.eta<1):
            raise ValueError(Errors.value_error("The Step size reduction constant must be between 0 and 1!"))

        if self.data_fid.Lipschitz != inf:
            if without_codomain_vectors:
                _, deriv = self.op.linearize(self.x,return_adjoint_eval=True)
            else:
                self.y, deriv = self.op.linearize(self.x)
            self.tau = 1./(deriv.norm(setting.h_domain,setting.h_codomain,
                            without_codomain_vectors=without_codomain_vectors)**2 * self.data_fid.Lipschitz)
            """The step size parameter"""
            self.backtracking = False
        else:
            self.y = self.op(self.x)
            self.tau = tau
            self.backtracking = True
        if self.tau<=0:
            raise ValueError(Errors.value_error("The Step size must be positive!"))
        self.t = 0
        self.t_old = 0
        self.mu = self.mu_data_fidelity+self.mu_penalty

        self.x_old = self.x
        self.q = (self.tau * self.mu) / (1+self.tau*self.mu_penalty)
        if self.mu>0:
            self.log.info('Setting up FISTA with convexity parameters mu_R={:.3e}, mu_S={:.3e} and step length tau={:.3e}.\n Expected linear convergence rate: {:.3e}'.format(
                self.mu_penalty,self.mu_data_fidelity,self.tau,1.-sqrt(self.q)))
        else: 
            self.log.info('Setting up FISTA with step length tau={:.3e}.'.format(self.tau))

    def _next(self):
        if self.mu == 0:
            self.t = (1 + sqrt(1+4*self.t_old**2))/2
            beta = (self.t_old-1) / self.t
        else: 
            self.q = (self.tau * self.mu) / (1+self.tau*self.mu_penalty)
            self.t = (1-self.q*self.t_old**2+sqrt((1-self.q*self.t_old**2)**2+4*self.t_old**2))/2
            beta = (self.t_old-1)/self.t * (1+self.tau*self.mu_penalty-self.t*self.tau*self.mu)/(1-self.tau*self.mu_data_fidelity)

        h = self.x+beta*(self.x-self.x_old)
        self.x_old = self.x
        self.t_old = self.t

        if self.without_codomain_vectors:
            adjoint_eval_h, deriv = self.op.linearize(h,return_adjoint_eval=True)
            grad =self.h_domain.gram_inv(adjoint_eval_h - deriv.adjoint_data(self.data))
        else:
            image_of_h, deriv = self.op.linearize(h)
            if self.backtracking: 
                data_fid_of_h = self.data_fid(image_of_h)
            grad = self.h_domain.gram_inv(deriv.adjoint(self.data_fid.subgradient(image_of_h)))
        self.x = self.penalty.proximal(h-self.tau*grad, self.tau * self.regpar, self.proximal_pars)
        while self.backtracking:
            if self.data_fid(self.op(self.x)) <= data_fid_of_h + self.setting.h_domain.inner(self.x - h, grad) + (1/(2*self.tau))*self.setting.h_domain.inner(self.x - h, self.x - h):
                break
            self.tau *= self.eta
            self.x = self.penalty.proximal(h-self.tau*grad, self.tau * self.regpar, self.proximal_pars)

        if self.without_codomain_vectors:
            self.y = np.broadcast_to(np.zeros(()), self.op.codomain.shape)
        else:
            self.y = self.op(self.x)
