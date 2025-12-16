import math as ma
import numpy as np

from regpy.util import Errors

from ..general import RegSolver

__all__ = ["ForwardBackwardSplitting","FISTA"]

class ForwardBackwardSplitting(RegSolver):
    r"""    Minimizes 

    .. math::
        \mathcal{S}(Tf)+\alpha*\mathcal{R}(f)

    by forward backward splitting. 

    Parameters
    ----------
    setting : regpy.solvers.Setting
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
        if not setting.is_tikhonov:
            raise ValueError(Errors.value_error("ForwardBackwardSplitting requires the setting to contain a regularization parameter!")) 
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="ForwardBackwardSplitting requires the operator to be linear!"))
        if init is not None and init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        self.x = self.op.domain.zeros() if init is None else init

        out, par = ForwardBackwardSplitting.check_applicability(setting)
        if out['applicable']:
            self.log.info(out['info'])
        else:
            raise ValueError('ForwardBackwardSplitting not applicable to this setting. '+out['info'])
        self.tau = par['tau'] if tau is None else tau
        """The step size parameter"""
        if self.tau<=0:
            raise ValueError(Errors.value_error("tau the step size needs to be positive!"))   
        self.proximal_pars = proximal_pars
        self.log.setLevel(logging_level)

        self.y = self.op(self.x)
        self._dual_variables_computed = False

    @staticmethod
    def check_applicability(setting, op_norm=None, op_lower_bound=0.):
        out = {'info':''}; par={}
        if 'proximal' not in setting.penalty.methods:
            out['info']+='Missing prox of penalty functional. '
        if 'subgradient' not in setting.data_fid.methods:
            out['info']+='Missing gradient of data functional. '
        if setting.data_fid.Lipschitz==np.inf:
            out['info']+='Gradient of data functional not Lipschitz.'        
        out['applicable'] = out['info']==''
        if out['applicable']: 
            op_norm = setting.op.norm(setting.h_domain,setting.h_codomain) if op_norm is None else op_norm
            par = {'tau':1/(op_norm**2 * setting.data_fid.Lipschitz)}
            mu_penalty  = setting.regpar * setting.penalty.convexity_param
            mu_data_fidelity = setting.data_fid.convexity_param * op_lower_bound**2 
            out['rate'] = (1. - par['tau'] * mu_data_fidelity) / (1. + par['tau']*mu_penalty)
            out['info'] = "ForwardBackwardSplitting used with step length tau={:.3e}.\n".format(par['tau'])
            if out['rate']<1.:
                out['info'] += "Expected linear convergence rate: {:.3e}.".format(out['rate'])
            else:
                out['info'] += "Expected convergence rate O(1/n)."
                out['rate'] = -1
        return out, par

    def primal(self): # for stopping rules and monitoring
        return (self.x,self.y)

    def dual(self): # for stopping rules and monitoring
        if self._dual_variables_computed==False:
            if not hasattr(self,"p"): # first iteration
                self.p = self.op.adjoint.domain.zeros()
            else:
                self.p *= (-1./self.regpar)
            if not hasattr(self,"Tp"): # first iteration
                self.Tp = self.op.adjoint.codomain.zeros()
            else:
                self.Tp *= (-1./self.regpar)
            self._dual_variables_computed = True
        return (self.p,self.Tp)


    def _next(self):
        self._dual_variables_computed = False
        self.p = self.data_fid.subgradient(self.y)
        self.Tp = self.op.adjoint(self.p)
        self.x -= self.tau*self.h_domain.gram_inv(self.Tp)
        self.x = self.penalty.proximal(self.x, self.regpar*self.tau, **self.proximal_pars)
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        self.y = self.op(self.x)
 
class FISTA(RegSolver):
    r"""
    The generalized FISTA algorithm for minimization of Tikhonov functionals

    .. math::
        \mathcal{S}_{g^{\delta}}(F(f)) + \alpha \mathcal{R}(f).

    Gradient steps are performed on the first term, and proximal steps on the second term. 
    
    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem. Includes the penalty and data fidelity functionals. 
    init : setting.op.domain [default: setting.op.domain.zeros()]
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
    """
    def __init__(self, setting, init= None, tau = None, op_lower_bound = 0, proximal_pars=None,logging_level= "INFO"):
        if not setting.is_tikhonov:
            raise ValueError(Errors.value_error("FISTA requires the setting to contain a regularization parameter!")) 
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="For nonlinear operators the FISTA method in regpy.solvers.nonlinear must be used."))
        if init is not None and init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        self.x = self.op.domain.zeros() if init is None else init

        self.log.setLevel(logging_level)

        self.y = self.op(self.x)

        self.proximal_pars = proximal_pars
        """Proximal parameters that are passed to prox-operator of penalty term. """

        out, par = FISTA.check_applicability(setting, op_lower_bound=op_lower_bound)
        if out['applicable']:
            self.log.info(out['info'])
        else:
            raise ValueError('FISTA not applicable to this setting. '+out['info'])
        self.mu_penalty, self.mu_data_fidelity, self.mu = par['mu_penalty'], par['mu_data_fidelity'], par['mu']
        if tau is None:
            self.tau, self.q = par['tau'], par['q'] 
        else: 
            self.tau= tau
            """The step size parameter"""
            self.q = self.tau * self.mu / (1.+self.tau * self.mu_penalty)            
        if self.tau<=0:
            raise ValueError(Errors.value_error("The step size tau needs to be positive!"))  
        self.t = 0
        self.t_old = 0

        self.x_old = self.x

        self._dual_variables_computed = False
        
    @staticmethod
    def check_applicability(setting,op_lower_bound=0.,op_norm=None):
        out = {'info':''}; par={}
        if 'proximal' not in setting.penalty.methods:
            out['info']+='Missing prox of penalty functional. '
        if 'subgradient' not in setting.data_fid.methods:
            out['info']+='Missing gradient of data functional. '
        if setting.data_fid.Lipschitz==np.inf:
            out['info']+='Gradient of data functional not Lipschitz.'        
        out['applicable'] = out['info']==''
        if out['applicable']: 
            par = {}
            par['mu_penalty']  = setting.regpar * setting.penalty.convexity_param
            par['mu_data_fidelity'] = setting.data_fid.convexity_param * op_lower_bound**2 
            par['mu'] = par['mu_data_fidelity']+par['mu_penalty']
            op_norm = setting.op.norm(setting.h_domain,setting.h_codomain) if op_norm is None else op_norm
            par['tau'] = 1./(op_norm**2 * setting.data_fid.Lipschitz)
            par['q'] = (par['tau'] * par['mu']) / (1+par['tau']*par['mu_penalty'])
            out['rate'] = 1.-ma.sqrt(par['q']) 
            if par['mu']>0:
                out['info'] = "FISTA used with convexity parameters mu_R={:.3e}, mu_S={:.3e} and step length tau={:.3e}.\nExpected linear convergence rate: {:.3e}.\n".format(par['mu_penalty'],par['mu_data_fidelity'],par['tau'],out['rate'])
            else:
                out['info'] = "Expected convergence rate O(1/n^2)."
                out['rate'] = -2
        return out, par

    def primal(self): # for stopping rules and monitoring
        return (self.x,self.y)
    
    def dual(self): # for stopping rules and monitoring
        if self._dual_variables_computed==False:
            if not hasattr(self,"p"): # first iteration
                self.p = self.op.adjoint.domain.zeros()
            else:
                self.p *= (-1./self.regpar)
            if not hasattr(self,"Tp"): # first iteration
                self.Tp = self.op.adjoint.codomain.zeros()
            else:
                self.Tp *= (-1./self.regpar)
            self._dual_variables_computed = True
        return (self.p,self.Tp)

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

        self._dual_variables_computed = False
        self.p = self.data_fid.subgradient(self.y)
        self.Tp = self.op.adjoint(self.p)
        grad = self.h_domain.gram_inv(self.Tp)
        self.x = self.penalty.proximal(h-self.tau*grad, self.tau * self.regpar, self.proximal_pars)
        self.y = self.op(self.x)

