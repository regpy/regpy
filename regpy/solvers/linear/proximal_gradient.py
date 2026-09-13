import math as ma
import numpy as np
from typing import Any

from regpy.functionals.base import NullFunctional
from regpy.util import Errors
from regpy.vecsps import Vector
from regpy.operators import ObsDataT

from ..solver import RegSolver
from ..setting import Setting

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
        The initial guess. In None, the reference vector of the penalty functional is used.
    tau : float , optional
        The step size parameter. Must be positive. 
        Default is the reciprocal of the operator norm of :math:`T^*T` 
    op_norm : float or None [default: None]
        Norm of the forward operator. If supplied, it is used to compute the
        default step size ``1 / (op_norm**2 * data_fid.Lipschitz)`` instead of
        estimating the norm numerically. An explicitly supplied ``tau`` takes
        precedence.
    proximal_pars: dict, optional
        Parameter dictionary passed to the computation of the prox-operator.
    logging_level: int [default: logging.INFO]
        logging level
    update_setting : bool [default: False]
        If True, explicitly supplied data and initial guess are stored in the setting.
    """

    def __init__(self, 
                 setting: Setting, 
                 init: Vector| None =None, 
                 tau: float = None, 
                 proximal_pars: dict = None, 
                 logging_level: int|str = "INFO",
                 op_norm: float | None = None, 
                 update_setting: bool = False):
        if not setting.is_tikhonov:
            raise ValueError(Errors.value_error("ForwardBackwardSplitting requires the setting to contain a regularization parameter!")) 
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="ForwardBackwardSplitting requires the operator to be linear!"))
        if init is not None and init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        self.x = self.op.domain.empty()
        self.setting.get_or_update_initial_guess(init, update_setting, out=self.x)
        out, par = ForwardBackwardSplitting.check_applicability(
            setting, op_norm=op_norm
        )
        if out['applicable']:
            self.log.info(out['info'])
        else:
            raise ValueError('ForwardBackwardSplitting not applicable to this setting. '+out['info'])
        self.tau = par['tau'] if tau is None else tau
        """The step size parameter"""
        if self.tau<=0:
            raise ValueError(Errors.value_error("tau the step size needs to be positive!"))   
        self.proximal_pars = {} if proximal_pars is None else dict(proximal_pars)
        self.log.setLevel(logging_level)

        self.y = self.op.codomain.empty()
        self.p = self.op.codomain.zeros()
        self.Tp = self.op.domain.zeros()
        self._domain_work = self.op.domain.empty()
        self.op(self.x, out=self.y)
        self._dual_variables_computed = False

    @staticmethod
    def check_applicability(setting, op_norm=None, op_lower_bound=0.):
        out = {'info':''}; par={}
        if not setting.is_tikhonov:
            out['info'] += 'Setting has no regularization parameter. '
        if not setting.is_convex:
            out['info'] += 'Setting is not convex. '
        if setting.codomain_free:
            out['info'] += (
                'ForwardBackwardSplitting requires codomain vectors. '
            )
        if 'proximal' not in setting.penalty.methods:
            out['info']+='Missing prox of penalty functional. '
        if not setting.codomain_free:
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
            scale = -1./self.regpar
            self.p *= scale
            self.Tp *= scale
            self._dual_variables_computed = True
        return (self.p,self.Tp)


    def _next(self):
        self._dual_variables_computed = False
        self.data_fid.subgradient(self.y, out=self.p)
        self.op.adjoint(self.p, out=self.Tp)
        self.h_domain.gram_inv(self.Tp, out=self._domain_work)
        self._domain_work *= -self.tau
        self.x += self._domain_work
        self.penalty.proximal(
            self.x, self.regpar*self.tau, out=self.x, **self.proximal_pars
        )
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        self.op(self.x, out=self.y)
 
class FISTA(RegSolver):
    r"""
    The generalized FISTA algorithm for minimization of Tikhonov functionals

    .. math::
        \mathcal{S}_{g^{\delta}}(F(f)) + \alpha \mathcal{R}(f).

    Gradient steps are performed on the first term, and proximal steps on the
    second term. For a setting without a regularization parameter, FISTA
    instead minimizes only the data-fidelity term. This is admissible when the
    setting penalty is of squared-norm type (and therefore supplies the domain
    Hilbert geometry) or is a :class:`~regpy.functionals.base.NullFunctional`.
    The solver uses a null functional as its proximal term in this mode.
    
    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem. Includes the penalty and data
        fidelity functionals. If ``setting.regpar`` is ``None``, the setting
        penalty must be of squared-norm type or a
        :class:`~regpy.functionals.base.NullFunctional`; it then determines
        ``setting.h_domain`` but is not part of the minimized objective.
    init : Vector or None [default: None]
        The initial guess. 
    tau : float [default: None]
        Step size of minimization procedure. In the default case the reciprocal of the operator norm of $T^*T$ is used.
    op_norm : float or None [default: None]
        Norm of the forward operator. If supplied, it is used to compute the
        default step size ``1 / (op_norm**2 * data_fid.Lipschitz)`` instead of
        estimating the norm numerically. For codomain-free settings, the data
        fidelity Lipschitz factor is one. An explicitly supplied ``tau`` takes
        precedence.
    op_lower_bound : float [default: 0]
        lower bound of the operator: :math:`|op(f)|\geq op_lower_bound * |f|`\.
        Used to define convexity parameter of data functional.     
    proximal_pars : dict [default: {}]
        Parameter dictionary passed to the computation of the prox-operator for the penalty term. 
    data : ObsDataT or None [default: None]
        The observed data. Only relevant if ``setting.codomain_free`` is True. 
        Can be an operator-specific handle accepted by ``op.adjoint_data``.
    logging_level: [default: logging.INFO]
        logging level
    update_setting : bool [default: False]
        If True, explicitly supplied data and initial guess are stored in the setting.

    Notes
    -----
    In the unregularized mode, the iteration minimizes

    .. math::
        \mathcal{S}_{g^{\delta}}(F(f)) + 0.

    This use as an iterative regularization method is exposed by
    :class:`~regpy.solvers.Setting` under the short method name ``"FISTA"``
    for applicable linear settings.
    """
    def __init__(
            self,
            setting: Setting,
            init: Vector | None = None,
            tau: float | None = None,
            op_lower_bound: float = 0.,
            proximal_pars: dict | None = None,
            data: Vector | ObsDataT | None = None,
            op_norm: float | None = None,
            logging_level: int | str = "INFO",
            update_setting: bool = False):
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="For nonlinear operators the FISTA method in regpy.solvers.nonlinear must be used."))
        if setting.is_tikhonov:
            self.regpar = setting.regpar
            self.penalty = setting.penalty
        else:
            # The setting penalty supplies the domain Hilbert geometry but is
            # not part of the objective in iterative-regularization mode.
            self.regpar = 1.
            self.penalty = NullFunctional(self.op.domain)
        self.x = setting.get_or_update_initial_guess(init, update_setting)

        self.log.setLevel(logging_level)

        self.codomain_free = setting.codomain_free
        
        self.y = None
        if self.codomain_free:
            data = setting.get_or_update_data(new_data=data, update=update_setting)
            if data is None:
                raise ValueError(Errors.value_error(
                    "codomain_free FISTA requires data either in the setting or as an argument."
            ))
            self.backpropagated_data = self.op.adjoint_data(data)

        self.proximal_pars = {} if proximal_pars is None else dict(proximal_pars)
        """Proximal parameters that are passed to prox-operator of penalty term. """

        out, par = FISTA.check_applicability(
            setting,
            op_lower_bound=op_lower_bound,
            op_norm=op_norm,
        )
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

        self.x_old = self.op.domain.empty()
        self.op.domain.copyto(self.x_old, self.x)
        self.p = None if self.codomain_free else self.op.codomain.zeros()
        self.Tp = self.op.domain.zeros()
        self._h = self.op.domain.empty()
        self._domain_work = self.op.domain.empty()
        self._dual_variables_computed = False
        
    @staticmethod
    def check_applicability(setting, op_lower_bound=0., op_norm=None):
        """Check the requirements for either supported use of linear FISTA.

        A Tikhonov setting must provide the proximal map of its penalty. A
        setting without a regularization parameter is accepted only if its
        penalty is of squared-norm type or is a :class:`NullFunctional`; the
        solver substitutes a :class:`NullFunctional` in that case.
        """
        out = {'info':''}; par={}
        without_regpar = not setting.is_tikhonov
        admissible_without_regpar = (
            setting.penalty.is_norm_squared_type
            or isinstance(setting.penalty, NullFunctional)
        )
        if without_regpar and not admissible_without_regpar:
            out['info'] += (
                'Without a regularization parameter, the penalty must be '
                'of squared-norm type or a NullFunctional. '
            )
        if not without_regpar and 'proximal' not in setting.penalty.methods:
            out['info']+='Missing prox of penalty functional. '
        if not setting.codomain_free:
            if 'subgradient' not in setting.data_fid.methods:
                out['info'] += 'Missing gradient of data functional. '
            if setting.data_fid.Lipschitz == np.inf:
                out['info'] += 'Gradient of data functional not Lipschitz. '
        out['applicable'] = out['info']==''
        if out['applicable']: 
            par = {}
            par['mu_penalty'] = (
                setting.regpar * setting.penalty.convexity_param
                if setting.is_tikhonov else 0.
            )
            par['mu_data_fidelity'] = (
                0. if setting.codomain_free
                else setting.data_fid.convexity_param * op_lower_bound**2
            )
            par['mu'] = par['mu_data_fidelity']+par['mu_penalty']
            if op_norm is None:
                op_norm = setting.op.norm(
                    setting.h_domain,
                    None if setting.codomain_free else setting.h_codomain,
                    codomain_free=setting.codomain_free,
                )
            data_lipschitz = (
                1. if setting.codomain_free else setting.data_fid.Lipschitz
            )
            par['tau'] = 1./(op_norm**2 * data_lipschitz)
            par['q'] = (par['tau'] * par['mu']) / (1+par['tau']*par['mu_penalty'])
            out['rate'] = 1.-ma.sqrt(par['q']) 
            if par['mu']>0:
                out['info'] = "FISTA used with convexity parameters mu_R={:.3e}, mu_S={:.3e} and step length tau={:.3e}.\nExpected linear convergence rate: {:.3e}.\n".format(par['mu_penalty'],par['mu_data_fidelity'],par['tau'],out['rate'])
            else:
                out['info'] = "Expected convergence rate O(1/n^2)."
                out['rate'] = -2
        return out, par

    def primal(self): # for stopping rules and monitoring
        #if self.codomain_free:
        #    raise RuntimeError(Errors.runtime_error("Primal variables not available when FISTA is run without codomain vectors!"))
        return (self.x,None)
    
    def dual(self): # for stopping rules and monitoring
        if self.codomain_free:
            raise RuntimeError(Errors.runtime_error("Primal variables not available when FISTA is run without codomain vectors!"))
        if self._dual_variables_computed==False:
            scale = -1./self.regpar
            self.p *= scale
            self.Tp *= scale
            self._dual_variables_computed = True
        return (self.p,self.Tp)

    def _next(self):
        if self.mu == 0:
            self.t = (1 + ma.sqrt(1+4*self.t_old**2))/2
            beta = (self.t_old-1) / self.t
        else: 
            self.t = (1-self.q*self.t_old**2+ma.sqrt((1-self.q*self.t_old**2)**2+4*self.t_old**2))/2
            beta = (self.t_old-1)/self.t * (1+self.tau*self.mu_penalty-self.t*self.tau*self.mu)/(1-self.tau*self.mu_data_fidelity)

        # h = x + beta * (x - x_old)
        self.op.domain.copyto(self._h, self.x)
        self.op.domain.copyto(self._domain_work, self.x)
        self._domain_work -= self.x_old
        self._domain_work *= beta
        self._h += self._domain_work

        self.t_old = self.t

        self._dual_variables_computed = False
        if self.codomain_free:
             self.op.adjoint_eval(self._h, out=self.Tp)
             self.Tp -= self.backpropagated_data
        else:
            self.op(self._h, out=self.p)
            self.data_fid.subgradient(self.p, out=self.p)
            self.op.adjoint(self.p, out=self.Tp)

        self.h_domain.gram_inv(self.Tp, out=self._domain_work)
        self._domain_work *= -self.tau
        self._h += self._domain_work

        # The old iterate is no longer needed after forming h. Reuse its
        # buffer for the next iterate and then swap the two solution buffers.
        self.penalty.proximal(
            self._h, self.tau * self.regpar, out=self.x_old,
            **self.proximal_pars
        )
        self.x, self.x_old = self.x_old, self.x
        
