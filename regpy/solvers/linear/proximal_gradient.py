from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from regpy.vecsps.base import ReadOnlyVector
import math as ma
import numpy as np
from typing import Any

from regpy.functionals.base import NullFunctional, FunctionalOnDirectSum
from regpy.functionals.integral_functionals import Quadratic
from regpy.util import Errors
from regpy.vecsps import Vector
from regpy.operators import ObsDataT

from ..solver import RegSolver
from ..setting import Setting

__all__ = ["ForwardBackwardSplitting","FISTA"]


def _unregularized_penalty(penalty):
    r"""Retain only admissible constraints when there is no regpar.

    Unconstrained squared-norm penalties define the Hilbert geometry but do
    not enter the unregularized objective. A Quadratic integral functional
    contributes only its box indicator; its a, b, and c coefficients are all
    discarded. Direct sums are handled recursively without changing the
    original setting penalty or its Hilbert geometry.
    """
    if isinstance(penalty, FunctionalOnDirectSum):
        return FunctionalOnDirectSum(
            [_unregularized_penalty(func) for func in penalty.funcs],
            domain=penalty.domain,
        )
    if isinstance(penalty, Quadratic):
        # Taylor extensions modify the integrand outside prescribed regions;
        # they are not box constraints, and must not be silently retained or
        # discarded as though they were constraints.
        if any(getattr(penalty, name) is not None for name in (
            'lin_taylor_l', 'lin_taylor_u',
            'quad_taylor_l', 'quad_taylor_u',
        )):
            raise ValueError(
                'Unregularized Quadratic penalties may have bound constraints, '
                'but not Taylor extensions.'
            )
        return Quadratic(
            penalty.domain, a=0., b=0., c=0.,
            constr_l=penalty.constr_l,
            constr_u=penalty.constr_u,
        )
    if isinstance(penalty, NullFunctional) or penalty.is_norm_squared_type:
        return NullFunctional(penalty.domain)
    raise ValueError(
        'Without a regularization parameter, the penalty must be a '
        'NullFunctional, a squared-norm-type functional, a Quadratic '
        'integral functional with optional bounds, or a direct sum of these.'
    )


class ForwardBackwardSplitting(RegSolver):
    r"""Minimizes 
    
    .. math::
        \mathcal{S}(Tf)+\alpha*\mathcal{R}(f)
    
    by forward-backward splitting. Without a regularization parameter,
    minimizes the data fidelity subject to the bounds of any supported
    Quadratic integral penalty. See :class:`FISTA` for the supported penalty
    types and direct sums. The quadratic coefficients are discarded in this
    mode; the setting's Hilbert geometry is unchanged.
    
    Parameters
    ----------
    setting: Setting
        The setting of the forward problem. Includes both penalty :math:`\mathcal{R}` and data fidelity :math:`\mathcal{S}` functional.
    init: ReadOnlyVector | None [default: None]
        The initial guess. In None, the reference vector of the penalty functional is used.
    tau: float | None [default: None]
        The step size parameter. Must be positive. 
        Default is the reciprocal of the operator norm of :math:`T^*T`
    proximal_pars: dict | None [default: None]
        Parameter dictionary passed to the computation of the prox-operator.
    logging_level: int | str [default: 'INFO']
        logging level
    op_norm: float | None [default: None]
        Norm of the forward operator. If supplied, it is used to compute the
        default step size ``1 / (op_norm**2 * data_fid.Lipschitz)`` instead of
        estimating the norm numerically (on the originating primal operator
        for a transformed dual setting). An explicitly supplied ``tau`` takes
        precedence.
    update_setting: bool [default: False]
        If True, explicitly supplied data and initial guess are stored in the setting.
    is_primal: bool [default: True]
        Whether iterates represent the original primal problem.
        Pass False when solving its dual. This labels coordinates and does not
        change the iteration formulas or construct a dual setting.
    data: Vector | ObsDataT | None [default: None]
        Observations used by ``op.adjoint_data`` in codomain-free mode. If
        omitted, observations are taken from ``setting``.
    """

    def __init__(self, 
                 setting: Setting, 
                 init: ReadOnlyVector | None =None, 
                 tau: float | None = None, 
                 proximal_pars: dict | None = None, 
                 logging_level: int|str = "INFO",
                 op_norm: float | None = None, 
                 update_setting: bool = False,
                 is_primal: bool = True,
                 data: Vector | ObsDataT | None = None,
                 ) -> None:
        super().__init__(setting, is_primal=is_primal)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(
                self.op, add_info='For nonlinear operators use the '
                'nonlinear ForwardBackwardSplitting solver.'
            ))
        if setting.is_tikhonov:
            self.regpar = setting.regpar
        else:
            self.regpar = 1.
            self.penalty = _unregularized_penalty(setting.penalty)
        self.codomain_free = setting.codomain_free
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

        self.y = None if self.codomain_free else self.op.codomain.empty()
        self.p = None if self.codomain_free else self.op.codomain.zeros()
        self.Tp = self.op.domain.zeros()
        self._domain_work = self.op.domain.empty()
        if self.codomain_free:
            data = setting.get_or_update_data(
                new_data=data, update=update_setting
            )
            self.backpropagated_data = self.op.adjoint_data(data)
        else:
            self.op(self.x, out=self.y)
        self._dual_variables_computed = False

    @staticmethod
    def check_applicability(
        setting: Setting, op_norm: float | None = None,
        op_lower_bound: float = 0.,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        r"""Check the same linear settings as :meth:`FISTA.check_applicability`.

        The unregularized penalty contributes only admissible box constraints.
        Codomain-free normal-operator evaluation is supported.
        """
        out, fista_par = FISTA.check_applicability(
            setting, op_lower_bound=op_lower_bound, op_norm=op_norm
        )
        if not out['applicable']:
            return out, {}
        tau = fista_par['tau']
        mu_penalty = fista_par['mu_penalty']
        mu_data = fista_par['mu_data_fidelity']
        rate = (1. - tau * mu_data) / (1. + tau * mu_penalty)
        out['rate'] = -1. if rate == 1. else rate
        out['info'] = (
            ('Dual ' if setting._primal_setting is not None else '')
            + f'FB used with step length tau={tau:.3e}.'
        )
        return out, {'tau': tau}

    def primal(self) -> tuple[Vector | None, Vector | None]: 
        r"""Return the current primal iterate and operator image for stopping rules and monitoring.
        
        
        Returns
        -------
        tuple[Vector | None, Vector | None]
            Result described above.
        """
        return (self.x,self.y)

    def dual(self) -> tuple[Vector | None, Vector | None]: 
        r"""Returns the dual vectors and its adjoint operator image for stopping rules and monitoring.
        (The dual vectors appearing in iteration formula must be rescaled by (-1/alpha)!) 
        
        
        Returns
        -------
        tuple[Vector | None, Vector | None]
            Result described above.
        """
        if self.codomain_free:
            raise RuntimeError(Errors.runtime_error(
                'Dual variables are unavailable in codomain-free FB.'
            ))
        if self._dual_variables_computed==False:
            scale = -1./self.regpar
            self.p *= scale
            self.Tp *= scale
            self._dual_variables_computed = True
        return (self.p,self.Tp)


    def _next(self) -> None:
        self._dual_variables_computed = False
        if self.codomain_free:
            self.op.adjoint_eval(self.x, out=self.Tp)
            self.Tp -= self.backpropagated_data
        else:
            self.data_fid.subgradient(self.y, out=self.p)
            self.op.adjoint(self.p, out=self.Tp)
        self.h_domain.gram_inv(self.Tp, out=self._domain_work)
        self._domain_work *= -self.tau
        self.x += self._domain_work
        self.penalty.proximal(
            self.x, self.regpar*self.tau, out=self.x, **self.proximal_pars
        )
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        if not self.codomain_free:
            self.op(self.x, out=self.y)

class FISTA(RegSolver):
    r"""The generalized FISTA algorithm for minimization of Tikhonov functionals
    
    .. math::
        \mathcal{S}_{g^{\delta}}(F(f)) + \alpha \mathcal{R}(f).
    
    Gradient steps are performed on the first term, and proximal steps on the
    second term. For a setting without a regularization parameter, FISTA
    minimizes only the data-fidelity term, subject to any permitted box
    constraints. The original penalty supplies the domain Hilbert geometry.
    Unconstrained squared-norm penalties are replaced by a null functional.
    For a Quadratic integral penalty only its box constraints remain: all
    quadratic, linear, and constant coefficients are set to zero. This also
    applies componentwise to FunctionalOnDirectSum penalties. The setting's
    Hilbert geometry is not changed.
    
    Parameters
    ----------
    setting: Setting
        The setting of the forward problem. Includes the penalty and data
        fidelity functionals. If ``setting.regpar`` is ``None``, the setting
        penalty may be of squared-norm type, a
        :class:`~regpy.functionals.base.NullFunctional`, a
        :class:`~regpy.functionals.integral_functionals.Quadratic` with
        optional bound constraints, or a
        :class:`~regpy.functionals.base.FunctionalOnDirectSum` of these.
        It still determines ``setting.h_domain``, but only bound constraints
        remain active in the unregularized objective.
    init: ReadOnlyVector | None [default: None]
        The initial guess.
    tau: float | None [default: None]
        Step size of minimization procedure. In the default case the reciprocal of the operator norm of $T^*T$ is used.
    op_lower_bound: float [default: 0.0]
        lower bound of the operator: :math:`|op(f)|\geq op_lower_bound * |f|`\.
        Used to define convexity parameter of data functional.
    proximal_pars: dict | None [default: None]
        Parameter dictionary passed to the computation of the prox-operator for the penalty term.
    data: Vector | ObsDataT | None [default: None]
        The observed data. Only relevant if ``setting.codomain_free`` is True. 
        Can be an operator-specific handle accepted by ``op.adjoint_data``.
    op_norm: float | None [default: None]
        Norm of the forward operator. If supplied, it is used to compute the
        default step size ``1 / (op_norm**2 * data_fid.Lipschitz)`` instead of
        estimating the norm numerically. For codomain-free settings, the data
        fidelity Lipschitz factor is one. For transformed dual settings the
        operator norm is estimated on the originating primal operator.
        An explicitly supplied ``tau`` takes precedence.
    logging_level: int | str [default: 'INFO']
        logging level
    update_setting: bool [default: False]
        If True, explicitly supplied data and initial guess are stored in the setting.
    is_primal: bool [default: True]
        Whether iterates represent the original primal problem.
        Pass False when solving its dual. This labels coordinates and does not
        change the iteration formulas or construct a dual setting.
    
    estimate_dual_init: bool [default: False]
        If True, initialize the dual estimate on the first call to ``dual()``
        by ``p = -data_fid.subgradient(op(x)) / regpar`` and its adjoint image.
        If False, the initial dual estimate is zero. This affects monitoring
        and stopping at iteration zero, not the primal iteration formulas.
        Requires an explicit codomain.

    Notes
    -----
    In the unregularized mode, the iteration minimizes the data fidelity
    subject to any admissible box constraints from the penalty. No
    quadratic penalty terms remain.
    
    This use as an iterative regularization method is exposed by
    :class:`~regpy.solvers.Setting` under the short method name ``"nFISTA"``
    for applicable linear settings.
    """
    def __init__(
            self,
            setting: Setting,
            init: ReadOnlyVector | None = None,
            tau: float | None = None,
            op_lower_bound: float = 0.,
            proximal_pars: dict | None = None,
            data: Vector | ObsDataT | None = None,
            op_norm: float | None = None,
            logging_level: int | str = "INFO",
            update_setting: bool = False,
                 is_primal: bool = True,
            estimate_dual_init: bool = False,
                 ) -> None:
        if not isinstance(estimate_dual_init, bool):
            raise TypeError('estimate_dual_init must be a bool.')
        if estimate_dual_init and setting.codomain_free:
            raise ValueError('estimate_dual_init requires an explicit codomain.')
        self.estimate_dual_init = estimate_dual_init
        super().__init__(setting, is_primal=is_primal)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="For nonlinear operators the FISTA method in regpy.solvers.nonlinear must be used."))
        if setting.is_tikhonov:
            self.regpar = setting.regpar
            self.penalty = setting.penalty
        else:
            # Retain the original setting's Hilbert geometry, but only keep
            # box constraints in the unregularized proximal term.
            self.regpar = 1.
            self.penalty = _unregularized_penalty(setting.penalty)
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
    def check_applicability(setting: Setting, op_lower_bound: float=0., op_norm: float | None=None) -> tuple[dict[str, Any], dict[str, Any]]:
        r"""Check the requirements for either supported use of linear FISTA.
        
        A Tikhonov setting must provide the proximal map of its penalty.
        Without a regularization parameter, the allowed penalty types are
        NullFunctional, squared-norm-type functionals, Quadratic integral
        functionals with optional bounds, and direct sums of these.
        Quadratic and affine coefficients are discarded, but bounds remain.
        
        Parameters
        ----------
        setting: Setting
            Problem setting containing the operator, penalty, and data fidelity.
        op_lower_bound: float [default: 0.0]
            Op lower bound.
        op_norm: float | None [default: None]
            Operator-norm estimate. If None, use the originating primal
            setting when called on a transformed dual setting.
        
        
        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            The first dictionary contains applicability information, the second parameters used 
            in the method. 
        """
        out = {'info':''}; par={}
        without_regpar = not setting.is_tikhonov
        if without_regpar:
            try:
                _unregularized_penalty(setting.penalty)
            except ValueError as error:
                out['info'] += str(error) + ' '
        if not setting.op.linear:
            out['info'] += 'The forward operator must be linear. '
        if setting.is_tikhonov and not setting.is_convex:
            out['info'] += 'Tikhonov setting must be convex. '
        if without_regpar and not setting.codomain_free and not setting.data_fid.is_convex:
            out['info'] += 'Data fidelity must be convex. '
        if not without_regpar and 'proximal' not in setting.penalty.methods:
            out['info'] += 'Missing prox of penalty functional. '
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
                # Avoid estimating ||T*|| on the transformed dual domain.
                # The norm equals ||T|| when the respective Hilbert spaces
                # are dual to each other, as in Setting.get_dual_setting().
                norm_setting = (setting._primal_setting
                                if setting._primal_setting is not None
                                else setting)
                op_norm = norm_setting.op.norm(
                    norm_setting.h_domain,
                    (None if norm_setting.codomain_free
                     else norm_setting.h_codomain),
                    codomain_free=norm_setting.codomain_free,
                )
            data_lipschitz = (
                1. if setting.codomain_free else setting.data_fid.Lipschitz
            )
            par['tau'] = 1./(op_norm**2 * data_lipschitz)
            par['q'] = (par['tau'] * par['mu']) / (1+par['tau']*par['mu_penalty'])
            out['rate'] = 1.-ma.sqrt(par['q']) 
            if par['mu']>0:
                out['info'] = 'dual ' if setting._primal_setting is not None else ''
                out['info'] += "FISTA used with convexity parameters mu_R={:.3e}, mu_S={:.3e} and step length tau={:.3e}.".format(par['mu_penalty'],par['mu_data_fidelity'],par['tau'])
            else:
                out['info'] = ''
                out['rate'] = -2
        return out, par

    def primal(self) -> tuple[Vector | None, Vector | None]: 
        #if self.codomain_free:
        #    raise RuntimeError(Errors.runtime_error("Primal variables not available when FISTA is run without codomain vectors!"))
        r"""Return the current primal iterate and None for stopping rules and monitoring. 
        (By default the operator image of the current iterate is not computed!)
        
        
        Returns
        -------
        tuple[Vector | None, Vector | None]
            Result described above.
        """
        return (self.x,None)
    
    def dual(self) -> tuple[Vector | None, Vector | None]: 
        r"""Return the current dual iterate and adjoint image for stopping rules and monitoring, when available .
        
        
        Returns
        -------
        tuple[Vector | None, Vector | None]
            Result described above.
        """
        if self.codomain_free:
            raise RuntimeError(Errors.runtime_error("Dual variables not available when FISTA is run without codomain vectors!"))
        if self._dual_variables_computed==False:
            if self.estimate_dual_init and self.iteration_step_nr == 0:
                self.op(self.x, out=self.p)
                self.data_fid.subgradient(self.p, out=self.p)
                self.op.adjoint(self.p, out=self.Tp)
            scale = -1./self.regpar
            self.p *= scale
            self.Tp *= scale
            self._dual_variables_computed = True
        return (self.p,self.Tp)

    def _next(self) -> None:
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
        self.y = None
