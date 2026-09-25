from __future__ import annotations
from collections.abc import Callable, Iterable
from math import sqrt
from copy import deepcopy

from regpy.util import Errors
from regpy.operators import Operator
from regpy.vecsps import Vector, ReadOnlyVector

from ..solver import RegSolver
from ..setting import Setting, check_Hilbert_applicability

__all__ = ["NewtonCG", "IterativelyRegularizedNewton", "NonlinBregman"]

class NewtonCG(RegSolver):
    r"""The Newton-CG method. Solves the potentially nonlinear, ill-posed equation:
    
    .. math::
        T(x) = y,

    where :math:`T` is a Fréchet-differentiable operator. The Newton equations are solved by the
    conjugate gradient method applied to the normal equation (CGNE) using the regularizing
    properties of CGNE with early stopping (see Hanke 1997).

    If simplified_op is specified, it will be used to generate an approximation of the derivative 
    of the forward operator setting.op, which may be cheaper to evaluate. E.g., it may be the 
    derivative at the initial guess, which would yield a frozen Newton method. 

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The regularization setting includes the operator and penalty and data fidelity functionals.
    data : Vector or None [default: None]
        Right-hand side in ``setting.op.codomain``. In the default case,
        ``setting.data`` is used.
    init : Vector or None [default: None]
        Initial guess. In the default case, it is taken from the setting.
    cgmaxit : int [default: 50]
        Maximum number of inner CG iterations.
    rho : float [default: 0.8]
        Inner discrepancy factor; must lie in ``(0, 1)``.
    simplified_op : regpy.operators.Operator or None [default: None]
        Optional operator used to approximate the derivative.
    logging_level : int or str [default: "INFO"]
        Logging level.
    """

    def __init__(self, 
                 setting:Setting, 
                 data:Vector | None=None,
                 init:Vector | None=None,
                 cgmaxit:int=50, rho:float=0.8, 
                 simplified_op:Operator | None = None,
                 logging_level:int | str = 'INFO') -> None:
        out, par = self.check_applicability(
            setting, user_data=data, user_init=init
        )
        if not out['applicable']:
            raise ValueError('NewtonCG not applicable to this setting. ' + out['info'])
        super().__init__(setting)
        initial = par['init']
        self.data = par['data']
        if self.data is None:
            raise ValueError(Errors.value_error(
                "Data have to be included in the setting or given directly."
            ))
        """The measured data."""

        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, initial)
        if simplified_op:
            self.simplified_op = simplified_op
            """Simplified operator for derivative.
            """
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y, self.deriv = self.op.linearize(self.x)
        self.rho = rho
        r"""A fix number related to the termination :math:`(0<\rho<1)`."""
        self.cgmaxit = cgmaxit
        """Maximum number of iterations for inner CG solver."""
        self._k = 0
        self.log.setLevel(logging_level)

    @staticmethod
    def check_applicability(
        setting: Setting,
        op_norm: float | None = None,
        user_data: Vector | None = None,
        user_init: Vector | None = None,
    ) -> tuple[dict, dict]:
        """Check whether Newton-CG can be used for ``setting``."""
        out, par = check_Hilbert_applicability(
            setting,
            op_norm=op_norm,
            user_data=user_data,
            user_init=user_init,
        )
        if setting.codomain_free:
            out['info'] += (
                'NewtonCG has no implementation without codomain vectors. '
            )
        out['applicable'] = out['info'] == ''
        return out, par
    
    def _next(self):
        self._k = 0
        self._s = self.data - self.y  
        # aux plays the role of s here to avoid storage for another vector in codomain
        self._x_k = self.op.domain.zeros()
        # self._s += - self.deriv(self._x_k)
        self._s2 = self.h_codomain.gram(self._s)
        self._norms0 = sqrt(self.op.codomain.vdot(self._s2, self._s).real)
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.h_domain.gram_inv(self._rtilde)
        self._d = self._r
        self._inner_prod = self.op.domain.vdot(self._r, self._rtilde).real
     
        while (self._k==0 or (sqrt(self.op.codomain.vdot(self._s2, self._s).real)
               > self.rho * self._norms0 and self._k < self.cgmaxit)):
            self._q = self.deriv(self._d)
            self._q2 = self.h_codomain.gram(self._q)
            self._alpha = self._inner_prod / self.op.codomain.vdot(self._q, self._q2).real
            self._x_k += self._alpha * self._d
            self._s += -self._alpha * self._q
            self._s2 += -self._alpha * self._q2
            self._rtilde = self.deriv.adjoint(self._s2)
            self._r = self.h_domain.gram_inv(self._rtilde)
            self._inner_prod = self.op.domain.vdot(self._r, self._rtilde).real
            self._beta = self.op.domain.vdot(self._r, self._rtilde).real / self._inner_prod
            self._d = self._r + self._beta * self._d
            self._k += 1
        self.log.info('Inner CG iteration required {} steps.'.format(self._k))
        self.x += self._x_k
        if hasattr(self,'simplified_op'):
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y , self.deriv = self.op.linearize(self.x)

    def nr_inner_its(self):
        return self._k

class IterativelyRegularizedNewton(RegSolver):
    r"""General method for iterated linearization of a forward operator :math:`F`
    and subsequent solution of regularized linearized operator equations. 
    In each iteration, an update :math:`h =x_{n+1}-x_n` is computed by minimizing

    .. math::
        S_{data}(F'[x_n]h+F(x_n)) + regpar_{n} \cdot R(h+x_n)

    using a given `inner_solver`.
    :math:`regpar_n` is a decreasing sequence of regularization parameters.

    Parameters
    ----------
    setting : Setting
        Non-Tikhonov setting with explicit codomain vectors.
    inner_solver : str or dict or None, optional
        Convex solver label, method dictionary, or None, as in Bregman.
        None uses ``setting._default_inner_solver``. A label retains the
        constructor and stopping parameters in ``setting._convex_solvers``.
        Dictionaries contain 'class' and may specify 'primal', 'full',
        'method_params', and 'stoprule_params'. Each step creates a fresh
        solver and stopping history. Dual solvers return a primal correction
        through Setting.run().
    data : Vector or None, optional
        Optional replacement data. Otherwise retain the setting's fidelity,
        including data already incorporated into it.
    regpar : float, optional
        First regularization parameter; defaults to 1.
    regpar_step : float, optional
        Geometric factor, default 2/3.
    init : Vector or None, optional
        Initial guess; defaults to setting.init. Copied into solver storage.
    regpars : iterable of float or None, optional
        Explicit sequence of positive parameters, overriding regpar and
        regpar_step. Otherwise use regpar * regpar_step**Naturals().
        Empty sequences raise ValueError; finite sequences stop after their
        last inner solve. Parameter validation and advancement match IRGNM.
    inner_solver_configurator : callable or None, optional
        Called as ``configurator(inner_setting, (x, y), method_dict)``
        immediately before the inner run. x and y are the current outer
        iterate and its nonlinear operator image; treat them as read-only.
        The dictionary is the actual entry in inner_setting._methods.
        Modify its method_params and/or stoprule_params in place; the return
        value is ignored. The inner unknown is h, not x, so initial guesses
        supplied through method_params must be in correction coordinates.
        Runs after applying the inner_solver configuration and may override
        it for the current step. None leaves that configuration unchanged.
    logging_level : int or str, optional
        Outer logging level, default 'INFO'. Configure inner logging through
        inner_solver's method_params or inner_solver_configurator if desired.
    """

    def __init__(
        self, setting: Setting, inner_solver: str | dict | None = None,
        data: Vector | None = None, regpar: float = 1.,
        regpar_step: float = 2 / 3, init: Vector | None = None,
        regpars: Iterable[float] | None = None,
        inner_solver_configurator: Callable[
            [Setting, tuple[Vector, Vector], dict], None
        ] | None = None,
        logging_level: int | str = 'INFO',
    ) -> None:
        from .irgnm import (
            _regularization_parameter_iterator, _next_regularization_parameter,
        )

        from ..linear.bregman import Bregman

        out, _ = IterativelyRegularizedNewton.check_applicability(setting)
        if not out['applicable']:
            raise ValueError('IterativelyRegularizedNewton not applicable. ' + out['info'])
        self._inner_template = Bregman._method_template(setting, inner_solver)
        if issubclass(self._inner_template['class'], IterativelyRegularizedNewton):
            raise ValueError('A Newton outer solver is not an inner convex minimizer.')
        if inner_solver_configurator is not None and not callable(inner_solver_configurator):
            raise TypeError('inner_solver_configurator must be callable or None.')
        initial = setting.init if init is None else init
        if initial not in setting.op.domain:
            raise ValueError('init must belong to the operator domain.')
        if data is not None and data not in setting.op.codomain:
            raise ValueError('data must belong to the operator codomain.')
        super().__init__(setting)
        self.log.setLevel(logging_level)
        self.init = self.op.domain.empty()
        self.op.domain.copyto(self.init, initial)
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, self.init)
        self.data = setting.data if data is None else data
        if data is not None:
            self.data_fid = deepcopy(setting.data_fid).as_data_func(data)
        self._regpar_iterator = _regularization_parameter_iterator(
            regpars, regpar, regpar_step
        )
        try:
            self.regpar = _next_regularization_parameter(self._regpar_iterator)
        except StopIteration as error:
            raise ValueError('regpars must provide at least one regularization parameter.') from error
        self.regpar_step = regpar_step
        self.inner_method_name = '_newton_inner'
        self.inner_setting: Setting | None = None
        self.inner_solver_configurator = inner_solver_configurator
        self._previous_regpar: float | None = None
        self.y, self.deriv = self.op.linearize(self.x)

    def _inner_penalty(self):
        """Penalty in correction coordinates for the current Newton step."""
        return self.penalty.shift(v=-self.x)

    def _next(self) -> None:
        from .irgnm import _next_regularization_parameter

        inner_setting = Setting(
            self.deriv,
            self._inner_penalty(),
            self.data_fid.shift(v=-self.y),
            regpar=self.regpar,
            init=self.op.domain.zeros(),
            incorporate_init_in_penalty=False,
            explicitDataWarning=False, logging_level=self.log.level,
        )
        self.inner_setting = inner_setting
        template = self._inner_template
        name = self.inner_method_name
        inner_setting.add_method(name, template['class'], primal=template['primal'],
                                 full_method_name=template['full'])
        entry = inner_setting._methods[name]
        entry['method_params'].update(template['method_params'])
        entry['stoprule_params'].update(deepcopy(template['stoprule_params']))
        if self.inner_solver_configurator is not None:
            self.inner_solver_configurator(
                inner_setting, (self.op.domain.read_only(self.x),
                                self.op.codomain.read_only(self.y)), entry,
            )
        level = entry['method_params'].get('logging_level', self.log.level)
        step, _ = inner_setting.run(name, logging_level=level)
        if step not in self.op.domain:
            raise RuntimeError('The inner solver did not return a domain correction.')
        self.x += step
        self.y, self.deriv = self.op.linearize(self.x)
        self.log.info('its.%s: alpha=%s', self.iteration_step_nr, self.regpar)
        self._previous_regpar = self.regpar
        try:
            self.regpar = _next_regularization_parameter(self._regpar_iterator)
        except StopIteration:
            self.converge()

    @staticmethod
    def check_applicability(
        setting: Setting, op_norm: float | None = None,
    ) -> tuple[dict, dict]:
        """Check outer requirements; the inner setting checks its own method."""
        out = {'info': '', 'warning': ''}
        if setting.is_tikhonov:
            out['info'] += 'Setting must not have a regularization parameter. '
        if setting.codomain_free:
            out['info'] += 'Explicit codomain vectors are required. '
        out['applicable'] = out['info'] == ''
        return out, {}


class NonlinBregman(IterativelyRegularizedNewton):
    r"""Newton steps regularized by a Bregman divergence of the original R.

    At x_n, solve the convex problem in the correction h = x_{n+1} - x_n:

    .. math::
        \min_h S_{data}(F'[x_n]h + F(x_n))
        + \alpha_n \Delta_R^{q_n}(h+x_n, x_n).

    The first divergence uses R.subgradient(x_0). In analogy to Bregman,
    subsequent steps form the candidate

    .. math::
        \widetilde q_n = -F'[x_n]^* S_{data}'(F(x_n))/\alpha_{n-1},

    and use ``R.dist_subdiff`` to project it onto the subdifferential of R
    at x_n before constructing ``R.bregman_func(x_n, q_n)``. The derivative
    and fidelity argument are evaluated at the current nonlinear iterate;
    the denominator is the parameter of the completed step, not the next
    parameter. If the fidelity has no subgradient implementation, use
    R.subgradient(x_n) instead.

    Parameters are those of IterativelyRegularizedNewton, with the additional
    keyword-only ``select_subgradient``. This optional callable receives
    ``(setting, x)`` with the original setting and a read-only iterate and
    returns a penalty subgradient covector, as in Bregman. It overrides the
    default selection, including at initialization. Selection is performed
    once at initialization and once before each later solve, never after the
    last solve. Each divergence is built from the original penalty.

    The setting must have no regularization parameter, an explicit codomain,
    and convex penalty and fidelity. The forward operator may be nonlinear.
    Inner guesses are in correction coordinates (or the selected dual
    solver's coordinates). The default primal correction guess is zero.
    ``x`` and ``y`` are the current nonlinear iterate and F(x), respectively.
    """

    def __init__(
        self, setting: Setting, inner_solver: str | dict | None = None,
        data: Vector | None = None, regpar: float = 1.,
        regpar_step: float = 2 / 3, init: Vector | None = None,
        regpars: Iterable[float] | None = None,
        inner_solver_configurator: Callable[
            [Setting, tuple[Vector, Vector], dict], None
        ] | None = None,
        logging_level: int | str = 'INFO',
        *,
        select_subgradient: Callable[[Setting, ReadOnlyVector], Vector] | None = None,
    ) -> None:
        if select_subgradient is not None and not callable(select_subgradient):
            raise TypeError('select_subgradient must be callable or None.')
        out, _ = self.check_applicability(setting, select_subgradient=select_subgradient)
        if not out['applicable']:
            raise ValueError('NonlinBregman not applicable. ' + out['info'])
        self.select_subgradient = select_subgradient
        super().__init__(
            setting, inner_solver=inner_solver, data=data, regpar=regpar,
            regpar_step=regpar_step, init=init, regpars=regpars,
            inner_solver_configurator=inner_solver_configurator,
            logging_level=logging_level,
        )
        self.inner_method_name = '_nonlin_bregman_inner'
        self._bregman_penalty = self._make_bregman_penalty()

    @staticmethod
    def check_applicability(
        setting: Setting, op_norm: float | None = None,
        select_subgradient: Callable[[Setting, ReadOnlyVector], Vector] | None = None,
    ) -> tuple[dict, dict]:
        out, par = IterativelyRegularizedNewton.check_applicability(setting, op_norm)
        if not setting.penalty.is_convex:
            out['info'] += 'The penalty must be convex. '
        if 'eval' not in setting.penalty.methods:
            out['info'] += 'The penalty must implement evaluation. '
        if select_subgradient is None and 'subgradient' not in setting.penalty.methods:
            out['info'] += 'The penalty must implement a subgradient or a selector must be supplied. '
        if select_subgradient is not None and not callable(select_subgradient):
            out['info'] += 'select_subgradient must be callable or None. '
        if not setting.codomain_free and not setting.data_fid.is_convex:
            out['info'] += 'The data fidelity must be convex. '
        out['applicable'] = out['info'] == ''
        return out, par

    def _make_bregman_penalty(self):
        if self.select_subgradient is not None:
            subgradient = self.select_subgradient(
                self.setting, self.op.domain.read_only(self.x))
            if subgradient is None:
                raise ValueError('select_subgradient must return a subgradient vector.')
            return self.penalty.bregman_func(self.x, subgradient)
        if self._previous_regpar is None or 'subgradient' not in self.data_fid.methods:
            return self.penalty.bregman_func(self.x)
        p = self.data_fid.subgradient(self.y)
        p *= -1. / self._previous_regpar
        candidate = self.deriv.adjoint(p)
        self.penalty.dist_subdiff(candidate, self.x, best_approx=candidate)
        return self.penalty.bregman_func(self.x, candidate)

    def _inner_penalty(self):
        if self._previous_regpar is not None:
            self._bregman_penalty = self._make_bregman_penalty()
        return self._bregman_penalty.shift(v=-self.x)
