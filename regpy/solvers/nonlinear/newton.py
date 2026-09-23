from __future__ import annotations
from collections.abc import Callable, Iterable
from math import sqrt
from copy import deepcopy,copy

from regpy.stoprules import CountIterations, StopRule
from regpy.util import Errors
from regpy.operators import Operator
from regpy.vecsps import Vector

from ..solver import RegSolver
from ..setting import Setting, check_Hilbert_applicability
from ..linear import SemiSmoothNewton

__all__ = ["NewtonCG","IterativelyRegularizedNewton"]

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
    inner_method : str, optional
        Inner method's registry abbreviation, e.g. 'FISTA' or 'SSN'.
        Defaults to 'FISTA'.
    inner_stoprule : callable or None, optional
        Factory ``factory(inner_setting, params) -> StopRule``. None keeps
        the inner setting's default factory.
    inner_params : dict or None, optional
        Updates to the inner method's constructor parameters.
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
        Mutually exclusive with non-None inner_params or inner_stoprule.
        If omitted, an internal configurator applies those user parameters.
    logging_level : int or str, optional
        Outer logging level, default 'INFO'. Configure inner logging through
        inner_params or inner_solver_configurator if desired.
    """

    def __init__(
        self, setting: Setting, inner_method: str = 'FISTA',
        inner_stoprule: Callable[[Setting, dict], StopRule] | None = None,
        inner_params: dict | None = None,
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

        out, _ = self.check_applicability(setting)
        if not out['applicable']:
            raise ValueError('IterativelyRegularizedNewton not applicable. ' + out['info'])
        if not isinstance(inner_method, str):
            raise TypeError('inner_method must be a method abbreviation string.')
        if inner_params is not None and not isinstance(inner_params, dict):
            raise TypeError('inner_params must be a dict or None.')
        if inner_stoprule is not None and not callable(inner_stoprule):
            raise TypeError('inner_stoprule must be a factory or None.')
        if inner_solver_configurator is not None:
            if not callable(inner_solver_configurator):
                raise TypeError('inner_solver_configurator must be callable or None.')
            if inner_params is not None or inner_stoprule is not None:
                raise ValueError('inner_solver_configurator cannot be combined with '
                                 'inner_params or inner_stoprule.')
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
        self.inner_method = inner_method
        self.inner_params = {} if inner_params is None else dict(inner_params)
        self.inner_stoprule = inner_stoprule
        if inner_solver_configurator is None:
            def configure(inner_setting, iterate, method_dict):
                method_dict.setdefault('method_params', {}).update(self.inner_params)
                if self.inner_stoprule is not None:
                    method_dict.setdefault('stoprule_params', {})['factory'] = self.inner_stoprule
            inner_solver_configurator = configure
        self.inner_solver_configurator = inner_solver_configurator
        self.y, self.deriv = self.op.linearize(self.x)

    def _next(self) -> None:
        from .irgnm import _next_regularization_parameter

        inner_setting = Setting(
            self.deriv,
            self.setting.penalty.shift(v=-self.x),
            self.data_fid.shift(v=-self.y),
            regpar=self.regpar,
            init=self.op.domain.zeros(),
            incorporate_init_in_penalty=False
        )
        self.inner_setting = inner_setting
        if self.inner_method not in inner_setting._methods:
            raise ValueError(f'Unknown inner method {self.inner_method!r}.')
        self.inner_solver_configurator(
            inner_setting, (self.x, self.y),
            inner_setting._methods[self.inner_method],
        )
        step, _ = inner_setting.run(self.inner_method)
        self.x += step
        self.y, self.deriv = self.op.linearize(self.x)
        self.log.info('its.%s: alpha=%s', self.iteration_step_nr, self.regpar)
        try:
            self.regpar = _next_regularization_parameter(self._regpar_iterator)
        except StopIteration:
            self.converge()

    @staticmethod
    def check_applicability(setting: Setting) -> tuple[dict, dict]:
        """Check outer requirements; the inner setting checks its own method."""
        out = {'info': '', 'warning': ''}
        if setting.is_tikhonov:
            out['info'] += 'Setting must not have a regularization parameter. '
        if setting.codomain_free:
            out['info'] += 'Explicit codomain vectors are required. '
        out['applicable'] = out['info'] == ''
        return out, {}
