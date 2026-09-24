"""Bregman iterations with a configurable convex inner minimizer."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from itertools import repeat
from math import isfinite
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

from ..solver import RegSolver
from ..sequences import Sequence

if TYPE_CHECKING:
    from regpy.vecsps import Vector, ReadOnlyVector
    from regpy.functionals.base import Functional
    from ..setting import Setting

__all__ = ["Bregman"]

_END = object()


class Bregman(RegSolver):
    r"""Minimize successive Bregman-regularized data-fidelity problems.

    Starting at ``setting.init`` (or ``init``) as f_0, outer step n >= 1
    computes an approximate minimizer

    .. math::
        f_n \in \operatorname{argmin}_f
        \left\{\alpha_n^{-1} S(Tf)
        + \Delta_R^{q_{n-1}}(f,f_{n-1})\right\}.

    By default, the first divergence uses ``R.subgradient(f_0)``. For the
    (n+1)th divergence, the choice is

    .. math::
        q_n = T^*p_n, \qquad -\alpha_n p_n \in \partial S(Tf_n),

    using ``S.subgradient(Tf_n)`` and the parameter alpha_n of the completed
    nth solve, not the parameter of the upcoming solve. If S does not
    implement a subgradient, ``R.subgradient(f_n)`` is used instead.
    A custom ``select_subgradient`` overrides this default at every base
    point, including initialization. Each divergence is constructed from
    the original R with ``R.bregman_func``.

    Parameters
    ----------
    setting : regpy.solvers.setting.Setting
        Linear operator and convex penalty and fidelity. A regularization
        parameter is optional when ``regpars`` is supplied. Explicit
        codomain vectors and an explicit data-fidelity functional are required.
        Data already incorporated into the fidelity are retained unchanged.
    regpars : Sequence or iterable of float or None, optional
        Finite or infinite sequence of finite, strictly positive parameters.
        Any iterable is accepted, including RegPy's arithmetic Sequences.
        Values are validated when used. An empty sequence raises ValueError.
        If None, repeat ``setting.regpar`` ``nr_its`` times.
    inner_solver : str or dict, optional
        Convex method abbreviation, default 'FISTA', or a method dictionary
        as stored in ``setting._methods``. Dictionaries must contain 'class'
        and may specify 'primal', 'full', 'method_params', and 'stoprule_params'.
        Executed solvers, stopping histories, and cached applicability results
        are not reused. A string selects from the convex-method registry;
        matching method parameters from the supplied setting are retained.
        Stopping parameters are also retained if that setting is Tikhonov;
        otherwise the inner problem uses convex-optimization stopping defaults.
        Dual methods are supported through Setting.run(), which returns a
        reconstruction in primal coordinates of the supplied setting.
    nr_its : int, optional
        Number of outer steps for a constant parameter, default 2. Must be
        positive. Used only when regpars is None, in which case the setting
        must be a convex Tikhonov setting.
    init : Vector or None, optional
        Initial Bregman base point, default setting.init. It must have finite
        penalty value and admit a subgradient. Initialization does not shift
        the supplied penalty.
    select_subgradient : callable or None, optional
        Called as ``select_subgradient(setting, x)`` with the original
        setting and a read-only view of the current outer iterate. Return a
        dual covector in ``setting.penalty``'s subdifferential at x, using the
        same representation as ``setting.penalty.subgradient(x)`` (not a
        Hilbert gradient). The result is copied and validated by
        ``bregman_func(x, ystar)``. None uses R's default subgradient for the
        first divergence and whenever S has no subgradient implementation;
        otherwise it uses ``T.adjoint(-S.subgradient(Tx) / alpha_previous)``.
        Selection occurs once at initialization and once before each later
        inner solve, never speculatively after the last completed step.
    logging_level : int or str, optional
        Outer logging level, default 'INFO'. Inner logging may be configured
        in the inner method's method_params.
    is_primal : bool, optional
        Coordinate label for compatibility with RegSolver, default True.
        This does not change which inner problem is solved.

    Notes
    -----
    The equivalent inner objective is
    ``S(Tf) + alpha_n * Delta_R(f,f_{n-1})``.
    Each inner run has a fresh Setting and fresh stopping-rule histories.
    ``inner_setting`` and ``inner_method_name`` identify the most recent
    subproblem, whose executed rule is available through ``get_stoprule``.
    Inner stopping tolerances control the accuracy of the outer iteration.

    ``x`` and ``y`` expose the initial pair immediately. Each solver step
    performs one inner solve; inherited iteration yields its result as
    read-only views. Finite sequences end without yielding a duplicate final
    iterate. One parameter is buffered in advance to recognize exhaustion.
    ``regpar`` records the parameter of the most recently completed step
    (the first parameter before any step).

    The original setting, its initialization and its configuration dictionaries
    are not modified. Operator and fidelity objects are shared; do not mutate
    them externally while the solver is running. Inner-method parameters such
    as a data override change the stated subproblem and are rejected. Initial
    guesses configured directly in method_params take precedence over the
    inner setting's default warm start and use that inner method's coordinates.

    Examples
    --------
    >>> solver = Bregman(setting, nr_its=2, inner_solver='PDHG')
    >>> x, y = solver.run()
    >>> from regpy.solvers.sequences import Naturals
    >>> solver = Bregman(setting, regpars=0.1 * 0.5**Naturals(),
    ...                  inner_solver='FISTA')
    >>> x, y = solver.run(max_it=5)
    """

    def __init__(
        self,
        setting: Setting,
        regpars: Sequence | Iterable[float] | None = None,
        inner_solver: str | dict[str, Any] = 'FISTA',
        *,
        nr_its: int = 2,
        init: Vector | None = None,
        select_subgradient: Callable[[Setting, ReadOnlyVector], Vector] | None = None,
        logging_level: int | str = 'INFO',
        is_primal: bool = True,
    ) -> None:
        if select_subgradient is not None and not callable(select_subgradient):
            raise TypeError('select_subgradient must be callable or None.')
        out, _ = self.check_applicability(setting, select_subgradient=select_subgradient)
        if not out['applicable']:
            raise ValueError('Bregman is not applicable. ' + out['info'])
        self.select_subgradient = select_subgradient
        if regpars is None:
            if not setting.is_tikhonov:
                raise ValueError('Supply regpars when the setting has no regularization parameter.')
            if isinstance(nr_its, bool) or not isinstance(nr_its, Integral):
                raise TypeError('nr_its must be a positive integer.')
            if nr_its < 1:
                raise ValueError('nr_its must be positive.')
            regpars = repeat(setting.regpar, int(nr_its))
        self._regpars = iter(regpars)
        self._pending_alpha = next(self._regpars, _END)
        if self._pending_alpha is _END:
            raise ValueError('regpars must contain at least one parameter.')
        alpha = self._validate_alpha(self._pending_alpha)
        self._inner_template = self._method_template(setting, inner_solver)
        initial = setting.init if init is None else init
        if initial not in setting.op.domain:
            raise ValueError('init must belong to the operator domain.')
        super().__init__(setting, is_primal=is_primal)
        self.log.setLevel(logging_level)
        self.regpar = alpha
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, initial)
        # This also validates the base point and subgradient before any solve.
        self._bregman_penalty = self._make_bregman_penalty()
        self.y = self.op(self.x)
        self.inner_setting: Setting | None = None
        self.inner_method_name = '_bregman_inner'

    @staticmethod
    def _validate_alpha(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError('Regularization parameters must be real numbers.')
        alpha = float(value)
        if not isfinite(alpha) or alpha <= 0.:
            raise ValueError('Regularization parameters must be finite and positive.')
        return alpha

    @staticmethod
    def _method_template(setting: Setting, method: str | dict) -> dict:
        from ..setting import get_convex_solver_dictionary

        if isinstance(method, str):
            registry = get_convex_solver_dictionary()
            configured = getattr(setting, '_methods', {}).get(method)
            if configured is not None and setting.is_tikhonov:
                template = configured
            elif method in registry:
                template = dict(registry[method])
                if configured is not None:
                    template['method_params'] = dict(configured.get('method_params', {}))
            else:
                raise ValueError(f'Unknown convex inner method {method!r}.')
        elif isinstance(method, dict):
            template = method
        else:
            raise TypeError('inner_solver must be a method abbreviation or dictionary.')
        cls = template.get('class')
        if not isinstance(cls, type) or not issubclass(cls, RegSolver):
            raise TypeError("The inner method's 'class' must be a RegSolver subclass.")
        if issubclass(cls, Bregman):
            raise ValueError('Bregman itself is not an inner convex minimizer.')
        if not callable(getattr(cls, 'check_applicability', None)):
            raise TypeError('The inner solver must provide check_applicability().')
        primal = template.get('primal', True)
        if not isinstance(primal, bool):
            raise TypeError("The inner method's 'primal' flag must be boolean.")
        params = template.get('method_params', {})
        stopping = template.get('stoprule_params', {})
        if not isinstance(params, dict) or not isinstance(stopping, dict):
            raise TypeError('method_params and stoprule_params must be dictionaries.')
        if params.get('data') is not None or params.get('regpar') is not None:
            raise ValueError('Configure data on the outer setting and parameters through regpars.')
        return dict(
            **{'class': cls, 'primal': primal, 'full': template.get('full', cls.__name__)},
            method_params=dict(params), stoprule_params=deepcopy(stopping),
        )

    @staticmethod
    def check_applicability(
        setting: Setting, op_norm: float | None = None,
        select_subgradient: Callable[[Setting, ReadOnlyVector], Vector] | None = None,
    ) -> tuple[dict, dict]:
        """Check outer requirements; each inner problem checks its own method."""
        info = []
        if not setting.op.linear:
            info.append('The forward operator must be linear.')
        if not setting.penalty.is_convex:
            info.append('The penalty must be convex.')
        if 'eval' not in setting.penalty.methods:
            info.append('The penalty must implement evaluation.')
        if select_subgradient is None and 'subgradient' not in setting.penalty.methods:
            info.append('The penalty must implement a subgradient or a selector must be supplied.')
        if select_subgradient is not None and not callable(select_subgradient):
            info.append('select_subgradient must be callable or None.')
        if setting.codomain_free:
            info.append('An explicit codomain and data-fidelity functional are required.')
        elif not setting.data_fid.is_convex:
            info.append('The data fidelity must be convex.')
        return {'applicable': not info, 'info': ' '.join(info)}, {}

    def _next(self) -> None:
        from ..setting import Setting

        alpha = self._validate_alpha(self._pending_alpha)
        # Reconstruct from the original R, never from the previous divergence.
        if self.iteration_step_nr > 1:
            self._bregman_penalty = self._make_bregman_penalty(self.regpar)
        inner = Setting(
            op=self.op, penalty=self._bregman_penalty,
            data_fid=self.data_fid, regpar=alpha,
            init=self.x, incorporate_init_in_penalty=False,
            explicitDataWarning=False, logging_level=self.log.level,
        )
        template = self._inner_template
        name = self.inner_method_name
        inner.add_method(name, template['class'], primal=template['primal'],
                         full_method_name=template['full'])
        entry = inner._methods[name]
        entry['method_params'].update(template['method_params'])
        entry['stoprule_params'].update(deepcopy(template['stoprule_params']))
        self.inner_setting = inner
        level = entry['method_params'].get('logging_level', self.log.level)
        x, y = inner.run(name, logging_level=level)
        if x not in self.op.domain:
            raise RuntimeError('The inner solver did not return a domain reconstruction.')
        self.op.domain.copyto(self.x, x)
        if y is None:
            self.y = self.op(self.x)
        else:
            self.y = self.op.codomain.empty()
            self.op.codomain.copyto(self.y, y)
        self.regpar = alpha
        self.log.info('Bregman iteration %s: alpha=%s', self.iteration_step_nr, alpha)
        self._pending_alpha = next(self._regpars, _END)
        if self._pending_alpha is _END:
            self.converge()

    def primal(self) -> tuple[Vector, Vector]:
        """Current pair in the coordinates of the supplied setting."""
        return self.x, self.y

    def _make_bregman_penalty(self, previous_alpha: float | None = None) -> Functional:
        """Build at the current iterate using the last completed solve's alpha."""
        if self.select_subgradient is None:
            if previous_alpha is None or 'subgradient' not in self.data_fid.methods:
                return self.penalty.bregman_func(self.x)
            # self.y is T f_n and previous_alpha is alpha_n, even when the
            # next solve uses a different regularization parameter.
            p = self.data_fid.subgradient(self.y)
            p *= (-1. / previous_alpha)
            Tp = self.op.adjoint(p)
            self.penalty.dist_subdiff(Tp,self.x,best_approx=Tp)
            return self.penalty.bregman_func(self.x,Tp)
        subgradient = self.select_subgradient(
            self.setting, self.op.domain.read_only(self.x))
        if subgradient is None:
            raise ValueError('select_subgradient must return a subgradient vector.')
        return self.penalty.bregman_func(self.x, subgradient)
