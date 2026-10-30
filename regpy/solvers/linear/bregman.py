"""Bregman iterations with a configurable convex inner minimizer."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from regpy.vecsps.base import ReadOnlyVector, Vector
    from regpy.functionals.base import Functional
    from regpy.solvers.setting import Setting
    from regpy.stoprules import StopRule

from collections.abc import Callable, Iterable
from copy import deepcopy
from itertools import repeat
from inspect import signature
from math import isfinite
from logging import Logger
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

from ..solver import RegSolver
from ..sequences import Sequence


__all__ = ["Bregman", "RunOnAlphaGrid"]

_END = object()


def _log_inner_summary(log: Logger, method_name: str, rule: StopRule) -> None:
    """Report completed inner steps and the recorded evaluation total, if available.

    Search nested combinations without relying on which component triggered.
    CountIterations.iteration includes stop checks, so use the attached solver's
    iteration_step_nr for the number of completed iterations. Prefer a counter
    with evaluation snapshots when several counters are present. Never sum
    snapshots or counters: each total is already cumulative for the inner run.
    """
    from regpy.stoprules import CountIterations

    pending = [rule]
    counters = []
    while pending:
        current = pending.pop()
        if isinstance(current, CountIterations):
            counters.append(current)
        pending.extend(reversed(getattr(current, 'rules', ())))
    if not counters:
        return
    counter = next((r for r in counters if r.eval_counts.get('total')), counters[0])
    if counter.solver is None:
        return
    iterations = counter.solver.iteration_step_nr
    totals = counter.eval_counts.get('total', ())
    if totals:
        log.info('%s stopped after %s its and %s op-evals.',
                 method_name, iterations, totals[-1])
    else:
        log.info('%s stopped after %s its (op-evals not recorded).',
                 method_name, iterations)


def _same_hilbert_geometry(left: object, right: object) -> bool:
    """Hilbert spaces may be copied when constructing successive inner settings.

    Do not reuse a norm just because the vector spaces agree: changing the
    Gram matrices changes the operator norm. RegPy Hilbert spaces implement
    equality under deepcopy; treat unknown equality as a cache miss.
    """
    if left is right:
        return True
    try:
        result = left == right
        return result is True or (hasattr(result, 'item') and result.item() is True)
    except (TypeError, ValueError, AttributeError):
        return False


def _validate_op_norm(op_norm: float | None) -> float | None:
    if op_norm is None:
        return None
    if isinstance(op_norm, bool) or not isinstance(op_norm, Real):
        raise TypeError('op_norm must be a finite, positive real number or None.')
    op_norm = float(op_norm)
    if not isfinite(op_norm) or op_norm <= 0.:
        raise ValueError('op_norm must be finite and strictly positive.')
    return op_norm


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
    setting: Setting
        Linear operator and convex penalty and fidelity. A regularization
        parameter is optional when ``regpars`` is supplied. Codomain-free
        settings are supported for the quadratic fidelity encoded by the
        operator's ``adjoint_eval`` and ``adjoint_data`` operations. Such
        settings must contain data. With an explicit codomain, data already
        incorporated into the fidelity are retained unchanged.
    regpars: Sequence | Iterable[float] | None [default: None]
        Finite or infinite sequence of finite, strictly positive parameters.
        Any iterable is accepted, including RegPy's arithmetic Sequences.
        Values are validated when used. An empty sequence raises ValueError.
        If None, repeat ``setting.regpar`` ``nr_its`` times.
    inner_solver: str | dict[str, Any] | None [default: None]
        Convex method abbreviation, or a method dictionary
        as stored in ``setting._methods`` or None. 
        If None, the default inner solver of the setting is used.
        Dictionaries must contain 'class'
        and may specify 'primal', 'full', 'method_params', and 'stoprule_params'.
        Executed solvers, stopping histories, and cached applicability results
        are not reused. A string selects from the convex-method registry;
        matching method parameters from the supplied setting are retained.
        On settings without a regularization parameter, configuration is
        taken from ``setting._convex_solvers``, including stopping parameters.
        Dual methods are supported through Setting.run(), which returns a
        reconstruction in primal coordinates of the supplied setting.
        Codomain-free settings require a primal inner method that supports
        codomain-free problems, for example TikhonovCG for quadratic penalties.
    nr_its: int [default: 2]
        Number of outer steps for a constant parameter. Must be
        positive. Used only when regpars is None, in which case the setting
        must be a convex Tikhonov setting.
    init: ReadOnlyVector | None [default: None]
        Initial Bregman base point, default setting.init. It must have finite
        penalty value and admit a subgradient. Initialization does not shift
        the supplied penalty.
    select_subgradient: Callable[[Setting, ReadOnlyVector], Vector] | None [default: None]
        Called as ``select_subgradient(setting, x)`` with the original
        setting and a read-only view of the current outer iterate. Return a
        dual covector in ``setting.penalty``'s subdifferential at x, using the
        same representation as ``setting.penalty.subgradient(x)`` (not a
        Hilbert gradient). The result is copied and validated by
        ``bregman_func(x, ystar)``. None uses R's default subgradient for the
        first divergence and whenever S has no subgradient implementation;
        otherwise it uses ``T.adjoint(-S.subgradient(Tx) / alpha_previous)``.
        For codomain-free quadratic fidelity, the candidate is computed as
        ``(T.adjoint_data(data) - T.adjoint_eval(x)) / alpha_previous``.
        As in the explicit-codomain case, this candidate is projected onto
        the penalty subdifferential using ``R.dist_subdiff``.
        Selection occurs once at initialization and once before each later
        inner solve, never speculatively after the last completed step.
    logging_level: int | str [default: 'INFO']
        Outer logging level, including an INFO summary of each inner run when
        CountIterations is present. Inner solver and stopping-rule logging
        default independently to WARNING. Configure them through the inner
        method's method_params and stoprule_params, respectively.
    op_norm: float | None [default: None]
        Optional physical forward-operator norm in the inner Hilbert
        geometries. If omitted, estimate once and cache across inner solves;
        a changed Hilbert geometry invalidates that cache. It is passed only
        to inner solvers whose constructors explicitly accept ``op_norm``.
    is_primal: bool [default: True]
        Coordinate label for compatibility with RegSolver.
        This does not change which inner problem is solved.
    
    Notes
    -----
    The equivalent inner objective is
    ``S(Tf) + alpha_n * Delta_R(f,f_{n-1})``.
    Each inner run has a fresh Setting and fresh stopping-rule histories.
    ``inner_setting`` and ``inner_method_name`` identify the most recent
    subproblem, whose executed rule is available through ``get_stoprule``.
    Inner stopping tolerances control the accuracy of the outer iteration.
    
    ``x`` and ``y`` expose the initial pair immediately. In codomain-free
    mode, ``y`` remains None throughout; neither initialization nor the outer
    iteration evaluates ``T(x)``. The backprojected data are computed once
    for the default subgradient selection. The operator's normal and
    adjoint-data operations must already include the fidelity's Gram factor.
    Each solver step
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
        inner_solver: str | dict[str, Any] | None = None,
        *,
        nr_its: int = 2,
        init: ReadOnlyVector | None = None,
        select_subgradient: Callable[[Setting, ReadOnlyVector], Vector] | None = None,
        logging_level: int | str = 'INFO',
        is_primal: bool = True,
        op_norm: float | None = None,
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
        if setting.codomain_free and not self._inner_template['primal']:
            raise ValueError('Codomain-free Bregman requires a primal inner solver.')
        initial = setting.init if init is None else init
        if initial not in setting.op.domain:
            raise ValueError('init must belong to the operator domain.')
        # The cache belongs to this outer run, not to the shared Operator.
        # It is reused across alpha values only for equal Hilbert geometries.
        self._supplied_op_norm = _validate_op_norm(op_norm)
        self._op_norm_cache: tuple[object, object | None, float] | None = None
        super().__init__(setting, is_primal=is_primal)
        self.log.setLevel(logging_level)
        self.regpar = alpha
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, initial)
        # This also validates the base point and subgradient before any solve.
        self._bregman_penalty = self._make_bregman_penalty()
        if setting.codomain_free:
            self.data = setting.data
            self.y = None
            if self.select_subgradient is None:
                self._backprojected_data = self.op.adjoint_data(self.data)
        else:
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
    def _method_template(setting: Setting, method: str | dict | None) -> dict:
        from ..setting import get_convex_solver_dictionary

        if method is None:
            if setting._default_inner_solver is None:
                raise RuntimeError('Inner solver is neither specified as argument of the constructor nor as default in the setting.')            
            else:
                method = setting._default_inner_solver            
        if isinstance(method, str):
            registry = get_convex_solver_dictionary()
            configured = getattr(setting,
                                 '_methods' if setting.is_tikhonov else '_convex_solvers',
                                 {}).get(method)
            if configured is not None:
                template = configured
            elif method in registry:
                template = dict(registry[method])
            else:
                raise ValueError(f'Unknown convex inner method {method!r}.')
        elif isinstance(method, dict):
            template = method
        else:
            raise TypeError('inner_solver must be a method abbreviation or dictionary.')
        cls = template.get('class')
        if not isinstance(cls, type) or not issubclass(cls, RegSolver):
            raise TypeError("The inner method's 'class' must be a RegSolver subclass.")
        if issubclass(cls, (Bregman, RunOnAlphaGrid)):
            raise ValueError('An outer parameter runner is not an inner convex minimizer.')
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
            raise ValueError('Configure data on the outer setting and regularizaparameters through regpar.')
        return dict(
            **{'class': cls, 'primal': primal, 'full': template.get('full', cls.__name__)},
            method_params=dict(params), stoprule_params=deepcopy(stopping),
        )

    @staticmethod
    def check_applicability(
        setting: Setting, op_norm: float | None = None,
        select_subgradient: Callable[[Setting, ReadOnlyVector], Vector] | None = None,
    ) -> tuple[dict, dict]:
        r"""Check outer requirements; each inner problem checks its own method.
        
        Parameters
        ----------
        setting: Setting
            Problem setting containing the operator, penalty, and data fidelity.
        op_norm: float | None [default: None]
            Operator-norm estimate; None requests numerical estimation.
        select_subgradient: Callable[[Setting, ReadOnlyVector], Vector] | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        tuple[dict, dict]
            Result described above.
        """
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
            if setting.data is None:
                info.append('Codomain-free Bregman requires data in the setting.')
        elif not setting.data_fid.is_convex:
            info.append('The data fidelity must be convex.')
        return {'applicable': not info, 'info': ' '.join(info)}, {}

    def _next(self) -> None:
        alpha = self._validate_alpha(self._pending_alpha)
        # Reconstruct from the original R, never from the previous divergence.
        if self.iteration_step_nr > 1:
            self._bregman_penalty = self._make_bregman_penalty(self.regpar)
        self._run_inner(alpha, self._bregman_penalty, self.x)

    def _cached_inner_op_norm(self, inner: Setting) -> float:
        """Norm for the *inner* Hilbert geometry, evaluated at most once per
        unchanged geometry. No shared operator counters are reset.

        An explicitly supplied norm is assumed valid for all inner problems.
        """
        if self._supplied_op_norm is not None:
            return self._supplied_op_norm
        domain = inner.h_domain
        codomain = None if inner.codomain_free else inner.h_codomain
        cached = self._op_norm_cache
        if (cached is not None
                and _same_hilbert_geometry(domain, cached[0])
                and _same_hilbert_geometry(codomain, cached[1])):
            return cached[2]
        norm = _validate_op_norm(inner.op.norm(
            domain, codomain, codomain_free=inner.codomain_free))
        self._op_norm_cache = (domain, codomain, norm)
        return norm

    def _run_inner(self, alpha: float, penalty: Functional, init: ReadOnlyVector) -> None:
        r"""Solve at the zero linearization point and advance the alpha iterator.

        Since the outer operator is linear, x0=y0=0 and deriv=setting.op
        recover the original data term. The explicit penalty is copied without
        a shift. This routine is shared with RunOnAlphaGrid, which also needs
        inner-method configuration, warm starts and parameter advancement.
        
        Parameters
        ----------
        alpha: float
            Alpha.
        penalty: Functional
            Penalty.
        init: ReadOnlyVector
            Initial iterate in the operator domain.
        """
        zero = self.op.domain.zeros(read_only=True)
        inner = self.setting.get_linearized_tikh_setting(
            x0=zero, regpar=alpha, deriv=self.setting.op,
            y0=None if self.setting.codomain_free else self.op.codomain.zeros(),
            # The codomain-free representation of y0=0 avoids allocating a
            # codomain vector or invoking an explicit adjoint.
            adjoint_y0=zero if self.setting.codomain_free else None,
            inner_penalty=penalty,
        )
        inner.init = init
        template = self._inner_template
        name = self.inner_method_name
        inner.add_method(name, template['class'], primal=template['primal'],
                         full_method_name=template['full'])
        entry = inner._methods[name]
        entry['method_params'].update(template['method_params'])
        entry['stoprule_params'].update(deepcopy(template['stoprule_params']))
        self.inner_setting = inner
        entry['stoprule_params'].setdefault('logging_level', 'WARNING')
        level = entry['method_params'].get('logging_level', 'WARNING')

        # All convex inner solvers can *be checked* with a common norm, but
        # not every constructor accepts op_norm (e.g. TikhonovCG or ADMM).
        # Supplying it to evaluate_methods avoids that method's unconditional
        # fresh estimation for every alpha. Pass it to the constructor only
        # when the constructor explicitly declares the parameter.
        configured_norm = entry['method_params'].get('op_norm')
        if configured_norm is not None:
            norm = _validate_op_norm(configured_norm)
        else:
            norm = self._cached_inner_op_norm(inner)
        inner.evaluate_methods([name], op_norm=norm)
        if 'op_norm' in signature(template['class'].__init__).parameters:
            entry['method_params'].setdefault('op_norm', norm)
        else:
            # Still use the norm for applicability/rates; never forward an
            # unsupported keyword to the solver constructor.
            entry['method_params'].pop('op_norm', None)
        x, y = inner.run(name, logging_level=level)
        _log_inner_summary(self.log, template['class'].__name__, inner.get_stoprule(name))
        if x not in self.op.domain:
            raise RuntimeError('The inner solver did not return a domain reconstruction.')
        self.op.domain.copyto(self.x, x)
        if self.setting.codomain_free:
            self.y = None
        elif y is None:
            self.y = self.op(self.x)
        else:
            self.y = self.op.codomain.empty()
            self.op.codomain.copyto(self.y, y)
        self.regpar = alpha
        self.log.info('%s iteration %s: alpha=%s',
                      type(self).__name__, self.iteration_step_nr, alpha)
        self._pending_alpha = next(self._regpars, _END)
        if self._pending_alpha is _END:
            self.converge()

    def primal(self) -> tuple[Vector, Vector | None]:
        r"""Current pair in the coordinates of the supplied setting.
        
        
        Returns
        -------
        tuple[Vector, Vector | None]
            Current reconstruction and operator image, or None for the
            image in codomain-free mode.
        """
        return self.x, self.y

    def _make_bregman_penalty(self, previous_alpha: float | None = None) -> Functional:
        r"""Build at the current iterate using the last completed solve's alpha.
        
        Parameters
        ----------
        previous_alpha: float | None [default: None]
            Previous alpha.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        if self.select_subgradient is None:
            if previous_alpha is None:
                return self.penalty.bregman_func(self.x)
            if self.setting.codomain_free:
                # Both terms are domain covectors, including the fidelity
                # Gram factor. Never form a vector in the operator codomain.
                Tp = self.op.domain.empty()
                self.op.domain.copyto(Tp, self._backprojected_data)
                Tp -= self.op.adjoint_eval(self.x)
                Tp /= previous_alpha
            else:
                if 'subgradient' not in self.data_fid.methods:
                    return self.penalty.bregman_func(self.x)
                # Use the parameter of the completed solve, not the next one.
                p = self.data_fid.subgradient(self.y)
                p *= (-1. / previous_alpha)
                Tp = self.op.adjoint(p)
            self.penalty.dist_subdiff(Tp, self.x, best_approx=Tp)
            return self.penalty.bregman_func(self.x, Tp)
        subgradient = self.select_subgradient(
            self.setting, self.op.domain.read_only(self.x))
        if subgradient is None:
            raise ValueError('select_subgradient must return a subgradient vector.')
        return self.penalty.bregman_func(self.x, subgradient)


class RunOnAlphaGrid(RegSolver):
    r"""Solve convex Tikhonov problems on a parameter grid with warm starts.
    
    Each step minimizes ``S(Tf) + alpha * R(f)`` using a fresh inner setting,
    solver, and stopping history. The original penalty is used at every step.
    The first solve starts at ``init`` (or ``setting.init``); each subsequent
    solve starts at a copy of the preceding reconstruction.
    
    This method is used to select the regularization parameter by a parameter choice rule. 
    
    Parameters
    ----------
    setting: Setting
        Must have no regularization parameter, a linear operator, convex
        penalty and fidelity (or codomain-free quadratic fidelity).
    regpars: Sequence | Iterable[float]
        Nonempty, finite or infinite iterable of finite positive parameters.
        Parameters are validated lazily, as in Bregman.
    inner_solver: str | dict[str, Any] | None [default: None]
        Convex solver label or method dictionary or None, as in
        Bregman. Labels retain configuration from ``setting._convex_solvers``.
    init: ReadOnlyVector | None [default: None]
        Initial guess for the first solve, default ``setting.init``.
        Later solves use the preceding reconstruction as their initial guess.
        An explicit inner method initialization takes precedence.
    logging_level: int | str [default: 'INFO']
        Outer logging level, including an INFO summary of each inner run when
        CountIterations is present. Inner solver and stopping-rule logging
        default to WARNING; explicit inner configuration takes precedence.
    op_norm: float | None [default: None]
        Optional physical operator norm, cached as in :class:`Bregman`.
        An inner method's configured ``op_norm`` takes precedence.
    is_primal: bool [default: True]
        Coordinate label for RegSolver.
    
    Notes
    -----
    Iteration yields one result per parameter; ``run()`` returns the last pair.
    ``regpar``, ``inner_setting`` and ``inner_method_name`` have the same meaning
    as in Bregman. No penalty subgradient or Bregman divergence is required.
    The supplied setting and its configuration are not modified.
    """

    _validate_alpha = staticmethod(Bregman._validate_alpha)
    _method_template = staticmethod(Bregman._method_template)
    _cached_inner_op_norm = Bregman._cached_inner_op_norm
    _run_inner = Bregman._run_inner
    primal = Bregman.primal

    def __init__(
        self,
        setting: Setting,
        regpars: Sequence | Iterable[float],
        inner_solver: str | dict[str, Any] | None = None,
        *,
        init: ReadOnlyVector | None = None,
        logging_level: int | str = 'INFO',
        is_primal: bool = True,
        op_norm: float | None = None,
    ) -> None:
        out, _ = self.check_applicability(setting)
        if not out['applicable']:
            raise ValueError('RunOnAlphaGrid is not applicable. ' + out['info'])
        self._regpars = iter(regpars)
        self._pending_alpha = next(self._regpars, _END)
        if self._pending_alpha is _END:
            raise ValueError('regpars must contain at least one parameter.')
        alpha = self._validate_alpha(self._pending_alpha)
        self._inner_template = self._method_template(setting, inner_solver)
        if setting.codomain_free and not self._inner_template['primal']:
            raise ValueError('Codomain-free RunOnAlphaGrid requires a primal inner solver.')
        initial = setting.init if init is None else init
        if initial not in setting.op.domain:
            raise ValueError('init must belong to the operator domain.')
        # The cache belongs to this outer run, not to the shared Operator.
        # It is reused across alpha values only for equal Hilbert geometries.
        self._supplied_op_norm = _validate_op_norm(op_norm)
        self._op_norm_cache: tuple[object, object | None, float] | None = None
        super().__init__(setting, is_primal=is_primal)
        self.log.setLevel(logging_level)
        self.regpar = alpha
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, initial)
        self.y = None if setting.codomain_free else self.op(self.x)
        self.inner_setting: Setting | None = None
        self.inner_method_name = '_alpha_grid_inner'

    @staticmethod
    def check_applicability(
        setting: Setting, op_norm: float | None = None,
    ) -> tuple[dict, dict]:
        r"""Check convexity and absence of a fixed regularization parameter.
        
        Parameters
        ----------
        setting: Setting
            Problem setting containing the operator, penalty, and data fidelity.
        op_norm: float | None [default: None]
            Operator-norm estimate; None requests numerical estimation.
        
        
        Returns
        -------
        tuple[dict, dict]
            Result described above.
        """
        info = []
        if setting.is_tikhonov:
            info.append('The setting must have no regularization parameter.')
        if not setting.op.linear:
            info.append('The forward operator must be linear.')
        if not setting.penalty.is_convex:
            info.append('The penalty must be convex.')
        if setting.codomain_free:
            if setting.data is None:
                info.append('Codomain-free RunOnAlphaGrid requires data in the setting.')
        elif not setting.data_fid.is_convex:
            info.append('The data fidelity must be convex.')
        return {'applicable': not info, 'info': ' '.join(info)}, {}

    def _next(self) -> None:
        alpha = self._validate_alpha(self._pending_alpha)
        # Warm-start from the last reconstruction without sharing writable
        # storage with the outer iterate. The penalty remains unchanged.
        initial = self.op.domain.empty()
        self.op.domain.copyto(initial, self.x)
        self._run_inner(alpha, self.penalty, initial)
