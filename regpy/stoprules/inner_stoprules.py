"""Stopping rules used inside iterative optimization methods.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np

from regpy.util import Errors
from regpy.vecsps import ReadOnlyVector, Vector
from .general import (
    Schedule,
    LoggingLevel,
    MissingValueError,
    StopRule,
    _copy_vector,
)

if TYPE_CHECKING:
    from regpy.solvers import RegSolver, Solver
    from regpy.solvers.setting import Setting

Norm: TypeAlias = Callable[[ReadOnlyVector], float]
GapPair: TypeAlias = tuple[ReadOnlyVector | None, ReadOnlyVector | None] | None
GapFunctional: TypeAlias = Callable[[GapPair, GapPair], float]
_EVERY_TENTH_STEP = range(0, 10**18, 10)

__all__: list[str] = [
    "RelativeChangeData", "RelativeChangeSol",
    "OptimalityCondStopping", "OptimalityCondStoppingNonConvex",
    "DualityGapStopping",
]


def _refine_penalty_minimum(pen, reference, max_steps: int = 2):
    """Apply at most ``max_steps`` feasible, decreasing Newton steps.

    The functional's Hessian maps primal increments to dual covectors;
    hence ``hessian.inverse(subgradient)`` is the Newton correction. If the
    Hessian inverse is unavailable, or no feasible descent step exists, keep
    the best candidate and leave the usual warning to the caller.
    """
    candidate = pen.domain.empty()
    pen.domain.copyto(candidate, reference)
    zero_covector = pen.domain.zeros(read_only=True)
    for _ in range(max_steps):
        distance = pen.dist_subdiff(zero_covector, candidate)
        if np.isfinite(distance) and distance <= 1e-10:
            break
        try:
            direction = pen.hessian(candidate).inverse(pen.subgradient(candidate))
        except (NotImplementedError, AttributeError, RuntimeError, ValueError,
                ArithmeticError, TypeError) as error:
            pen.log.debug('Newton refinement of penalty reference unavailable: %s', error)
            break
        old_value = pen(candidate)
        accepted = False
        # Backtrack to avoid stepping outside the essential domain or losing
        # the decrease from an already accurate approximate minimizer.
        for scale in (1., 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial = candidate - scale * direction
            try:
                trial_value = pen(trial)
            except (ValueError, ArithmeticError, RuntimeError):
                continue
            if np.isfinite(trial_value) and trial_value <= old_value:
                pen.domain.copyto(candidate, trial)
                accepted = True
                break
        if not accepted:
            break
    return candidate


######### stopping rules for convex optimization problems #########

class OptimalityCondStopping(StopRule):
    r"""Stopping rule based on optimality condition violation.
    
    Parameters
    ----------
    tol: float [default: 0.0]
        Tolerance for the sum of the two optimality-condition residuals.
    logging_level: LoggingLevel [default: 'INFO']
        Logging level.
    schedule: Schedule [default: _EVERY_TENTH_STEP]
        Iterations at which the condition is evaluated. By default it is
        evaluated at iterations ``0, 10, 20, ...``. Pass ``None`` to
        evaluate it at every iteration.
    """
    def __init__(
        self,
        tol: float = 0.,
        logging_level: LoggingLevel = "INFO",
        schedule: Schedule = _EVERY_TENTH_STEP,
    ) -> None:
        super().__init__(
            logging_level=logging_level,
            schedule=schedule,
        )
        self.tol: float = float(tol)
        self.history_dict["dSstar"] = []
        self.history_dict["dR"] = []

    def __repr__(self) -> str:
        return 'OptimalityCondStopping(tol={})'.format(
            self.tol)

    def _complete_init_with_solver(self, solver: Solver) -> None:
        reg_solver = cast("RegSolver", solver)
        if not (reg_solver.setting.is_tikhonov
                and reg_solver.setting.is_convex):
            raise RuntimeError(Errors.generic_message("It is not possible to compute the dual in the implementation of this setting. The setting needs to be convex and contain a regularization parameter!"))
        super()._complete_init_with_solver(solver)

    def _stop(self) -> bool:
        if self.solver is None:
            raise MissingValueError("No solver is attached to this rule.")
        solver = cast("RegSolver", self.solver)
        primal = solver.primal() if hasattr(solver,"primal") and callable(solver.primal) else None
        dual = solver.dual() if hasattr(solver,"dual") and callable(solver.dual) else None
        if primal is None and dual is None:
            raise RuntimeError(Errors.generic_message("The solver needs to provide at least one of the methods 'primal' or 'dual'."))
        dSstar,dR = solver.setting.violation_optimality_cond(primal, dual)
        self.history_dict["dSstar"].append(dSstar)
        self.history_dict["dR"].append(dR)
        if not np.isfinite(dSstar) or not np.isfinite(dR):
            return self._set_stop_status(False, "Nonfinite optimality violation.", failure=True)
        triggered = (dSstar+dR<=self.tol)
        if triggered:
            self._set_stop_status(True, "Optimality violation satisfies the tolerance.")
        self.log_info = '{:.1e} + {:.1e} = {:.2e}  <= {:.1e}'.format(dSstar,dR,dSstar+dR,self.tol)
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)    
        return triggered 

class OptimalityCondStoppingNonConvex(StopRule):
    r"""Stop when the nonconvex first-order optimality violation is small.
    
    This rule evaluates :meth:`Setting.violation_optimality_cond` at the
    solver's current primal iterate and stops when the returned violation is
    at most ``tol``. The solver must provide a callable ``primal`` method.
    
    If the solver has an attribute ``deriv``, it is passed to the setting to
    avoid computing the derivative again. Such an attribute is expected to
    contain the derivative of the forward operator at the *current* iterate;
    a stale derivative makes the stopping criterion invalid.
    
    Parameters
    ----------
    tol: float [default: 0.0]
        Tolerance for the optimality-condition violation.
    logging_level: LoggingLevel [default: 'INFO']
        Logging level.
    schedule: Schedule [default: _EVERY_TENTH_STEP]
        Iterations at which the condition is evaluated. By default it is
        evaluated at iterations ``0, 10, 20, ...``. Pass ``None`` to evaluate
        it at every iteration.
    """

    def __init__(self, tol: float = 0.,
                 logging_level: LoggingLevel = "INFO",
                 schedule: Schedule = _EVERY_TENTH_STEP) -> None:
        super().__init__(logging_level=logging_level, schedule=schedule)
        self.tol: float = float(tol)
        self.history_dict["optimality condition violation"] = []

    def __repr__(self) -> str:
        return 'OptimalityCondStoppingNonConvex(tol={})'.format(self.tol)

    def _complete_init_with_solver(self, solver: Solver) -> None:
        reg_solver = cast("RegSolver", solver)
        if not (reg_solver.setting.is_tikhonov
                and not reg_solver.setting.is_convex):
            raise RuntimeError(Errors.generic_message(
                "This stopping rule requires a nonconvex setting with a "
                "regularization parameter."
            ))
        if not (hasattr(reg_solver, "primal")
                and callable(reg_solver.primal)):
            raise RuntimeError(Errors.generic_message(
                "The solver needs to provide the method 'primal'."
            ))
        super()._complete_init_with_solver(solver)

    def _stop(self) -> bool:
        if self.solver is None:
            raise MissingValueError("No solver is attached to this rule.")
        solver = cast("RegSolver", self.solver)
        primal = solver.primal()
        if primal is None:
            raise RuntimeError(Errors.generic_message(
                "The solver's 'primal' method must return the current iterate."
            ))
        deriv = solver.deriv if hasattr(solver, "deriv") else None
        violation = solver.setting.violation_optimality_cond(
            primal, deriv=deriv
        )
        self.history_dict["optimality condition violation"].append(violation)
        if not np.isfinite(violation):
            return self._set_stop_status(False, "Nonfinite optimality violation.", failure=True)
        triggered = violation <= self.tol
        if triggered:
            self._set_stop_status(True, "First-order optimality violation satisfies the tolerance.")
        self.log_info = '{:.2e} <= {:.1e}'.format(violation, self.tol)
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,
                         self.log_info)
        return triggered
    
class DualityGapStopping(StopRule):
    r"""Stopping rule based on duality gap.
    
    Parameters
    ----------
    tol: float [default: 1e-12]
        The tolerance for the duality gap.
    reltol: float [default: 0.1]
        relative tolerance: Stop if the error in the ``1/rpow``-power of the Bregman distance of the penalty functional 
        is smaller than ``reltol`` times the ``1/rpow``-power of the difference of the penalty functional at the current iterate 
        and the minimum of the penalty functional. The reference always uses
        the original primal penalty, including for methods operating on a dual
        setting. For a dual iterate ``(p, T*p)``, the primal point is recovered
        by the primal setting's ``dual_to_primal`` method; a supplied ``T*p``
        is reused. The minimum is estimated from the primal penalty's reference
        vector, optionally refined by up to two Newton steps for a smooth
        penalty with invertible Hessian. If no minimizer can be verified,
        zero is used as a fallback.
    rpow: int [default: 2]
        Rpow.
    stop_at_infty: bool [default: True]
        Whether an infinite duality gap triggers a nonconverged stop.
        NaN always triggers a nonconverged stop. Nonfinite gaps never certify
        convergence, even if the relative tolerance bound is infinite.
    logging_level: LoggingLevel [default: 'INFO']
        Logging level.
    schedule: Schedule [default: _EVERY_TENTH_STEP]
        Iterations at which the duality gap is evaluated. By default it is
        evaluated at iterations ``0, 10, 20, ...``. Pass ``None`` to evaluate
        it at every iteration. After the solver marks itself converged, its
        final iterate is checked regardless of this schedule. Repeated checks
        of that converged iterate reuse the recorded result.
    additional_gap_functions: list[GapFunctional] | None [default: None]
        Extra functionals called as ``gap(primal, dual)`` using the same
        arguments as the setting's duality gap. Each argument is a pair of
        vectors (entries may be None), or None if unavailable. These values
        are monitored only and do not affect stopping. They are evaluated
        on the same schedule, including the final converged iterate.
        Histories are stored under ``additional_gap_keys``, in list order;
        keys include a one-based index and the callable name when available.
    gap_part: Literal['full', 'data_fid', 'penalty'] [default: 'full']
        ``'full'`` uses the full primal-dual gap as before. ``'data_fid'``
        uses only ``Setting._duality_gap_data_fid`` (the Fenchel/Bregman gap
        for the *entire* data fidelity S), omitting the penalty contribution.
        The latter is a diagnostic stopping heuristic, NOT a certificate of
        primal or dual convergence: even a zero data-fidelity gap does not
        imply ``T* p in partial R(f)``. On this criterion the rule terminates
        with ``converged=False`` and an explanatory stop reason.

        In ``'data_fid'`` mode, the relative bound uses
        ``reltol/(1-reltol) * (S(Tf) - inf S) / alpha`` (no ``rpow`` power).
        ``inf S = -S*(0)`` is evaluated via the conjugate at zero. If this
        value cannot be computed, only the absolute tolerance is used.
        This mode requires an explicit-codomain setting and both the primal
        and dual iterates (or a reconstructible missing iterate).

        ``'penalty'`` evaluates ``solver.setting._duality_gap_pen`` on the
        *active solver setting*: for a transformed dual solver this is its
        dual-domain penalty, not the original primal penalty. Only the current
        dual-setting iterate and the adjoint image of its saddle variable are
        needed. The original primal reconstruction is never formed. In this
        mode the relative bound uses the active penalty and its infimum,
        computed as ``-penalty.conj(0)`` if finite. If the infimum is not
        available, the absolute tolerance is still used.

        Neither partial gap by itself certifies optimality. Accepting a
        partial gap stops with ``converged=False``.

        ``history_dict['duality gap']`` retains the existing plotting key and
        records the *selected* gap. Partial modes also store samples in
        ``history_dict['data fidelity gap']`` or ``['penalty gap']``.
    """
    def __init__(
        self,
        tol: float = 1e-12,
        reltol: float = 1e-1,
        rpow: int = 2,
        stop_at_infty: bool = True,
        logging_level: LoggingLevel = "INFO",
        schedule: Schedule = _EVERY_TENTH_STEP,
        additional_gap_functions: list[GapFunctional] | None = None,
        gap_part: Literal["full", "data_fid", "penalty"] = "full",
    ) -> None:
        if gap_part not in ("full", "data_fid", "penalty"):
            raise ValueError("gap_part must be 'full', 'data_fid', or 'penalty'.")
        self.gap_part = gap_part
        super().__init__(
            logging_level=logging_level,
            schedule=schedule,
        )
        if additional_gap_functions is not None and (
            not isinstance(additional_gap_functions, list)
            or not all(callable(gap) for gap in additional_gap_functions)
        ):
            raise TypeError("additional_gap_functions must be a list of callables or None.")
        self.additional_gap_functions = tuple(additional_gap_functions or ())
        self.additional_gap_keys = tuple(
            f"additional duality gap {index}: {getattr(gap, '__name__', type(gap).__name__)}"
            for index, gap in enumerate(self.additional_gap_functions, start=1)
        )
        for key in self.additional_gap_keys:
            self.history_dict[key] = []
        self.tol: float = float(tol)
        if (reltol>=1):
            raise ValueError('Relative tolerance must be smaller than 1.', reltol)
        self.reltol: float = float(reltol)
 
        self.rpow = rpow
        self.history_dict["duality gap"] = []
        if self.gap_part == "data_fid":
            self.history_dict["data fidelity gap"] = []
        elif self.gap_part == "penalty":
            self.history_dict["penalty gap"] = []
        self.stop_at_infty: bool = stop_at_infty

    def __repr__(self) -> str:
        return (f'DualityGapStopping(tol={self.tol}, reltol={self.reltol}, '
                f'gap_part={self.gap_part!r})')
    
    def _complete_init_with_solver(self, solver: Solver) -> None:
        reg_solver = cast("RegSolver", solver)
        if not (reg_solver.setting.is_tikhonov
                and reg_solver.setting.is_convex):
            raise RuntimeError(Errors.generic_message(
                "Duality-gap stopping requires a convex Tikhonov setting."))
        reference_setting = (reg_solver.setting._primal_setting
                             or reg_solver.setting)
        if self.gap_part == "data_fid" and reference_setting.codomain_free:
            raise ValueError(
                "gap_part='data_fid' requires an explicit data fidelity; "
                "codomain-free settings have no such functional.")
        reg_solver.setting.duality_gap_warning_incr = 1.
        super()._complete_init_with_solver(solver)
        if self.reltol <= 0:
            return

        if self.gap_part == "data_fid":
            # For a proper convex functional, inf S = -S*(0). Unlike an
            # approximate reference_vector this gives an exact normalizer
            # whenever S*(0) can be evaluated. No penalty minimum is needed.
            fidelity = reference_setting.data_fid
            self.minfid = None
            try:
                minfid = -float(fidelity.conj(
                    fidelity.domain.zeros(read_only=True)))
                if np.isfinite(minfid):
                    self.minfid = minfid
            except (NotImplementedError, AttributeError, ValueError,
                    RuntimeError, ArithmeticError, TypeError):
                pass
            if self.minfid is None:
                self.log.warning(
                    "Cannot evaluate inf S = -S*(0). The relative "
                    "data-fidelity-gap threshold is disabled; only the "
                    "absolute tolerance is used.")
            return

        if self.gap_part == "penalty":
            # Crucially use the *active* setting. For dPDHG its penalty is
            # S_primal^*(-alpha * p), not the original (often null) R_primal.
            # Fenchel identity: inf R = -R*(0), when this is finite.
            pen = reg_solver.setting.penalty
            self.minpen = None
            try:
                minpen = -float(pen.conj(pen.domain.zeros(read_only=True)))
                if np.isfinite(minpen):
                    self.minpen = minpen
            except (NotImplementedError, AttributeError, ValueError,
                    RuntimeError, ArithmeticError, TypeError):
                pass
            if self.minpen is None:
                self.log.warning(
                    "Cannot evaluate inf R = -R*(0) in the active setting. "
                    "The relative penalty-gap threshold is disabled; "
                    "only the absolute tolerance is used.")
            return

        pen = reference_setting.penalty
        refsol = pen.reference_vector()
        zero_covector = pen.domain.zeros(read_only=True)
        dist = pen.dist_subdiff(zero_covector, refsol)
        if not np.isfinite(dist) or dist > 1e-14:
            # The reference vector need not minimize the penalty precisely.
            if {'subgradient', 'hessian'} <= pen.methods:
                refsol = _refine_penalty_minimum(pen, refsol, max_steps=2)
                dist = pen.dist_subdiff(zero_covector, refsol)
        if not np.isfinite(dist) or dist > 1e-14:
            self.log.warning(
                'Penalty reference vector is not a verified minimizer '
                '(subdifferential distance: %s). Assuming 0 as infimum '
                'in duality-gap relative error estimate.', dist
            )
            self.minpen = 0.
        else:
            self.minpen = pen(refsol)

    def stop(self) -> bool:
        r"""Always record the converged solver's final gap, even off schedule.
        
        The solver's converge() flag is not a certificate that the gap is
        below tolerance: the final value is evaluated and recorded normally.
        The configured schedule is restored afterwards, including on errors.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        if self.solver is None or not self.solver.is_converged():
            return super().stop()
        step = getattr(self.solver, 'iteration_step_nr', None)
        recorded = self.history_step_dict.get('duality gap', ())
        if step is not None and recorded and recorded[-1] == step:
            return self._last_result
        schedule = self.schedule
        try:
            self.schedule = None
            return super().stop()
        finally:
            self.schedule = schedule

    def _stop(self) -> bool:
        if self.solver is None:
            raise MissingValueError("No solver is attached to this rule.")
        solver = cast("RegSolver", self.solver)
        if (self.gap_part == "penalty"
                and getattr(solver, "x", None) is not None
                and not self.additional_gap_functions):
            # PDHG.primal() evaluates the operator image on demand. The
            # penalty gap needs only x, so avoid an unnecessary evaluation
            # of the physical forward model on every stopping-rule check.
            primal = (solver.x, None)
        else:
            primal = (solver.primal() if hasattr(solver, "primal")
                      and callable(solver.primal) else None)
        dual = (None if solver.setting.codomain_free else
                solver.dual() if hasattr(solver, "dual")
                and callable(solver.dual) else None)
        if primal is None and dual is None:
            raise RuntimeError(Errors.generic_message(
                "The solver must provide 'primal' or 'dual'."))

        if self.gap_part == "full":
            gap = solver.setting.duality_gap(primal=primal, dual=dual)
            reference_setting = (solver.setting._primal_setting
                                 or solver.setting)
        elif self.gap_part == "penalty":
            # In contrast to data_fid mode, do NOT refer to the original
            # primal setting. A dual PDHG solver already supplies x_dual
            # and A_dual* pstar_dual in the correct dual-setting coordinates.
            reference_setting = solver.setting
            if primal is None or primal[0] is None:
                raise ValueError(
                    "Penalty-gap stopping requires a current iterate "
                    "in the active setting's domain.")
            if dual is None or (dual[0] is None and dual[1] is None):
                raise ValueError(
                    "Penalty-gap stopping requires a saddle-point dual "
                    "iterate or its adjoint image.")
            if dual[1] is None:
                dual = (dual[0], reference_setting.op.adjoint(dual[0]))
            gap = reference_setting._duality_gap_pen(
                primal=primal, dual=dual)
        else:
            # Reuse Setting's Bregman term. Complete operator images only
            # if the solver did not supply them. For a solver working on the
            # transformed dual setting, its primal pair is the *original*
            # dual pair; the primal Setting owns the data-fidelity term.
            reference_setting = (solver.setting._primal_setting
                                 or solver.setting)
            if solver.setting._primal_setting is None:
                complete_primal, complete_dual = (
                    reference_setting._complete_primal_dual_tuples(
                        primal=primal, dual=dual))
            else:
                if primal is None:
                    raise ValueError(
                        "Data-fidelity-gap stopping on a dual setting "
                        "requires the solver's dual-coordinate primal pair.")
                complete_primal, complete_dual = (
                    reference_setting._complete_primal_dual_tuples(
                        dual=primal))
            gap = reference_setting._duality_gap_data_fid(
                complete_primal, complete_dual)

        # Evaluate additional monitors before writing any histories.
        additional_gaps = [
            float(functional(primal, dual))
            for functional in self.additional_gap_functions
        ]
        self.history_dict["duality gap"].append(gap)
        if self.gap_part == "data_fid":
            self.history_dict["data fidelity gap"].append(gap)
        elif self.gap_part == "penalty":
            self.history_dict["penalty gap"].append(gap)
        for key, value in zip(self.additional_gap_keys, additional_gaps):
            self.history_dict[key].append(value)

        rel_bound = None
        if self.reltol > 0:
            if self.gap_part == "full":
                if solver.setting._primal_setting is None:
                    prim = primal[0] if isinstance(primal, tuple) else primal
                else:
                    # solver.primal() is (p,T*p) in transformed dual coordinates.
                    pair = primal if isinstance(primal, tuple) else (primal, None)
                    prim = (reference_setting.dual_to_primal(pair)
                            if primal is not None else None)
                if prim is None:
                    raise RuntimeError(Errors.runtime_error(
                        'primal does not provide a domain solution.', primal))
                pen_val = reference_setting.penalty(prim)
                rel_bound = ((self.reltol / (1. - self.reltol)) ** self.rpow
                             * (pen_val - self.minpen))
            elif self.gap_part == "data_fid" and self.minfid is not None:
                # This is a relative *functional-value* normalization, not
                # the penalty-based norm-error bound used in 'full' mode.
                fid_value = reference_setting.data_fid(complete_primal[1])
                fid_excess = max(0., (fid_value - self.minfid)
                                 / reference_setting.regpar)
                rel_bound = self.reltol / (1. - self.reltol) * fid_excess
            elif self.gap_part == "penalty" and self.minpen is not None:
                pen_value = reference_setting.penalty(primal[0])
                pen_excess = max(0., float(pen_value - self.minpen))
                rel_bound = ((self.reltol / (1. - self.reltol)) ** self.rpow
                             * pen_excess)

        finite_gap = np.isfinite(gap)
        accepted = (finite_gap and
                    (gap <= self.tol or
                     (rel_bound is not None and np.isfinite(rel_bound)
                      and gap <= rel_bound)))
        failed = np.isnan(gap) or (np.isinf(gap) and self.stop_at_infty)
        triggered = bool(accepted or failed)
        label = {"full": "Full duality gap",
                 "data_fid": "Data-fidelity Bregman gap",
                 "penalty": "Penalty Bregman gap"}[self.gap_part]
        if failed:
            self.log_info = f"Nonfinite {label.lower()} ({gap})."
            self._set_stop_status(False, self.log_info, failure=True)
        elif accepted:
            if self.gap_part == "full":
                self.log_info = f"Full duality gap ({gap:.2e}) satisfies the tolerance."
                self._set_stop_status(True, self.log_info)
            else:
                unchecked = ("penalty/adjoint" if self.gap_part == "data_fid"
                             else "data-fidelity")
                self.log_info = (
                    f"{label} ({gap:.2e}) satisfies the stopping tolerance; "
                    "full convergence is NOT certified "
                    f"(the {unchecked} optimality condition is unchecked).")
                # An accepted partial gap is a *stopping* decision, not a
                # mathematical convergence certificate.
                self._set_stop_status(False, self.log_info)
        else:
            bound = rel_bound if rel_bound is not None else self.tol
            self.log_info = f'{label.lower()}: {gap:.2e} > {bound:.1e}'
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,
                         self.log_info)
        return triggered


##### general fall-back stopping rules


class RelativeChangeData(StopRule):
    r"""Stop if the relative change in the residual becomes small.
    
    Stops at the first iterate at which the difference between the old residual
    and the new residual is smaller than a pre-determined tol::
    
        ||y_k-y_{k+1}|| < tol max(||y_k||,||y_{k+1}||)
    
    Parameters
    ----------
    norm: Norm | None [default: None]
        The norm with respect to which the difference should be measured.
        By default, ``solver.op.codomain.norm`` is used after a solver is
        attached.
    tol: float [default: 0.0]
        Nonnegative relative tolerance.
    schedule: Schedule | None [default: None]
        Iterations at which the rule is evaluated. If ``None``, it
        is evaluated at every iteration.
    """

    def __init__(
        self,
        norm: Norm | None = None,
        tol: float = 0.,
        schedule: Schedule | None = None,
    ) -> None:
        if not callable(norm) and not norm is None:
            raise TypeError(Errors.type_error("The norm in the relative change of data stopping needs to be a callable or None!"))
        if not isinstance(tol,(int,float)):
            raise TypeError(Errors.type_error("The tol in the relative change of data stopping should be real scalar!"))
        if tol<0:
            raise ValueError(Errors.value_error("The tol in the relative change of data stopping needs to be bigger or equal to zero!"))
        super().__init__(schedule=schedule)
        self.norm: Norm | None = norm
        self.tol: float = float(tol)
        self.data_old: Vector | None = None
        self.history_dict["relative change of y"] = []

    def _complete_init_with_solver(self, solver: Solver) -> None:
        if self.norm is None:
            self.norm = solver.op.codomain.norm
        super()._complete_init_with_solver(solver)   

    def reset(self) -> None:
        r"""Reset the state for a new run.
        """
        super().reset()
        self.data_old = None
        
    def __repr__(self) -> str:
        return 'RelativeChangeData(tol={})'.format(
            self.tol)

    def _stop(self) -> bool:
        if self.solver is None or self.solver.y is None:
            raise MissingValueError
        if self.norm is None:
            raise MissingValueError("No norm is available for data vectors.")
        if self.data_old is None:   
            self.data_old = _copy_vector(self.solver.op.codomain, self.solver.y)
            self.log_info = 'First iteration, no change computed.'
            if self.is_main_rule:
                self.log.debug(self.log_info)
            return False
        change = self.norm(self.solver.y - self.data_old)
        ref = max([self.norm(self.solver.y),self.norm(self.data_old)])
        self.data_old = _copy_vector(self.solver.op.codomain, self.solver.y)
        if not np.isfinite(change) or not np.isfinite(ref):
            return self._set_stop_status(False, "Nonfinite change in y.", failure=True)
        self.history_dict["relative change of y"].append(change/ref if ref else 0.)
        self.log_info = 'rel. data change {:.3e}<{:.3e}'.format(change,self.tol*ref)
        triggered = change < self.tol*ref or change == 0
        if triggered:
            self._set_stop_status(True, "Relative change in x satisfies the tolerance.") or change == 0
        if triggered:
            self._set_stop_status(True, "Relative change in y satisfies the tolerance.")
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)
        return triggered


class RelativeChangeSol(StopRule):
    r"""Stop if the relative change in the solution space becomes small.
    
    Stops at the first iterate at which the difference between the old estimate
    and the new estimate is smaller than a pre-determined tol::
    
        ||x_k-x_{k+1}|| < tol max(||x_k||,||x_{k+1}||)
    
    Parameters
    ----------
    norm: Norm | None [default: None]
        The norm with respect to which the difference should be measured.
        By default, ``solver.op.domain.norm`` is used after a solver is
        attached.
    tol: float [default: 0.0]
        Nonnegative relative tolerance.
    schedule: Schedule | None [default: None]
        Iterations at which the rule is evaluated. If ``None``, it
        is evaluated at every iteration.
    """

    def __init__(
        self,
        norm: Norm | None = None,
        tol: float = 0.,
        schedule: Schedule | None = None,
    ) -> None:
        if not callable(norm) and not norm is None:
            raise TypeError(Errors.type_error("The norm in the relative change of solution stopping needs to be a callable!"))
        if not isinstance(tol,(int,float)):
            raise TypeError(Errors.type_error("The tol in the relative change of solution stopping should be real scalar!"))
        if tol<0:
            raise ValueError(Errors.value_error("The tol in the relative change of solution stopping needs to be larger or equal to zero!"))
        super().__init__(schedule=schedule)
        self.norm: Norm | None = norm
        self.tol: float = float(tol)
        self.sol_old: Vector | None = None
        self.history_dict["relative change of x"] = []

    def _complete_init_with_solver(self, solver: Solver) -> None:
        if self.norm is None:
            self.norm = solver.op.domain.norm
        super()._complete_init_with_solver(solver)   

    def __repr__(self) -> str:
        return 'RelativeChangeSol(tol={})'.format(
            self.tol)
    
    def reset(self) -> None:
        r"""Reset the state for a new run.
        """
        super().reset()
        self.sol_old = None

    def _stop(self) -> bool:
        if self.solver is None or self.solver.x is None:
            raise MissingValueError
        if self.norm is None:
            raise MissingValueError("No norm is available for domain vectors.")
        if self.sol_old is None:   
            self.sol_old = _copy_vector(self.solver.op.domain, self.solver.x)
            self.log_info = 'First iteration, no change computed.'
            if self.is_main_rule:
                self.log.debug(self.log_info)
            return False
        change = self.norm(self.solver.x - self.sol_old)
        ref = max([self.norm(self.solver.x),self.norm(self.sol_old)])
        self.sol_old = _copy_vector(self.solver.op.domain, self.solver.x)
        if not np.isfinite(change) or not np.isfinite(ref):
            return self._set_stop_status(False, "Nonfinite change in x.", failure=True)
        self.history_dict["relative change of x"].append(change/ref if ref else 0.)
        self.log_info = 'rel. change sol: {:.3e}<{:.3e}'.format(change,self.tol*ref)
        triggered = change < self.tol*ref or change == 0
        if triggered:
            self._set_stop_status(True, "Relative change in x satisfies the tolerance.")
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)
        return triggered

from .general import _capped_factory
RelativeChangeData_factory = _capped_factory(RelativeChangeData)
RelativeChangeSol_factory = _capped_factory(RelativeChangeSol)
OptimalityCondStopping_factory = _capped_factory(OptimalityCondStopping)
OptimalityCondStoppingNonConvex_factory = _capped_factory(OptimalityCondStoppingNonConvex)
DualityGapStopping_factory = _capped_factory(DualityGapStopping)
__all__ += ['RelativeChangeData_factory', 'RelativeChangeSol_factory', 'OptimalityCondStopping_factory', 'OptimalityCondStoppingNonConvex_factory', 'DualityGapStopping_factory']
