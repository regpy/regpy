"""General stopping-rule infrastructure.

This module contains the base class, rule combinators, and the iteration-count rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from regpy.operators.base import Operator
    from regpy.solvers.setting import Setting
from copy import deepcopy
from numpy import sqrt
import logging
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import tee
from numbers import Integral
from types import NotImplementedType
from typing import TYPE_CHECKING, TypeAlias, cast

from typing_extensions import Self

from regpy.util import ClassLogger, Errors
from regpy.vecsps import ReadOnlyVector, Vector, VectorSpaceBase

if TYPE_CHECKING:
    from regpy.solvers import Solver

LoggingLevel: TypeAlias = int | str
Schedule: TypeAlias = Iterable[int] | None
Iterate: TypeAlias = tuple[ReadOnlyVector, ReadOnlyVector | None]
BestIterate: TypeAlias = Iterate | Vector
Distance: TypeAlias = Callable[
    [tuple[ReadOnlyVector, ReadOnlyVector]], float
]

__all__: list[str] = [
    "LoggingLevel", "Schedule", "MissingValueError", "StopRule",
    "NoneRule", "CombineRules", "AndCombineRules", "CountIterations",
    "ErrorMonitor",
]

class MissingValueError(Exception):
    """Indicates that a stopping rule needs an unavailable solver value."""

    pass


def _copy_vector(
    space: VectorSpaceBase, vector: ReadOnlyVector
) -> Vector:
    r"""Return a writable copy of ``vector`` allocated in ``space``.
    
    Parameters
    ----------
    space: VectorSpaceBase
        Space.
    vector: ReadOnlyVector
        Input vector; its coefficients are only read.
    
    
    Returns
    -------
    Vector
        Writable result vector.
    """
    result = space.empty()
    space.copyto(result, vector)
    return result


class _ApplicationSchedule(Iterable[int]):
    r"""Lazily validate and replay the steps of an ``schedule`` iterable.
    
    Parameters
    ----------
    values: Iterable[int]
        Values.
    """

    def __init__(self, values: Iterable[int]) -> None:
        try:
            self._source: Iterator[int] = iter(values)
        except TypeError as error:
            raise TypeError(Errors.type_error(
                "schedule must be None or an iterable of integers. "
                f"Got {values!r}."
            )) from error
        self._cache: list[int] = []
        self._cursor: int = 0
        self._exhausted: bool = False
        self._last_query_step: int | None = None
        self._last_query_result: bool = False
        # Validate the first value eagerly. Further values of a possibly
        # infinite iterator are validated when they are needed.
        self._ensure_cached(0)

    def _ensure_cached(self, index: int) -> bool:
        while len(self._cache) <= index and not self._exhausted:
            try:
                value = next(self._source)
            except StopIteration:
                self._exhausted = True
                break
            if isinstance(value, (bool,)) or not isinstance(value, Integral):
                raise TypeError(Errors.type_error(
                    "schedule must contain integers. "
                    f"Got {value!r}."
                ))
            value = int(value)
            if value < 0:
                raise ValueError(Errors.value_error(
                    "schedule must contain nonnegative integers. "
                    f"Got {value}."
                ))
            if self._cache and value <= self._cache[-1]:
                raise ValueError(Errors.value_error(
                    "schedule must be strictly increasing. "
                    f"Got consecutive values {self._cache[-1]} and {value}."
                ))
            self._cache.append(value)
        return len(self._cache) > index

    def contains(self, step: int) -> bool:
        r"""Return whether ``step`` is the next scheduled application step.
        
        Parameters
        ----------
        step: int
            Step.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        # A combined rule may retry a component after computing a missing
        # operator value. Such a retry belongs to the same iteration and must
        # not consume the next scheduled value.
        if step == self._last_query_step:
            return self._last_query_result
        while self._ensure_cached(self._cursor):
            selected_step = self._cache[self._cursor]
            if selected_step < step:
                self._cursor += 1
                continue
            if selected_step == step:
                self._cursor += 1
                self._last_query_step = step
                self._last_query_result = True
                return self._last_query_result
            break
        self._last_query_step = step
        self._last_query_result = False
        return self._last_query_result

    def reset(self) -> None:
        r"""Replay the schedule from its beginning.
        """
        self._cursor = 0
        self._last_query_step = None
        self._last_query_result = False

    def __iter__(self) -> Iterator[int]:
        index = 0
        while self._ensure_cached(index):
            yield self._cache[index]
            index += 1

    def __deepcopy__(self, memo: dict[int, object]) -> _ApplicationSchedule:
        result = object.__new__(type(self))
        memo[id(self)] = result
        result._cache = self._cache.copy()
        result._cursor = self._cursor
        result._exhausted = self._exhausted
        result._last_query_step = self._last_query_step
        result._last_query_result = self._last_query_result
        if self._exhausted:
            result._source = iter(())
        else:
            self._source, result._source = tee(self._source)
        return result

class StopRule:
    r"""Abstract base class for stopping rules.
    
    A solver attaches itself with :meth:`_complete_init_with_solver` before the
    iteration starts. :meth:`stop` then delegates the decision to
    :meth:`_stop`. If this is the main rule, a positive decision stores the
    current iterate through :meth:`trigger`.
    
    Parameters
    ----------
    logging_level: LoggingLevel [default: 'INFO']
        Logging level.
    schedule: Schedule | None [default: None]
        Strictly increasing, nonnegative iteration numbers at which the rule
        is evaluated. Iterators, including infinite iterators, are consumed
        lazily. On other iterations the most recent decision is reused. If
        ``None``, the rule is evaluated on every call.
        This is useful for expensive stopping criteria that do not need to be checked on every iteration.
    """

    log = ClassLogger()

    solver: Solver | None
    x: ReadOnlyVector
    y: ReadOnlyVector | None

    def __init__(
        self,
        logging_level: LoggingLevel = "INFO",
        schedule: Schedule | None = None,
    ) -> None:
        self.solver: Solver | None = None

        self.schedule: _ApplicationSchedule | None = (
            None if schedule is None else _ApplicationSchedule(schedule)
        )
        """Schedule restricting the iterations on which the rule is applied."""
        self._stop_call_nr: int = 0
        self._currently_applied: bool = True
        self._last_result: bool = False
        self._log_current_step: bool = True
        self._current_step: int = 0

        self.triggered: bool = False
        """Whether the stoprule decided to stop."""
        self.history_dict: dict[str, list[float]] = {}
        """A place to save scalars for later use/analysis. An entry of the form {"parameter_name":[]} needs to be added in the implementation of the stoprule."""
        self.history_step_dict: dict[str, list[int]] = {}
        """Iteration numbers corresponding to entries in ``history_dict``."""
        self.is_main_rule: bool = True
        r"""Whether this is the main stoprule of the solver or a sub-rule used for example in a combined rule."""
        self.log_info: str = ""
        self.log.setLevel(logging_level)

    def _complete_init_with_solver(self, solver: Solver) -> None:
        r"""Attach a solver and complete rule-specific initialization.
        
        Subclasses may override this hook when their criterion needs
        information from the concrete solver.
        
        Parameters
        ----------
        solver: Solver
            Solver to which this rule is attached.
        """
        if self.solver is not None:
            self.log.warning("the solver was already set and is now overwritten")
        self.solver = solver

    def copy_and_reset(self) -> Self:
        r"""Return an independent copy reset to its initial state.
        
        
        Returns
        -------
        Self
            Result described above.
        """
        rule = self.copy()
        rule.reset()
        return rule
    
    def reset(self) -> None:
        r"""Reset the rule to its initial state.
        
        Subclasses that maintain additional state must override this method and
        reset that state as well.
        """
        self.solver = None
        self.triggered = False
        self._stop_call_nr = 0
        self._currently_applied = True
        self._last_result = False
        self._log_current_step = True
        self._current_step = 0
        if self.schedule is not None:
            self.schedule.reset()
        if hasattr(self,"x"):
            del self.x
        if hasattr(self,"y"):
            del self.y

        for key in self.history_dict.keys():
            self.history_dict[key] = []
        for key in self.history_step_dict.keys():
            self.history_step_dict[key] = []

    def copy(self) -> Self:
        r"""Copy.
        
        
        Returns
        -------
        Self
            Result described above.
        """
        return cast(Self, deepcopy(self))

    def stop(self) -> bool:
        r"""Check whether to stop iterations.
        
        If :attr:`schedule` is set, :meth:`_stop` is called only at the
        selected solver iteration numbers. Between these iterations the last
        decision is returned without recomputing the criterion. A subclass
        may override :meth:`_skip` to update inexpensive internal state on
        skipped iterations.
        
        Returns
        -------
        bool
            ``True`` if iterations should be stopped.
        """
        solver_step = getattr(self.solver, "iteration_step_nr", None)
        if (isinstance(solver_step, Integral)
                and not isinstance(solver_step, (bool,))
                and solver_step >= 0):
            step = int(solver_step)
        else:
            step = self._stop_call_nr
        self._stop_call_nr += 1
        self._current_step = step
        self._currently_applied = (
            self.schedule is None or self.schedule.contains(step)
        )
        self._log_current_step = self._currently_applied

        if self._currently_applied:
            old_history_lengths = {
                key: len(history)
                for key, history in self.history_dict.items()
            }
            should_stop = self._stop()
            self._last_result = should_stop
            for key, history in self.history_dict.items():
                number_new = len(history) - old_history_lengths.get(key, 0)
                if number_new > 0:
                    self.history_step_dict.setdefault(key, []).extend(
                        [step] * number_new
                    )
        else:
            self._skip()
            should_stop = self._last_result
        if should_stop and self.is_main_rule:
            self.trigger()
        return should_stop

    def _skip(self) -> None:
        r"""Update state when the rule is not evaluated at the current step.
        
        The default implementation does nothing. Subclasses may override this
        hook for inexpensive bookkeeping that must happen on every iteration.
        """

    @property
    def log_current_step(self) -> bool:
        r"""Whether a combined rule should log this component at this step.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self._log_current_step

    def trigger(self) -> None:
        r"""Force the stoprule to trigger at the current iterate.
        
        This sets the :attr:`triggered` attribute to ``True`` and stores
        the current iterate in :attr:`x` and :attr:`y`.
        """
        if self.solver is None or self.solver.x is None:
            raise MissingValueError(
                "The stopping rule cannot trigger without a current iterate."
            )
        self.triggered = True
        self.x = self.solver.x
        self.y = self.solver.y if hasattr(self.solver,"y") else None

    def best_iterate(self) -> BestIterate | None:
        r"""Return the best iterate selected by this stopping rule.
        
        The default implementation returns the last iterate available when
        the rule triggered or the solver converged. Parameter-choice rules may
        override this method and return only the selected solution vector.
        
        Returns
        -------
        BestIterate | None
            The selected solution and, for ordinary stopping rules, its
            operator value if available. ``None`` is returned if neither the
            solver converged nor the rule triggered.
        
        Raises
        ------
        RuntimeError
            If a result should be available but the rule did not store a
            solution vector.
        """
        if (self.solver is not None and self.solver.is_converged()
                and self.solver.x is not None):
            self.x = self.solver.x
            self.y = self.solver.y if hasattr(self.solver,"y") else None
        elif not self.triggered:
            self.log.warning("The stoprule has not triggered yet and the solver has not converged, so no best iterate is available!")
            return None
        if not hasattr(self,"x"):
            raise RuntimeError(Errors.generic_message("The stoprule did not store self.x when triggered! Please re-implement the best_iterate method of the stoprule!"))
        if not hasattr(self,"y"):
            self.y = None
        return self.x, self.y

    def _stop(self) -> bool:
        r"""Check whether to stop iterations.
        
        This is an abstract method. Child classes should override it.
        
        Parameters and return values are the same as for the public interface
        method :meth:`stop`.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        raise NotImplementedError

    def __add__(self, other: StopRule) -> CombineRules:
        return CombineRules([self, other])
    
    def __radd__(self, other: object) -> StopRule | NotImplementedType:
        # Adding zero should not change the rule, this allows to use sum() to combine rules.
        if other == 0:
            return self
        return NotImplemented

    def __or__(self, other: StopRule) -> CombineRules:
        return CombineRules([self, other])

    def __and__(self, other: StopRule) -> AndCombineRules:
        return AndCombineRules([self, other])

class NoneRule(StopRule):
    r"""Default rule that never stops an iteration.
    
    This rule is intended for solvers that terminate by marking themselves as
    converged rather than through an external stopping criterion.
    
    Parameters
    ----------
    schedule: Schedule | None [default: None]
        Schedule.
    """

    def __init__(self, schedule: Schedule | None = None) -> None:
        super().__init__(schedule=schedule)
        self.triggered = True

    def _stop(self) -> bool:
        return False
    
    def best_iterate(self) -> Iterate:
        r"""Best iterate.
        
        
        Returns
        -------
        Iterate
            Result described above.
        """
        if self.solver is None or self.solver.x is None:
            raise MissingValueError(
                "NoneRule has not been attached to a solver with an iterate."
            )
        return self.solver.x, self.solver.y if hasattr(self.solver,"y") else None


class ErrorMonitor(StopRule):
    r"""Record errors to a known reference solution without stopping.
    
    Parameters
    ----------
    exact_sol: ReadOnlyVector | tuple[ReadOnlyVector | None, ReadOnlyVector]
        A reference for the solver iterates ``x``, or references ``(x, y)``
        for both the iterates and their operator values.  In the tuple form,
        the first entry may be ``None`` to monitor only ``y``.
    dist_x: Distance | None [default: None]
        Distance functions called as ``dist((current, reference))``.  If a
        required distance is omitted, the norm of ``solver.setting.h_domain``
        or ``solver.setting.h_codomain``, respectively, is used.
    dist_y: Distance | None [default: None]
        Distance functions called as ``dist((current, reference))``.  If a
        required distance is omitted, the norm of ``solver.setting.h_domain``
        or ``solver.setting.h_codomain``, respectively, is used.
    logging_level: LoggingLevel [default: 'INFO']
        Logging level.
    schedule: Schedule | None [default: None]
        Iterations at which errors are recorded. If ``None``, errors
        are recorded at every call.
    
    Notes
    -----
    This rule always returns ``False`` and therefore never stops a solver.
    Recorded values are available as ``history_dict['error_x']`` and/or
    ``history_dict['error_y']``.
    """

    def __init__(
        self,
        exact_sol: (
            ReadOnlyVector
            | tuple[ReadOnlyVector | None, ReadOnlyVector]
        ),
        dist_x: Distance | None = None,
        dist_y: Distance | None = None,
        logging_level: LoggingLevel = "INFO",
        schedule: Schedule | None = None,
    ) -> None:
        super().__init__(logging_level=logging_level, schedule=schedule)
        if isinstance(exact_sol, tuple):
            if len(exact_sol) != 2:
                raise TypeError(Errors.type_error(
                    "exact_sol must be a vector or a pair (x, y). "
                    f"Got a tuple of length {len(exact_sol)}."
                ))
            self.reference_x, self.reference_y = exact_sol
            if self.reference_y is None:
                raise TypeError(Errors.type_error(
                    "The y entry of exact_sol=(x, y) must be a vector."
                ))
        elif exact_sol is None:
            raise TypeError(Errors.type_error(
                "exact_sol must be a vector or a pair (x, y)."
            ))
        else:
            self.reference_x = exact_sol
            self.reference_y = None

        if dist_x is not None and not callable(dist_x):
            raise TypeError(Errors.type_error("dist_x must be callable."))
        if dist_y is not None and not callable(dist_y):
            raise TypeError(Errors.type_error("dist_y must be callable."))
        self._user_dist_x = dist_x
        self._user_dist_y = dist_y
        self.dist_x: Distance | None = dist_x
        self.dist_y: Distance | None = dist_y

        if self.reference_x is not None:
            self.history_dict["error_x"] = []
        if self.reference_y is not None:
            self.history_dict["error_y"] = []

    def __repr__(self) -> str:
        monitored = []
        if self.reference_x is not None:
            monitored.append("x")
        if self.reference_y is not None:
            monitored.append("y")
        return f"ErrorMonitor({', '.join(monitored)})"

    @staticmethod
    def _norm_distance(norm: Callable[[ReadOnlyVector], float]) -> Distance:
        return lambda pair: float(norm(pair[0] - pair[1])/sqrt(2))

    def _complete_init_with_solver(self, solver: Solver) -> None:
        super()._complete_init_with_solver(solver)
        setting = getattr(solver, "setting", None)
        if setting is None:
            raise RuntimeError(Errors.runtime_error(
                "ErrorMonitor requires a solver with a setting."
            ))

        if self.reference_x is not None:
            if self.reference_x not in setting.op.domain:
                raise ValueError(Errors.value_error(
                    "The x reference does not belong to the operator domain."
                ))
            self.dist_x = self._user_dist_x
            if self.dist_x is None:
                try:
                    norm_x = setting.h_domain.norm
                except (AttributeError, NotImplementedError) as error:
                    raise RuntimeError(Errors.runtime_error(
                        "No domain Hilbert norm is available; supply dist_x."
                    )) from error
                self.dist_x = self._norm_distance(norm_x)

        if self.reference_y is not None:
            if self.reference_y not in setting.op.codomain:
                raise ValueError(Errors.value_error(
                    "The y reference does not belong to the operator codomain."
                ))
            self.dist_y = self._user_dist_y
            if self.dist_y is None:
                try:
                    norm_y = setting.h_codomain.norm
                except (AttributeError, NotImplementedError) as error:
                    raise RuntimeError(Errors.runtime_error(
                        "No codomain Hilbert norm is available; supply dist_y."
                    )) from error
                self.dist_y = self._norm_distance(norm_y)

    def _stop(self) -> bool:
        if self.solver is None or self.solver.x is None:
            raise MissingValueError(
                "ErrorMonitor requires a current solver iterate."
            )
        if self.reference_y is not None and (
            not hasattr(self.solver, "y") or self.solver.y is None
        ):
            raise MissingValueError(
                "ErrorMonitor requires the current operator value to monitor y."
            )

        error_x = None
        error_y = None
        if self.reference_x is not None:
            if self.dist_x is None:
                raise RuntimeError(Errors.runtime_error(
                    "ErrorMonitor has not resolved dist_x."
                ))
            error_x = float(self.dist_x((self.solver.x, self.reference_x)))
        if self.reference_y is not None:
            if self.dist_y is None:
                raise RuntimeError(Errors.runtime_error(
                    "ErrorMonitor has not resolved dist_y."
                ))
            error_y = float(self.dist_y((self.solver.y, self.reference_y)))

        messages = []
        if error_x is not None:
            self.history_dict["error_x"].append(error_x)
            messages.append(f"error_x: {error_x:.3e}")
        if error_y is not None:
            self.history_dict["error_y"].append(error_y)
            messages.append(f"error_y: {error_y:.3e}")
        self.log_info = " | ".join(messages)
        if self.is_main_rule:
            self.log.debug(self.log_info)
        return False

class CombineRules(StopRule):
    r"""Combine several stopping rules into one that stops if one of the rules stops. (logical OR)
    
    The resulting rule triggers when any of the given rules triggers. The
    first rule that triggers is responsible for selecting the best solution.
    
    Parameters
    ----------
    rules: Sequence[StopRule]
        The rules to be combined.
    logging_level: LoggingLevel | None [default: None]
        Logging level. If ``None``, the minimum level of the component rules
        is used.
    schedule: Schedule | None [default: None]
        Optional schedule for the combined rule as a whole. Component rules
        retain and apply their own independent schedules.
    """

    def __init__(
        self,
        rules: Sequence[StopRule],
        logging_level: LoggingLevel | None = None,
        schedule: Schedule | None = None,
    ) -> None:
        if not isinstance(rules, Sequence) or any(
            not isinstance(rule, StopRule) for rule in rules
        ):
            raise TypeError(Errors.type_error(
                "Combining stopping rules requires a sequence of StopRule "
                f"instances. Got {rules} of type {type(rules)}."
            ))
        if not rules:
            raise ValueError(Errors.value_error(
                "At least one stopping rule is required."
            ))
        super().__init__(schedule=schedule)
        self.rules: list[StopRule] = []
        r"""List of :class:`StopRule` the combined rules.
        """
        self.history_dict = {}
        r"""Dictionary of the convergence histories of the rules."""
        for rule in rules:
            if type(rule) is type(self) and hasattr(rule,"solver") and rule.solver is self.solver:
                self.rules.extend(cast(CombineRules, rule).rules)
            else:
                self.rules.append(rule)
            self.history_dict.update(rule.history_dict)
            rule.is_main_rule = False
        if logging_level is None:
            self.log.setLevel(min([rule.log.level if hasattr(rule,'log') else 50 for rule in self.rules]))
        else:
            self.log.setLevel(logging_level)
        
        self.active_rule: StopRule | None = None
        r"""
        The rule that triggered the stop condition, or ``None`` if no rule has triggered yet.
        """

    def __repr__(self) -> str:
        return 'CombineRules({})'.format(self.rules)
    
    def reset(self) -> None:
        r"""Reset the state for a new run.
        """
        self.active_rule = None
        super().reset()
        for rule in self.rules:
            rule.reset()
            self.history_dict.update(rule.history_dict) 
    
    def _complete_init_with_solver(self, solver: Solver) -> None:
        self.solver = solver
        for rule in self.rules:
            rule._complete_init_with_solver(self.solver)

    def _stop(self) -> bool:
        triggered =False
        self.log_info = ''
        log_current_step = False
        for rule in self.rules:
            try:
                rule_triggered = rule.stop()
            except MissingValueError:
                if self.solver is None or self.solver.x is None:
                    raise RuntimeError(Errors.generic_message("One of the combined stopping rules needs the operator value to evaluate the stopping condition. Please provide the operator to the solver or make sure that the solver computes the operator value before calling the stopping rules."))
                op = getattr(self.solver, 'op', None)
                if op is None:
                    raise RuntimeError(Errors.generic_message("One of the combined stopping rules needs the operator value to evaluate the stopping condition. Please provide the operator to the solver or make sure that the solver computes the operator value before calling the stopping rules."))
                self.solver.y = op(self.solver.x)
                rule_triggered = rule.stop()
            log_current_step = log_current_step or rule.log_current_step
            if rule_triggered:
                self.log_info += 'Rule {} triggered.'.format(rule)
                if self.active_rule is None:
                    # The first triggering rule selects the best iterate.
                    self.active_rule = rule
                    self.active_rule.trigger()
                    self.x, self.y = self.active_rule.x, self.active_rule.y
                self.triggered = True
                triggered = True
            else:
                self.log_info = ''
        self._log_current_step = log_current_step or triggered
        log_infos_rules = ''
        for rule in self.rules:
            log_infos_rules += rule.log_info + ' | '
        log_infos_rules = log_infos_rules[:-3]
        if self.is_main_rule:
            if self._log_current_step:
                self.log.log(
                    logging.INFO if self.triggered else logging.DEBUG,
                    log_infos_rules
                    + ('\n' if self.log_info != '' else '')
                    + self.log_info,
                )
        else:
            self.log_info = '(' + log_infos_rules + (')' if self.log_info == '' else '['+self.log_info+'])')
        return triggered

class AndCombineRules(StopRule):
    r"""Combine several stopping rules into one that stops if all of the rules stop.
    
    The resulting rule triggers when all of the given rules trigger. 
    It delegates selecting the solution to the first rule.
    
    Parameters
    ----------
    rules: Sequence[StopRule]
        The rules to be combined.
    logging_level: LoggingLevel | None [default: None]
        Logging level. If ``None``, the minimum level of the component rules
        is used.
    schedule: Schedule | None [default: None]
        Optional schedule for the combined rule as a whole. Component rules
        retain and apply their own independent schedules.
    """

    def __init__(
        self,
        rules: Sequence[StopRule],
        logging_level: LoggingLevel | None = None,
        schedule: Schedule | None = None,
    ) -> None:
        if not isinstance(rules, Sequence) or any(
            not isinstance(rule, StopRule) for rule in rules
        ):
            raise TypeError(Errors.type_error(
                "Combining stopping rules requires a sequence of StopRule "
                f"instances. Got {rules} of type {type(rules)}."
            ))
        if not rules:
            raise ValueError(Errors.value_error(
                "At least one stopping rule is required."
            ))
        super().__init__(schedule=schedule)
        self.rules: list[StopRule] = []
        r"""List of :class:`StopRule` the combined rules.
        """
        self.history_dict = {}
        r"""Dictionary of the convergence histories of the rules."""
        for rule in rules:
            if type(rule) is type(self) and hasattr(rule,"solver") and rule.solver is self.solver:
                self.rules.extend(cast(AndCombineRules, rule).rules)
            else:
                self.rules.append(rule)
            self.history_dict.update(rule.history_dict)
            rule.is_main_rule = False
        if logging_level is None:         
            self.log.setLevel(min([rule.log.level if hasattr(rule,'log') else 50 for rule in self.rules]))
        else:
            self.log.setLevel(logging_level)
        self.active_rule: StopRule | None = None

    def __repr__(self) -> str:
        return 'AndCombineRules({})'.format(self.rules)
    
    def reset(self) -> None:
        r"""Reset the state for a new run.
        """
        self.active_rule = None
        super().reset()
        for rule in self.rules:
            rule.reset()
            self.history_dict.update(rule.history_dict) 
    
    def _complete_init_with_solver(self, solver: Solver) -> None:
        self.solver = solver
        for rule in self.rules:
            rule._complete_init_with_solver(self.solver)

    def _stop(self) -> bool:
        triggered =True
        self.log_info = ''
        log_current_step = False
        for rule in self.rules:
            try:
                rule_triggered = rule.stop()
            except MissingValueError:
                if self.solver is None or self.solver.x is None:
                    raise RuntimeError(Errors.generic_message("One of the combined stopping rules needs the operator value to evaluate the stopping condition. Please provide the operator to the solver or make sure that the solver computes the operator value before calling the stopping rules."))
                op = getattr(self.solver, 'op', None)
                if op is None:
                    raise RuntimeError(Errors.generic_message("One of the combined stopping rules needs the operator value to evaluate the stopping condition. Please provide the operator to the solver or make sure that the solver computes the operator value before calling the stopping rules."))
                self.solver.y = op(self.solver.x)
                rule_triggered = rule.stop()
            log_current_step = log_current_step or rule.log_current_step
            if not rule_triggered:
                triggered = False
            else:
                self.log_info = ''
        if triggered:
            self.log_info += 'All rules triggered.'
            self.triggered = True
            self.rules[0].trigger() # first rule decides best iterate
            self.x, self.y = self.rules[0].x, self.rules[0].y
        self._log_current_step = log_current_step or triggered
        log_infos_rules = ''
        for rule in self.rules:
            log_infos_rules += rule.log_info + ' & '
        log_infos_rules = log_infos_rules[:-3]
        if self.is_main_rule:
            if self._log_current_step:
                self.log.log(
                    logging.INFO if self.triggered else logging.DEBUG,
                    log_infos_rules
                    + ('\n' if self.log_info != '' else '')
                    + self.log_info,
                )
        else:
            self.log_info = '(' + log_infos_rules + (')' if self.log_info == '' else '['+self.log_info+'])')
        return triggered
    

class CountIterations(StopRule):
    r"""Stopping rule based on number of iterations.
    
    Each call to :meth:`stop` increments the iteration count by one.
    
    Parameters
    ----------
    max_iterations: int
        The number of iterations after which to stop.
    while_type: bool [default: True]
        Indicates whether the rule is used in a ``while`` loop or an
        ``until`` loop.
    timing: bool [default: False]
        If ``True``, the CPU time elapsed since the start of the iteration is
        saved in :attr:`history_dict`.
    logging_level: LoggingLevel [default: 'INFO']
        Logging level.
    schedule: Schedule | None [default: None]
        Iterations at which the limit is checked and timing information is
        stored. The iteration counter is updated on every call. If the limit
        is reached at an omitted step, the rule stops at the next scheduled
        step. If ``None``, the limit is checked on every call.
    eval_count: bool [default: False]
        If True, enable counting only on solver.op and reset its counters
        when attaching the rule, then store
        cumulative snapshots in eval_counts, a dictionary of integer lists
        with the six Operator keys and 'total'. eval_count_steps records the
        corresponding solver steps, including step zero in while-style runs.
        Snapshots are recorded even on steps omitted by schedule. Constructor
        work is included when Setting.run prepares the rule before solver
        construction. Direct solver.run calls start measurements at attachment
        unless prepare_run was called beforehand. Counting remains enabled after
        the run for inspection; use solver.op.disable_eval_count() to remove
        instrumentation. False leaves existing operator counting unchanged.
    max_op_evals: int | None [default: None]
        Stop when the weighted operator evaluation total reaches this cap,
        or when max_iterations is reached. None disables this additional cap.
        Supplying a cap automatically enables evaluation counting. Uses
        Operator.total_eval_count(), including its doubled weights for
        adjoint_eval and adjoint_deriv. Checked when this rule is evaluated;
        constructor work or one iteration can overshoot the cap. A schedule
        can delay checking until the next scheduled application.
    """

    # The evaluation cap is checked between iterations, not inside operator
    # calls; a single iteration or constructor may exceed the requested cap.
    def __init__(
        self,
        max_iterations: int,
        while_type: bool = True,
        timing: bool = False,
        logging_level: LoggingLevel = "INFO",
        schedule: Schedule | None = None,
        eval_count: bool = False,
        max_op_evals: int | None = None,
    ) -> None:
        if not isinstance(eval_count, bool):
            raise TypeError("eval_count must be a bool.")
        if max_op_evals is not None:
            if isinstance(max_op_evals, bool) or not isinstance(max_op_evals, Integral):
                raise TypeError("max_op_evals must be a nonnegative integer or None.")
            if max_op_evals < 0:
                raise ValueError("max_op_evals must be nonnegative.")
        self.max_op_evals = None if max_op_evals is None else int(max_op_evals)
        self._prepared_operator = None
        self.eval_count = eval_count or max_op_evals is not None
        self.eval_counts: dict[str, list[int]] = {}
        self.eval_count_steps: list[int] = []
        if not isinstance(max_iterations,Integral):
            raise TypeError(Errors.type_error("The maximal iteration in the CountIterations should be an integer!",max_iterations))
        if max_iterations<0:
            raise ValueError(Errors.value_error("The maximal iteration in CountIteration needs to be at least zero (for no iteration)!",max_iterations))
        super().__init__(
            logging_level=logging_level,
            schedule=schedule,
        )
        self.max_iterations: int = max_iterations
        self.iteration: int = 0
        self.while_type: bool = while_type
        self.timing: bool = timing
        self.start_time: float | None = None
        if timing: 
            self.history_dict['time'] = []
            self.start_time = time.process_time()


    def __repr__(self) -> str:
        cap = '' if self.max_op_evals is None else f', max_op_evals={self.max_op_evals}'
        return f'CountIterations(max_iterations={self.max_iterations}{cap})' 
    
    def reset(self) -> None:
        r"""Reset the state for a new run.
        """
        super().reset()
        self._prepared_operator = None
        self.iteration = 0
        self.eval_counts = {}
        self.eval_count_steps = []
        if self.timing:
            self.start_time = time.process_time()

    def prepare_run(self, operator: Operator) -> None:
        r"""Start measurements before constructing a solver (used by Setting).
        
        Attachment to the same operator preserves construction-time work.
        Direct solver.run calls instead start measurements at attachment.
        
        Parameters
        ----------
        operator: Operator
            Operator to inspect or evaluate.
        """
        if self.timing:
            self.start_time = time.process_time()
        if self.eval_count:
            operator.enable_eval_count(reset=True)
        self._prepared_operator = operator

    def _complete_init_with_solver(self, solver: Solver) -> None:
        super()._complete_init_with_solver(solver)
        operator = getattr(solver, 'op', None)
        if self._prepared_operator is None or self._prepared_operator is not operator:
            if self.timing:
                self.start_time = time.process_time()
            if self.eval_count:
                operator.enable_eval_count(reset=True)
        self._prepared_operator = None
        if self.eval_count:
            self.eval_counts = {key: [] for key in operator.eval_counts}
            self.eval_counts['total'] = []
            self.eval_count_steps = []

    def _record_eval_counts(self) -> None:
        if self.eval_count:
            if self.solver is None:
                raise RuntimeError("Evaluation counting requires an attached solver.")
            snapshot = self.solver.op.eval_counts
            snapshot['total'] = self.solver.op.total_eval_count()
            for key, value in snapshot.items():
                self.eval_counts[key].append(value)
            self.eval_count_steps.append(self._current_step)

    def _stop(self) -> bool:
        self._record_eval_counts()
        if self.timing:
            if self.start_time is None:
                self.start_time = time.process_time()
            self.history_dict['time'].append(
                time.process_time() - self.start_time
            )
        if self.while_type:
            triggered = self.iteration >= self.max_iterations
            self.log_info = 'it. {}>={}'.format(self.iteration, self.max_iterations)
            self.iteration += 1
        else:
            self.iteration += 1
            triggered = self.iteration >= self.max_iterations
            self.log_info = 'it. {}>={}'.format(self.iteration, self.max_iterations)
        if self.max_op_evals is not None:
            total = self.eval_counts['total'][-1]
            triggered = triggered or total >= self.max_op_evals
            self.log_info += f' | op_evals {total}>={self.max_op_evals}'
        # In a combination, an ordinary iteration-count message should not
        # cause logging between scheduled evaluations of an informative rule.
        # The actual stopping event must always remain visible.
        self._log_current_step = triggered
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)
        return triggered

    def _skip(self) -> None:
        r"""Count an iteration without checking the limit or storing timing.
        """
        self._record_eval_counts()
        self.iteration += 1
        self._log_current_step = False
        self.log_info = (
            f'it. {self.iteration}, iteration limit checked only at selected '
            'steps'
        )



def CountIterations_factory(setting: Setting, params: dict) -> StopRule:
    r"""Create a timed cap on iterations and weighted operator evaluations.
    
    max_it defaults to 1000; max_op_evals defaults to 2000. Explicitly set
    max_op_evals=None to disable the evaluation cap. setting is accepted for
    the uniform factory interface. All default factories use this cap.
    
    Parameters
    ----------
    setting: Setting
        Problem setting containing the operator, penalty, and data fidelity.
    params: dict
        Options for params.
    
    
    Returns
    -------
    StopRule
        Result described above.
    """
    from numbers import Integral
    max_it = params.get('max_it', 1000)
    if isinstance(max_it, bool) or not isinstance(max_it, Integral):
        raise TypeError("max_it must be a nonnegative integer.")
    if max_it < 0:
        raise ValueError("max_it must be nonnegative.")
    return CountIterations(int(max_it), timing=True, eval_count=True,
                           max_op_evals=params.get('max_op_evals', 2000),
                           logging_level=params.get('logging_level', 'INFO'))


def _capped_factory(rule_class: type[StopRule]) -> Callable[[Setting, dict], StopRule]:
    r"""Build a factory for a concrete rule, preserving its constructor defaults.
    
    Factories accept (setting, params). Recognized constructor arguments can
    be supplied directly in params; rule_params supplies explicit arguments
    and takes precedence. max_it controls the added cap. Parameters belonging
    to the default-selector factory are not passed to the rule constructor.
    Rules requiring a solver can receive it through params['solver']; otherwise
    normal attachment supplies it after construction.
    
    Parameters
    ----------
    rule_class: type[StopRule]
        Rule class.
    
    
    Returns
    -------
    Callable[[Setting, dict], StopRule]
        Factory combining the requested stopping rule with an iteration cap.
    """
    from inspect import signature
    names = set(signature(rule_class).parameters)

    def factory(setting: Setting, params: dict) -> StopRule:
        r"""Create a StopRule instance from a parameter dictionary and potentially further 
        information contained in the setting. 
        
        Parameters
        ----------
        setting: Setting
            Problem setting containing the operator, penalty, and data fidelity.
        params: dict
            Options for params.
        
        
        Returns
        -------
        StopRule
            A CombinedSetting consisting of the actualy StopRule and a CountIterations instance. 
        """
        options = {key: value for key, value in params.items() if key in names}
        if 'setting' in names:
            options['setting'] = setting
        if 'solver' in names:
            options.setdefault('solver', None)
        if 'max_iter' in names:
            options.setdefault('max_iter', params.get('max_it', 1000))
        if rule_class.__name__ in ('RelativeChangeSol', 'RelativeChangeData'):
            space = setting.h_domain if rule_class.__name__ == 'RelativeChangeSol' else setting.h_codomain
            options.setdefault('norm', space.norm)
            options.setdefault('tol', 1e-3)
        options.update(params.get('rule_params', {}))
        rule = rule_class(**options)
        rule.log.setLevel(params.get('logging_level', 'INFO'))
        return rule + CountIterations_factory(setting, params)

    factory.__name__ = rule_class.__name__ + '_factory'
    factory.__qualname__ = factory.__name__
    factory.__module__ = rule_class.__module__
    factory.__doc__ = (
        'Create ' + rule_class.__name__ + ' plus a timed, evaluation-counting '
        'iteration cap. Call as factory(setting, params). Constructor options '
        'are read from params; rule_params overrides them. max_it defaults to '
        '1000. Required rule-specific arguments must be supplied by the caller.'
    )
    return factory


NoneRule_factory = _capped_factory(NoneRule)
ErrorMonitor_factory = _capped_factory(ErrorMonitor)
CombineRules_factory = _capped_factory(CombineRules)
AndCombineRules_factory = _capped_factory(AndCombineRules)
__all__ += ['CountIterations_factory', 'NoneRule_factory', 'ErrorMonitor_factory',
            'CombineRules_factory', 'AndCombineRules_factory']
