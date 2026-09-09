"""General stopping-rule infrastructure.

This module contains the base class, rule combinators, and the iteration-count rule.
"""

from __future__ import annotations

from copy import deepcopy
import logging
import time
from collections.abc import Sequence
from types import NotImplementedType
from typing import TYPE_CHECKING, TypeAlias, cast

from typing_extensions import Self

from regpy.util import ClassLogger, Errors
from regpy.vecsps import ReadOnlyVector, Vector, VectorSpaceBase

if TYPE_CHECKING:
    from regpy.solvers import Solver

LoggingLevel: TypeAlias = int | str
Iterate: TypeAlias = tuple[ReadOnlyVector, ReadOnlyVector | None]
BestIterate: TypeAlias = Iterate | Vector

__all__: list[str] = [
    "MissingValueError", "StopRule", "NoneRule", "CombineRules",
    "AndCombineRules", "CountIterations",
]

class MissingValueError(Exception):
    """Indicates that a stopping rule needs an unavailable solver value."""

    pass


def _copy_vector(
    space: VectorSpaceBase, vector: ReadOnlyVector
) -> Vector:
    """Return a writable copy of ``vector`` allocated in ``space``."""
    result = space.empty()
    space.copyto(result, vector)
    return result

class StopRule:
    r"""Abstract base class for stopping rules.

    A solver attaches itself with :meth:`_complete_init_with_solver` before the
    iteration starts. :meth:`stop` then delegates the decision to
    :meth:`_stop`. If this is the main rule, a positive decision stores the
    current iterate through :meth:`trigger`.

    Parameters
    ----------
    logging_level : int | str, optional
        Logging level. Defaults to ``"INFO"``.
    """

    log = ClassLogger()

    solver: Solver | None
    x: ReadOnlyVector
    y: ReadOnlyVector | None

    def __init__(self, logging_level: LoggingLevel = "INFO") -> None:
        self.solver: Solver | None = None

        self.triggered: bool = False
        """Whether the stoprule decided to stop."""
        self.history_dict: dict[str, list[float]] = {}
        """A place to save scalars for later use/analysis. An entry of the form {"parameter_name":[]} needs to be added in the implementation of the stoprule."""
        self.is_main_rule: bool = True
        r"""Whether this is the main stoprule of the solver or a sub-rule used for example in a combined rule."""
        self.log_info: str = ""
        self.log.setLevel(logging_level)

    def _complete_init_with_solver(self, solver: Solver) -> None:
        """Attach a solver and complete rule-specific initialization.

        Subclasses may override this hook when their criterion needs
        information from the concrete solver.

        Parameters
        ----------
        solver : regpy.solvers.Solver
            Solver to which this rule is attached.
        """
        if self.solver is not None:
            self.log.warning("the solver was already set and is now overwritten")
        self.solver = solver

    def copy_and_reset(self) -> Self:
        """Return an independent copy reset to its initial state."""
        rule = self.copy()
        rule.reset()
        return rule
    
    def reset(self) -> None:
        """Reset the rule to its initial state.

        Subclasses that maintain additional state must override this method and
        reset that state as well.
        """
        self.solver = None
        self.triggered = False
        if hasattr(self,"x"):
            del self.x
        if hasattr(self,"y"):
            del self.y

        for key in self.history_dict.keys():
            self.history_dict[key] = []

    def copy(self) -> Self:
        return cast(Self, deepcopy(self))

    def stop(self) -> bool:
        """Check whether to stop iterations.

        Returns
        -------
        bool
            ``True`` if iterations should be stopped.
        """
        should_stop = self._stop()
        if should_stop and self.is_main_rule:
            self.trigger()
        return should_stop

    def trigger(self) -> None:
        """Force the stoprule to trigger at the current iterate.

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
        """Return the best iterate selected by this stopping rule.

        The default implementation returns the last iterate available when
        the rule triggered or the solver converged. Parameter-choice rules may
        override this method and return only the selected solution vector.

        Returns
        -------
        tuple[ReadOnlyVector, ReadOnlyVector | None], Vector, or None
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
        """Check whether to stop iterations.

        This is an abstract method. Child classes should override it.

        Parameters and return values are the same as for the public interface
        method :meth:`stop`.

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
    """

    def __init__(self) -> None:
        super().__init__()
        self.triggered = True

    def _stop(self) -> bool:
        return False
    
    def best_iterate(self) -> Iterate:
        if self.solver is None or self.solver.x is None:
            raise MissingValueError(
                "NoneRule has not been attached to a solver with an iterate."
            )
        return self.solver.x, self.solver.y if hasattr(self.solver,"y") else None

class CombineRules(StopRule):
    r"""Combine several stopping rules into one that stops if one of the rules stops. (logical OR)

    The resulting rule triggers when any of the given rules triggers. The
    first rule that triggers is responsible for selecting the best solution.

    Parameters
    ----------
    rules : sequence of StopRule
        The rules to be combined.
    logging_level : int | str | None, optional
        Logging level. If ``None``, the minimum level of the component rules
        is used.
    """

    def __init__(
        self,
        rules: Sequence[StopRule],
        logging_level: LoggingLevel | None = None,
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
        super().__init__()
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
        log_infos_rules = ''
        for rule in self.rules:
            log_infos_rules += rule.log_info + ' | '
        log_infos_rules = log_infos_rules[:-3]
        if self.is_main_rule:
            self.log.log(logging.INFO if self.triggered else logging.DEBUG,
                         log_infos_rules+('\n' if self.log_info != '' else '')+self.log_info
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
    rules : sequence of StopRule
        The rules to be combined.
    logging_level : int | str | None, optional
        Logging level. If ``None``, the minimum level of the component rules
        is used.
    """

    def __init__(
        self,
        rules: Sequence[StopRule],
        logging_level: LoggingLevel | None = None,
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
        super().__init__()
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
            if not rule_triggered:
                triggered = False
            else:
                self.log_info = ''
        if triggered:
            self.log_info += 'All rules triggered.'
            self.triggered = True
            self.rules[0].trigger() # first rule decides best iterate
            self.x, self.y = self.rules[0].x, self.rules[0].y
        log_infos_rules = ''
        for rule in self.rules:
            log_infos_rules += rule.log_info + ' & '
        log_infos_rules = log_infos_rules[:-3]
        if self.is_main_rule:
            self.log.log(logging.INFO if self.triggered else logging.DEBUG,
                        log_infos_rules+('\n' if self.log_info != '' else '')+self.log_info
                        )
        else:
            self.log_info = '(' + log_infos_rules + (')' if self.log_info == '' else '['+self.log_info+'])')
        return triggered
    

class CountIterations(StopRule):
    r"""Stopping rule based on number of iterations.

    Each call to :meth:`stop` increments the iteration count by one.

    Parameters
    ----------
    max_iterations : int
        The number of iterations after which to stop.
    while_type : bool, optional
        Indicates whether the rule is used in a ``while`` loop (default) or an
        ``until`` loop.
    timing : bool, optional
        If ``True``, the CPU time elapsed since the start of the iteration is
        saved in :attr:`history_dict` on each call. Defaults to ``False``.
    logging_level : int | str, optional
        Logging level. Defaults to ``"INFO"``.

    """

    def __init__(
        self,
        max_iterations: int,
        while_type: bool = True,
        timing: bool = False,
        logging_level: LoggingLevel = "INFO",
    ) -> None:
        if not isinstance(max_iterations,int):
            raise TypeError(Errors.type_error("The maximal iteration in the CountIterations should be an integer!"))
        if max_iterations<0:
            raise ValueError(Errors.value_error("The maximal iteration in CountIteration needs to be at least zero (for no iteration)!"))
        super().__init__()
        self.max_iterations: int = max_iterations
        self.iteration: int = 0
        self.while_type: bool = while_type
        self.timing: bool = timing
        self.start_time: float | None = None
        if timing: 
            self.history_dict['time'] = []
            self.start_time = time.process_time()
        self.log.setLevel(logging_level)


    def __repr__(self) -> str:
        return 'CountIterations(max_iterations={})'.format(self.max_iterations)
    
    def reset(self) -> None:
        super().reset()
        self.iteration = 0
        if self.timing:
            self.start_time = time.process_time()

    def _stop(self) -> bool:
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
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)
        return triggered

