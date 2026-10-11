"""Solver lifecycle, generator-based iteration, and stopping-rule drivers."""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from regpy.solvers.setting import Setting

from collections.abc import Callable, Generator

from numbers import Integral
import sys

import numpy as np

from regpy.vecsps import Vector, ReadOnlyVector
from regpy.stoprules import StopRule, NoneRule
from regpy.util import ClassLogger, Errors
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from regpy.operators import Operator
    from regpy.functionals import Functional
    from regpy.hilbert import HilbertSpace

__all__ = ["Solver", "RegSolver", "CombinedSolver"]

class Solver:
    r"""Abstract base class for iterative solvers. 
    Solvers typically implement iterative methods for solving operator equations or optimization problems involving an operator. However, the operator is not part of the base class interface, allowing solvers to be used in a variety of contexts. 

    Solvers do not implement loops themselves, but are driven by
    repeatedly calling the `next` method. They expose the current iterate and its value under the operator as attributes
    `x` and `y`, and can be iterated over, yielding the `(x, y)` tuple on every iteration (which
    may or may not be the same arrays as before, modified in-place).
    
    The convenience methods ``run``, ``while_`` and ``until`` run the solver with a `regpy.stoprules.StopRule`. The latter two allow for monitoring, plotting, etc, of the iterates by the caller. 
    
    Each call to ``iter(solver)`` creates a generator driving the same mutable
    solver state; it does not restart the solver. The solver is iterable but
    does not itself implement ``__next__``. Use ``solver.next()`` for a single
    step or ``next(iter(solver))`` to obtain a pair after a step.
    
    Yielded vectors are read-only views, not copies. Solver updates may
    still change their contents. Copy them when retaining snapshots of the
    iteration history. Either of the components may be ``None``.
    

    Subclasses should override the method `_next(self)` to perform a single iteration where the values of 
    the attributes `x` and `y` are updated. The main difference to `next` is that `_next` does not have a
    return value. If the solver converged, `converge` should be called, afterwards `_next` will never be
    called again. Most solvers will probably never converge on their own, but rely on the caller or a
    `regpy.stoprules.StopRule` for termination.
    
    Parameters
    ----------
    x: Vector | None [default: None]
        Initial argument for iteration.
    y: Vector | None [default: None]
        Initial value of the operator at the current iterate.
    """

    log = ClassLogger()

    def __init__(self, 
                 x:Vector | None = None,
                 y:Vector | None = None) -> None:
        self.x: Vector | None = x
        """The current iterate."""
        self.y: Vector | None = y
        """The value of the operator at the current iterate. May be needed by stopping rules, but callers should
        handle the case when it is not available."""
        self.__converged: bool = False
        self.iteration_step_nr: int = 0
        """Current number of iterations performed."""

    def converge(self) -> None:
        r"""Mark the solver as converged. This is intended to be used by child classes
        implementing the `_next` method.
        """
        self.__converged = True

    def is_converged(self) -> bool:
        r"""Return whether :meth:`converge` has been called.
        
        An external stopping rule does not by itself set this flag.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.__converged

    def next(self) -> bool:
        r"""Perform a single iteration.
        
        Returns
        -------
        bool
            False if the solver already converged and no step was performed.
            True otherwise.
        """
        if self.__converged:
            return False
        self.iteration_step_nr += 1    
        self._next()
        return True

    def _next(self) -> None:
        r"""Perform a single iteration. This is an abstract method called from the public method
        `next`. Child classes should override it.
        
        The main difference to `next` is that `_next` does not have a return value. If the solver
        converged, `converge` should be called.
        """
        raise NotImplementedError

    def _read_only_iterate(
        self,
    ) -> tuple[ReadOnlyVector | None, ReadOnlyVector | None]:
        r"""Expose the current pair without making solver buffers read-only.
        
        Operator-based solvers use their vector spaces' ``read_only`` API.
        Plain solvers support NumPy arrays directly. 
        For solver without an operator using non-NumPy vectors the iterates 
        are returned as writable vectors (at the risk that they are accidentally modified).        
        
        
        Returns
        -------
        tuple[ReadOnlyVector | None, ReadOnlyVector | None]
            Result described above.
        """
        op = getattr(self, "op", None)

        def view(value: ReadOnlyVector | None, component: str) -> ReadOnlyVector | None:
            r"""View.
            
            Parameters
            ----------
            value: ReadOnlyVector | None
                Input vector; its coefficients are only read.
            component: str
                Component.
            
            
            Returns
            -------
            ReadOnlyVector | None
                Result described above.
            """
            if value is None:
                return None
            if op is not None:
                return getattr(op, component).read_only(value)
            if isinstance(value, np.ndarray):
                result = value.view()
                result.flags.writeable = False
                return result
            return result
            #raise TypeError(
            #    "A solver without an operator using non-NumPy vectors must "
            #    "implement _read_only_iterate()."
            #)

        return view(self.x, "domain"), view(self.y, "codomain")

    def __iter__(self) -> Generator[tuple[ReadOnlyVector | None, ReadOnlyVector | None], None, None]:
        r"""Generate pairs after successive solver steps, excluding the initial pair.
        
        Execution starts when the generator is advanced. It ends when
        :meth:`next` returns ``False``. A step that marks convergence still
        yields its resulting pair as read-only views. No vector copies are made.
        
        Yields
        ------
        tuple[ReadOnlyVector | None, ReadOnlyVector | None]
            The current pair, with either component possibly unavailable.
        """
        while self.next():
            yield self._read_only_iterate()

    @staticmethod
    def _add_max_iterations(
        stoprule: StopRule | None,
        max_it: int | None,
        *,
        while_type: bool,
        max_op_evals: int | None = None,
    ) -> StopRule:
        r"""Return ``stoprule`` augmented by iteration and operator-evaluation limits.
        
        Parameters
        ----------
        stoprule: StopRule | None
            Stoprule.
        max_it: int | None
            Maximum number of iterations.
        max_op_evals: int | None [default: None]
            Optional weighted operator-evaluation limit.
        while_type: bool
            Whether to check before the first iteration.
        
        
        Returns
        -------
        StopRule
            Result described above.
        """
        if stoprule is None:
            stoprule = NoneRule()
        elif not isinstance(stoprule, StopRule):
            raise TypeError(Errors.type_error(
                f'stoprule must be a StopRule or None. Got {stoprule!r}.'
            ))

        for name, value in (("max_it", max_it), ("max_op_evals", max_op_evals)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer or None. Got {value!r}.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative. Got {value}.")
        if max_it is None and max_op_evals is None:
            return stoprule

        from regpy.stoprules import CountIterations

        count_rule = CountIterations(
            max_iterations=sys.maxsize if max_it is None else int(max_it),
            while_type=while_type,
            max_op_evals=max_op_evals,
        )
        if isinstance(stoprule, NoneRule):
            return count_rule
        return stoprule + count_rule

    def while_(
        self,
        stoprule: StopRule | None = None,
        max_it: int | None = None,
        max_op_evals: int | None = None,
    ) -> Generator[tuple[ReadOnlyVector | None, ReadOnlyVector | None], None, None]:
        r"""Generate current iterates under control of a stopping rule.
        
        A typical use is:
        
        for x,y in mysolver.while_(mystoprule,max_it=100):
            plot x,y
            save x,y
            etc.
        
        The rule is checked before each attempted step, including at the
        initial state. Every completed step is yielded before the next rule
        check. An initially satisfied rule yields nothing. Calling this method
        creates a generator: initialization and argument validation occur only
        when it is advanced. 
        Yielded vectors are read-only views, not snapshots to prevent accidental 
        modifications by the user.
        
        Parameters
        ----------
        stoprule: StopRule | None [default: None]
            The stoprule to be used. If omitted, stopping will only be
            based on the return value of `next` and on ``max_it`` or ``max_op_evals``.
        max_it: int | None [default: None]
            Maximum number of iterations. If provided, a
            :class:`regpy.stoprules.CountIterations` rule is combined with
            ``stoprule``. Must be non-negative.
        max_op_evals: int | None [default: None]
            Maximum weighted operator evaluations since attaching the cap to
            this solver. Enables counting automatically and requires a solver
            with an operator. Stops when either cap is reached. Checks occur
            between iterations, so one step may exceed the evaluation cap.
            Constructor work is excluded. Must be non-negative; ``until``
            requires a positive cap because it attempts a first step before
            checking. If only this cap is supplied, the iteration cap is
            ``sys.maxsize`` (effectively unlimited).
        
        Yields
        ------
        tuple[ReadOnlyVector | None, ReadOnlyVector | None]
            The current ``(x, y)`` pair after a step. The stopping rule's best
            iterate is not substituted here; use :meth:`run` to select it.
        
        Raises
        ------
        TypeError
            If ``stoprule`` is not a stopping rule or either cap is not an
            integer or ``None``.
        ValueError
            If either cap is negative.
        """
        stoprule = self._add_max_iterations(
            stoprule, max_it, while_type=True, max_op_evals=max_op_evals
        )
        stoprule._complete_init_with_solver(self)
        while not stoprule.stop() and self.next(): 
            yield self._read_only_iterate()
        self.log.info('Solver converged after {} iteration.'.format(self.iteration_step_nr))
 


    def until(
        self,
        stoprule: StopRule | None = None,
        max_it: int | None = None,
        max_op_evals: int | None = None,
    ) -> Generator[tuple[ReadOnlyVector | None, ReadOnlyVector | None], None, None]:
        r"""Generate current iterates under control of a stopping rule.
        
        One step is attempted and the current pair is yielded before the
        first stopping-rule check. If already converged, the solver performs
        no step but still yields its current pair once. Subsequent steps are
        checked and yielded as in :meth:`while_`. Calling this method creates
        a generator; validation is deferred until it is advanced. It accepts
        no meaningful sent values and returns no final value. Vectors are
        yielded as read-only views, not snapshots if possible to prevent accidental 
        modifications by the user.
        
        Parameters
        ----------
        stoprule: StopRule | None [default: None]
            The stoprule to be used. If omitted, stopping will only be
            based on the return value of `next` and on ``max_it`` or ``max_op_evals``.
        max_it: int | None [default: None]
            Maximum number of iterations. If provided, a
            :class:`regpy.stoprules.CountIterations` rule is combined with
            ``stoprule``. Must be non-negative.
        max_op_evals: int | None [default: None]
            Maximum weighted operator evaluations since attaching the cap to
            this solver. Enables counting automatically and requires a solver
            with an operator. Stops when either cap is reached. Checks occur
            between iterations, so one step may exceed the evaluation cap.
            Constructor work is excluded. Must be non-negative; ``until``
            requires a positive cap because it attempts a first step before
            checking. If only this cap is supplied, the iteration cap is
            ``sys.maxsize`` (effectively unlimited).
        
        Yields
        ------
        tuple[ReadOnlyVector | None, ReadOnlyVector | None]
            The current ``(x, y)`` pair after a step. The stopping rule's best
            iterate is not substituted here; use :meth:`run` to select it.
        
        Raises
        ------
        TypeError
            If ``stoprule`` is not a stopping rule or either cap is not an
            integer or ``None``.
        ValueError
            If either cap is negative or zero. Since :meth:`until` always
            attempts an initial iteration, use :meth:`while_` or :meth:`run`
            when zero iterations should be allowed.
        """
        stoprule = self._add_max_iterations(
            stoprule, max_it, while_type=False, max_op_evals=max_op_evals
        )
        if max_it == 0 or max_op_evals == 0:
            raise ValueError(Errors.value_error(
                'until always performs at least one iteration and therefore '
                'does not accept max_it=0 or max_op_evals=0. Use while_ or run instead.'
            ))
        stoprule._complete_init_with_solver(self)
        self.next()
        yield self._read_only_iterate()
        while not stoprule.stop() and self.next(): 
            yield self._read_only_iterate()

        self.log.info('Solver converged after {} iteration.'.format(self.iteration_step_nr))

    def run(
        self,
        stoprule: StopRule | None = None,
        max_it: int | None = None,
        max_op_evals: int | None = None,
    ) -> tuple[ReadOnlyVector | None, ReadOnlyVector | None] | ReadOnlyVector:
        r"""Run the solver with the given stoprule.
        
        Parameters
        ----------
        stoprule: StopRule | None [default: None]
            The stoprule to be used. If omitted, stopping will only be based
            on convergence of the solver or either supplied cap.
        max_it: int | None [default: None]
            Maximum number of iterations. If provided, a
            :class:`regpy.stoprules.CountIterations` rule is combined with
            ``stoprule``. Must be non-negative.
        max_op_evals: int | None [default: None]
            Maximum weighted operator evaluations since attaching the cap to
            this solver. Enables counting automatically and requires a solver
            with an operator. Stops when either cap is reached. Checks occur
            between iterations, so one step may exceed the evaluation cap.
            Constructor work is excluded. Must be non-negative; ``until``
            requires a positive cap because it attempts a first step before
            checking. If only this cap is supplied, the iteration cap is
            ``sys.maxsize`` (effectively unlimited).
        
        Returns
        -------
        tuple[ReadOnlyVector | None, ReadOnlyVector | None] | ReadOnlyVector
            The result selected by ``stoprule.best_iterate()``, or the current
            pair if no best iterate is available. Parameter-choice rules may
            return a solution vector alone. Treat returned vectors as read-only. No copies are made. With
            zero permitted steps the initial state can be returned.
        
        Raises
        ------
        TypeError
            If ``stoprule`` is not a stopping rule or either cap is not an
            integer or ``None``.
        ValueError
            If either cap is negative.
        """
        stoprule = self._add_max_iterations(
            stoprule, max_it, while_type=True, max_op_evals=max_op_evals
        )
        for _ in self.while_(stoprule):
            pass
        best = stoprule.best_iterate()
        if best is None:
            self.log.info(f"Could not find a best iterate with stop rule {stoprule}!")
            return self.x, self.y
        return best


class RegSolver(Solver):
    r"""Base class for solvers operating on a regularization setting.
    
    The import of :class:`regpy.solvers.setting.Setting` is intentionally
    deferred until initialization. This keeps ``solver`` independent while
    allowing ``setting`` and concrete solver modules to import ``RegSolver``
    without creating a circular module import.
    
    Parameters
    ----------
    setting: Setting
        Setting used by the solver.
    x: Vector | None [default: None]
        Initial domain iterate.
    y: Vector | None [default: None]
        Value at the initial iterate, if available.
    is_primal: bool [default: True]
        Label current coordinates as primal (True, default) or dual (False).
    """

    def __init__(
        self,
        setting: "Setting",
        x: Vector | None = None,
        y: Vector | None = None,
        is_primal: bool = True,
    ) -> None:
        from .setting import Setting

        if not isinstance(setting, Setting):
            raise TypeError(Errors.not_instance(setting, Setting))
        self.is_primal = is_primal
        self.op: Operator = setting.op
        self.penalty: Functional = setting.penalty
        self.h_domain: HilbertSpace = setting.h_domain
        if not setting.codomain_free:
            self.data_fid: Functional = setting.data_fid
            self.h_codomain: HilbertSpace = setting.h_codomain
        self.setting: Setting = setting
        if setting.is_tikhonov:
            self.regpar: float | None = setting.regpar
        super().__init__(x, y)

    @property
    def is_primal(self) -> bool:
        r"""Whether the current iterate uses original primal coordinates.
        
        False denotes dual coordinates. Hybrid solvers may change this flag
        at a stage transition; setting it does not convert any vectors.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self._is_primal

    @is_primal.setter
    def is_primal(self, value: bool) -> None:
        r"""Is primal.
        
        Parameters
        ----------
        value: bool
            Value.
        """
        if not isinstance(value, bool):
            raise TypeError("is_primal must be a bool.")
        self._is_primal = value

    def runWithDP(
        self,
        data: ReadOnlyVector,
        delta: float = 0,
        tau: float = 2.1,
        max_its: int = 1000,
    ) -> tuple[Vector | None, Vector | None]:
        r"""Run the solver with Morozov's discrepancy principle.
        
        Requires an explicit codomain Hilbert space. The discrepancy rule is
        combined with an iteration cap; :meth:`run` returns the selected pair.
        A warning is logged if the active rule is not the discrepancy rule.
        
        Parameters
        ----------
        data: ReadOnlyVector
            Observed data used to measure the residual in ``h_codomain``.
        delta: float [default: 0]
            Noise-level bound.
        tau: float [default: 2.1]
            Discrepancy multiplier.1.
        max_its: int [default: 1000]
            Iteration cap.
        
        Returns
        -------
        tuple[Vector | None, Vector | None]
            Selected reconstruction and its forward value, when available.
        """
        from regpy.stoprules import CountIterations, Discrepancy

        stoprule = (
            CountIterations(max_iterations=max_its)
            + Discrepancy(
                self.h_codomain.norm,
                data,
                noiselevel=delta,
                tau=tau,
            )
        )
        reco, reco_data = self.run(stoprule)
        if not isinstance(stoprule.active_rule, Discrepancy):
            self.log.warning(
                'Discrepancy principle not satisfied after maximum number '
                'of iterations.'
            )
        return reco, reco_data

    def compute_dual(self) -> None:
        r"""Complete and store primal and dual tuples for a Tikhonov setting.
        
        Uses ``(self.x, self.y)`` and an existing ``self.dual`` tuple (or
        ``None``). Assigns the completed tuples to ``self.primal`` and
        ``self.dual``; these attributes are data, not accessor methods.
        Subclasses using callable primal/dual accessors should not use this
        helper without adapting it.
        
        Raises
        ------
        RuntimeError
            If the setting has no regularization parameter.
        """
        if not self.setting.is_tikhonov:
            raise RuntimeError(Errors.generic_message(
                "It is not possible to compute the dual in the "
                "implementation of this setting"
            ))
        self.primal, self.dual = self.setting._complete_primal_dual_tuples(
            (self.x, self.y), self.dual
        )
    

class CombinedSolver(RegSolver):
    r"""Run two solvers successively, selecting a warm start with a stopping rule.
    
    Parameters
    ----------
    solver1: RegSolver
        Initialized first solver. Its current state is the combined solver's
        initial state; it is not restarted. Do not drive it independently while
        using this wrapper.
    solver2: Callable[..., RegSolver]
        Factory called exactly once as ``solver2(init=x)`` when switching.
        For example, ``lambda *, init: SemiSmoothNewton(setting, init=init)``.
        It must return a new solver for the same setting and coordinates.
        A constructed solver instance is not accepted.
    stoprule: StopRule | None [default: None]
        Rule controlling the first phase. An independent, reset copy is used
        and exposed as ``stoprule``.
        It is checked at solver1's initial state and after each completed step.
        Natural convergence of solver1 also starts the second phase.
    
    Notes
    -----
    The transition calls ``best_iterate()`` once and passes a copy of its
    selected x to the factory. If no best iterate is available, solver1's
    current x is used. Both pair-valued and vector-valued selections are
    supported. A missing solution vector raises ``ValueError``.
    
    The inherited iteration drivers yield read-only views. As for other
    solvers, the initial state of solver1 is available through ``x`` and ``y``
    but is not yielded by ``iter`` or ``while_``. Every subsequent first-phase
    iterate, solver2's initial state, and every second-phase iterate is exposed.
    The transition counts as one combined iteration, even if solver2's initial
    state equals the warm start. An exterior rule can therefore stop at each
    of these states; stopping before the transition avoids constructing solver2.
    
    ``iteration_step_nr`` counts combined steps, independently of the child
    counters. Missing public attributes (including primal/dual accessors and
    ``deriv``, when available) are delegated to ``active_solver``. Both children
    must represent the same setting so external stopping histories remain
    comparable. The wrapper does not restart either child or their counters.
    
    Examples
    --------
    >>> combined = CombinedSolver(
    ...     Landweber(setting),
    ...     lambda *, init: SemiSmoothNewton(setting, init=init),
    ...     stoprule=CountIterations(10),
    ... )
    >>> x, y = combined.run(exterior_stoprule)
    """

    def __init__(
        self,
        solver1: RegSolver,
        solver2: Callable[..., RegSolver],
        stoprule: StopRule | None = None,
    ) -> None:
        from regpy.stoprules import CountIterations

        if not isinstance(solver1, RegSolver):
            raise TypeError("solver1 must be an initialized RegSolver.")
        if isinstance(solver2, Solver) or not callable(solver2):
            raise TypeError("solver2 must be a factory accepting init as a keyword.")
        if stoprule is not None and not isinstance(stoprule, StopRule):
            raise TypeError("stoprule must be a StopRule or None.")
        self.solver1 = solver1
        self.solver2: RegSolver | None = None
        self.active_solver = solver1
        self._solver2_factory = solver2
        self.stoprule = (CountIterations(10) if stoprule is None
                         else stoprule.copy_and_reset())
        # Geometry and optional interfaces are delegated, rather than copied:
        # stage-specific values such as regpar and deriv must not become stale.
        Solver.__init__(self, solver1.x, solver1.y)
        self.stoprule._complete_init_with_solver(solver1)

    @property
    def is_primal(self) -> bool:
        r"""Coordinate label of the currently active child solver.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.active_solver.is_primal

    def __getattr__(self, name: str) -> Any:
        r"""Expose the active child's public interfaces without stale copies.
        
        Parameters
        ----------
        name: str
            Name.
        
        
        Returns
        -------
        Any
            Result described above.
        """
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.active_solver, name)

    def next(self) -> bool:
        r"""Advance a child or expose the second child's initial state once.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        if self.is_converged():
            return False
        if self.solver2 is None:
            stopped = self.stoprule.stop()
            if stopped or self.solver1.is_converged():
                best = self.stoprule.best_iterate()
                x = (self.solver1.x if best is None else
                     best[0] if isinstance(best, tuple) else best)
                if x is None:
                    raise ValueError("The first solver supplied no warm-start vector.")
                child = self._solver2_factory(init=x.copy())
                if not isinstance(child, RegSolver) or child is self.solver1:
                    raise TypeError("solver2 must return a new RegSolver.")
                if child.setting is not self.solver1.setting:
                    raise ValueError("Both solvers must use the same setting instance.")
                self.solver2 = child
                self.active_solver = child
            else:
                self.solver1.next()
        elif not self.solver2.next():
            self.converge()
            return False
        self.x, self.y = self.active_solver.x, self.active_solver.y
        self.iteration_step_nr += 1
        if self.solver2 is not None and self.solver2.is_converged():
            self.converge()
        return True

