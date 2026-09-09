"""Solver lifecycle, generator-based iteration, and stopping-rule drivers."""

from __future__ import annotations

from collections.abc import Generator

from regpy.vecsps import Vector
from regpy.stoprules import StopRule, NoneRule
from regpy.util import ClassLogger, Errors
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from regpy.operators import Operator
    from regpy.functionals import Functional
    from regpy.hilbert import HilbertSpace
    from .setting import Setting

__all__ = ["Solver", "RegSolver"]

class Solver:
    r"""Abstract base class for solvers. Solvers do not implement loops themselves, but are driven by
    repeatedly calling the `next` method. They expose the current iterate and its value as attributes
    `x` and `y`, and can be iterated over, yielding the `(x, y)` tuple on every iteration (which
    may or may not be the same arrays as before, modified in-place).

    Each call to ``iter(solver)`` creates a generator driving the same mutable
    solver state; it does not restart the solver. The solver is iterable but
    does not itself implement ``__next__``. Use ``solver.next()`` for a single
    step or ``next(iter(solver))`` to obtain a pair after a step.

    Yielded vectors are not copied. Copy them when retaining snapshots of
    the iteration history. Both components may be ``None``.

    Convenience methods run the solver with a `regpy.stoprules.StopRule`.

    Subclasses should override the method `_next(self)` to perform a single iteration where the values of 
    the attributes `x` and `y` are updated. The main difference to `next` is that `_next` does not have a
    return value. If the solver converged, `converge` should be called, afterwards `_next` will never be
    called again. Most solvers will probably never converge on their own, but rely on the caller or a
    `regpy.stoprules.StopRule` for termination.

    Parameters
    ----------
    x : Vector | None [default: None]
        Initial argument for iteration. Defaults to None.
    y : Vector | None [default: None]
        Initial value at current iterate. Defaults to None.
    """

    log = ClassLogger()

    def __init__(self, 
                 x:Vector | None = None,
                 y:Vector | None = None) -> None:
        self.x: Vector | None = x
        """The current iterate."""
        self.y: Vector | None = y
        """The value at the current iterate. May be needed by stopping rules, but callers should
        handle the case when it is not available."""
        self.__converged: bool = False
        self.iteration_step_nr: int = 0
        """Current number of iterations performed."""

    def converge(self) -> None:
        """Mark the solver as converged. This is intended to be used by child classes
        implementing the `_next` method.
        """
        self.__converged = True

    def is_converged(self) -> bool:
        """Return whether :meth:`converge` has been called.

        An external stopping rule does not by itself set this flag.
        """
        return self.__converged

    def next(self) -> bool:
        r"""Perform a single iteration.

        Returns
        -------
        boolean
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

    def __iter__(self) -> Generator[tuple[Vector | None, Vector | None], None, None]:
        r"""Generate pairs after successive solver steps, excluding the initial pair.

        Execution starts when the generator is advanced. It ends when
        :meth:`next` returns ``False``. A step that marks convergence still
        yields its resulting pair. No vector copies are made.

        Yields
        ------
        tuple[Vector | None, Vector | None]
            The current pair, with either component possibly unavailable.
        """
        while self.next():
            yield self.x, self.y

    @staticmethod
    def _add_max_iterations(
        stoprule: StopRule | None,
        max_it: int | None,
        *,
        while_type: bool,
    ) -> StopRule:
        """Return ``stoprule`` augmented by an iteration limit."""
        if stoprule is None:
            stoprule = NoneRule()
        elif not isinstance(stoprule, StopRule):
            raise TypeError(Errors.type_error(
                f'stoprule must be a StopRule or None. Got {stoprule!r}.'
            ))

        if max_it is None:
            return stoprule
        if isinstance(max_it, bool) or not isinstance(max_it, int):
            raise TypeError(Errors.type_error(
                f'max_it must be an integer or None. Got {max_it!r}.'
            ))
        if max_it < 0:
            raise ValueError(Errors.value_error(
                f'max_it must be non-negative. Got {max_it}.'
            ))

        from regpy.stoprules import CountIterations

        count_rule = CountIterations(
            max_iterations=max_it,
            while_type=while_type,
        )
        if isinstance(stoprule, NoneRule):
            return count_rule
        return stoprule + count_rule

    def while_(
        self,
        stoprule: StopRule | None = None,
        max_it: int | None = None,
    ) -> Generator[tuple[Vector | None, Vector | None], None, None]:
        r"""Generate current iterates under control of a stopping rule.

        The rule is checked before each attempted step, including at the
        initial state. Every completed step is yielded before the next rule
        check. An initially satisfied rule yields nothing. Calling this method
        creates a generator: initialization and argument validation occur only
        when it is advanced. It accepts no meaningful sent values and returns
        no final value. Yielded vectors are references, not snapshots.

        Parameters
        ----------
        stoprule : regpy.stoprules.StopRule, optional
            The stoprule to be used. If omitted, stopping will only be
            based on the return value of `next` and on ``max_it``.
        max_it : int or None, optional
            Maximum number of iterations. If provided, a
            :class:`regpy.stoprules.CountIterations` rule is combined with
            ``stoprule``. Must be non-negative. Defaults to ``None``.

        Yields
        ------
        tuple[Vector | None, Vector | None]
            The current ``(x, y)`` pair after a step. The stopping rule's best
            iterate is not substituted here; use :meth:`run` to select it.

        Raises
        ------
        TypeError
            If ``stoprule`` is not a stopping rule or ``max_it`` is not an
            integer or ``None``.
        ValueError
            If ``max_it`` is negative.
        """
        stoprule = self._add_max_iterations(
            stoprule, max_it, while_type=True
        )
        stoprule._complete_init_with_solver(self)
        while not stoprule.stop() and self.next(): 
            yield self.x, self.y
        self.log.info('Solver converged after {} iteration.'.format(self.iteration_step_nr))
 


    def until(
        self,
        stoprule: StopRule | None = None,
        max_it: int | None = None,
    ) -> Generator[tuple[Vector | None, Vector | None], None, None]:
        r"""Generate current iterates under control of a stopping rule.

        One step is attempted and the current pair is yielded before the
        first stopping-rule check. If already converged, the solver performs
        no step but still yields its current pair once. Subsequent steps are
        checked and yielded as in :meth:`while_`. Calling this method creates
        a generator; validation is deferred until it is advanced. It accepts
        no meaningful sent values and returns no final value. Vectors are
        yielded by reference.

        Parameters
        ----------
        stoprule : regpy.stoprules.StopRule, optional
            The stoprule to be used. If omitted, stopping will only be
            based on the return value of `next` and on ``max_it``.
        max_it : int or None, optional
            Maximum number of iterations. If provided, a
            :class:`regpy.stoprules.CountIterations` rule is combined with
            ``stoprule``. Must be non-negative. Defaults to ``None``.

        Yields
        ------
        tuple[Vector | None, Vector | None]
            The current ``(x, y)`` pair after a step. The stopping rule's best
            iterate is not substituted here; use :meth:`run` to select it.

        Raises
        ------
        TypeError
            If ``stoprule`` is not a stopping rule or ``max_it`` is not an
            integer or ``None``.
        ValueError
            If ``max_it`` is negative or zero. Since :meth:`until` always
            attempts an initial iteration, use :meth:`while_` or :meth:`run`
            when zero iterations should be allowed.
        """
        stoprule = self._add_max_iterations(
            stoprule, max_it, while_type=False
        )
        if max_it == 0:
            raise ValueError(Errors.value_error(
                'until always performs at least one iteration and therefore '
                'does not accept max_it=0. Use while_ or run instead.'
            ))
        stoprule._complete_init_with_solver(self)
        self.next()
        yield self.x, self.y
        while not stoprule.stop() and self.next(): 
            yield self.x, self.y

        self.log.info('Solver converged after {} iteration.'.format(self.iteration_step_nr))

    def run(
        self,
        stoprule: StopRule | None = None,
        max_it: int | None = None,
    ) -> tuple[Vector | None, Vector | None]:
        r"""Run the solver with the given stoprule.

        Parameters
        ----------
        stoprule : regpy.stoprules.StopRule, optional
            The stoprule to be used. If omitted, stopping will only be based
            on convergence of the solver or ``max_it``.
        max_it : int or None, optional
            Maximum number of iterations. If provided, a
            :class:`regpy.stoprules.CountIterations` rule is combined with
            ``stoprule``. Must be non-negative. Defaults to ``None``.

        Returns
        -------
        tuple[Vector | None, Vector | None]
            The pair selected by ``stoprule.best_iterate()``, or the current
            pair if no best iterate is available. No copies are made. With
            zero permitted steps the initial state can be returned.

        Raises
        ------
        TypeError
            If ``stoprule`` is not a stopping rule or ``max_it`` is not an
            integer or ``None``.
        ValueError
            If ``max_it`` is negative.
        """
        stoprule = self._add_max_iterations(
            stoprule, max_it, while_type=True
        )
        for x, y in self.while_(stoprule):
            if not 'x' in locals(): 
                # This happens if the stopping criterion is satisfied for the initial guess.
                x = self.x
                y = self.y
        if stoprule.best_iterate() is None:
            self.log.info(f"Could not find a best iterate with stop rule {stoprule}!")
            return self.x, self.y
        return stoprule.best_iterate()


class RegSolver(Solver):
    r"""Base class for solvers operating on a regularization setting.

    The import of :class:`regpy.solvers.setting.Setting` is intentionally
    deferred until initialization. This keeps ``solver`` independent while
    allowing ``setting`` and concrete solver modules to import ``RegSolver``
    without creating a circular module import.

    Parameters
    ----------
    setting : regpy.solvers.setting.Setting
        Setting used by the solver.
    x : Vector or None [default: None]
        Initial domain iterate.
    y : Vector or None [default: None]
        Value at the initial iterate, if available.
    """

    def __init__(
        self,
        setting: "Setting",
        x: Vector | None = None,
        y: Vector | None = None,
    ) -> None:
        from .setting import Setting

        if not isinstance(setting, Setting):
            raise TypeError(Errors.not_instance(setting, Setting))
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

    def runWithDP(
        self,
        data: Vector,
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
        data : Vector
            Observed data used to measure the residual in ``h_codomain``.
        delta : float, optional
            Noise-level bound. Defaults to zero.
        tau : float, optional
            Discrepancy multiplier. Defaults to 2.1.
        max_its : int, optional
            Iteration cap. Defaults to 1000.

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
        """Complete and store primal and dual tuples for a Tikhonov setting.

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
    
