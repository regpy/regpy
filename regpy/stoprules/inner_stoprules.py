"""Stopping rules used inside iterative optimization methods.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, TypeAlias, cast

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

Norm: TypeAlias = Callable[[ReadOnlyVector], float]
_EVERY_TENTH_STEP = range(0, 10**18, 10)

__all__: list[str] = [
    "RelativeChangeData", "RelativeChangeSol", "LinearSequence",
    "OptimalityCondStopping", "DualityGapStopping",
]

class RelativeChangeData(StopRule):
    r"""Stop if the relative change in the residual becomes small.

    Stops at the first iterate at which the difference between the old residual
    and the new residual is smaller than a pre-determined tol::

        ||y_k-y_{k+1}|| < tol max(||y_k||,||y_{k+1}||)

    Parameters
    ----------
    norm : callable | None, optional
        The norm with respect to which the difference should be measured.
        By default, ``solver.op.codomain.norm`` is used after a solver is
        attached.
    tol : float, optional
        Nonnegative relative tolerance. Defaults to ``0``.
    schedule : iterable of int or None, optional
        Iterations at which the rule is evaluated. If ``None`` (default), it
        is evaluated at every iteration.
    """

    def __init__(
        self,
        norm: Norm | None = None,
        tol: float = 0.,
        schedule: Schedule = None,
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
        self.history_dict["relative change of y"].append(change/ref)
        self.log_info = 'rel. data change {:.3e}<{:.3e}'.format(change,self.tol*ref)
        triggered =  change < self.tol*ref
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
    norm : callable | None, optional
        The norm with respect to which the difference should be measured.
        By default, ``solver.op.domain.norm`` is used after a solver is
        attached.
    tol : float, optional
        Nonnegative relative tolerance. Defaults to ``0``.
    schedule : iterable of int or None, optional
        Iterations at which the rule is evaluated. If ``None`` (default), it
        is evaluated at every iteration.
    """

    def __init__(
        self,
        norm: Norm | None = None,
        tol: float = 0.,
        schedule: Schedule = None,
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
        self.history_dict["relative change of x"].append(change/ref)
        self.log_info = 'rel. change sol: {:.3e}<{:.3e}'.format(change,self.tol*ref)
        triggered = change < self.tol*ref
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)
        return triggered

######### stopping rules for convex optimization problems #########

class LinearSequence(Iterator[int]):
    r"""Iterator generating a linear sequence ``c, c+a, c+2a, ...``.
    
    Parameters
    ----------
    a : int, optional
        Difference between neighboring sequence elements. Defaults to ``1``.
    start : int, optional
        First element of the sequence. Defaults to ``0``.
    """    
    def __init__(self, a: int = 1, start: int = 0) -> None:
        self.a: int = a
        self.x: int = start

    def __iter__(self) -> LinearSequence:
        return self

    def __next__(self)->int:
        result = self.x
        self.x += self.a
        return result

class OptimalityCondStopping(StopRule):
    r"""Stopping rule based on optimality condition violation.
        
        Parameters
        ----------
        tol : float, optional
            Tolerance for the sum of the two optimality-condition residuals.
            Defaults to ``0``.
        schedule : iterable of int or None, optional
            Iterations at which the condition is evaluated. By default it is
            evaluated at iterations ``0, 10, 20, ...``. Pass ``None`` to
            evaluate it at every iteration.
        logging_level : int | str, optional
            Logging level. Defaults to ``"INFO"``.
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
        triggered = (dSstar+dR<=self.tol)
        self.log_info = '{:.1e} + {:.1e} = {:.2e}  <= {:.1e}'.format(dSstar,dR,dSstar+dR,self.tol)
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)    
        return triggered 
    
class DualityGapStopping(StopRule):
    r"""Stopping rule based on duality gap.

    Parameters
    ----------
    tol : float, optional
        The tolerance for the duality gap.
    stop_at_infty : bool, optional
        Whether an infinite duality gap triggers the rule. Defaults to
        ``True``.
    logging_level : int | str, optional
        Logging level. Defaults to ``"INFO"``.
    schedule : iterable of int or None, optional
        Iterations at which the duality gap is evaluated. By default it is
        evaluated at iterations ``0, 10, 20, ...``. Pass ``None`` to evaluate
        it at every iteration.
    """    
    def __init__(
        self,
        tol: float = 0.,
        stop_at_infty: bool = True,
        logging_level: LoggingLevel = "INFO",
        schedule: Schedule = _EVERY_TENTH_STEP,
    ) -> None:
        super().__init__(
            logging_level=logging_level,
            schedule=schedule,
        )
        self.tol: float = float(tol)
        self.history_dict["duality gap"] = []
        self.stop_at_infty: bool = stop_at_infty

    def __repr__(self) -> str:
        return 'DualityGapStopping(tol={})'.format(
            self.tol)
    
    def _complete_init_with_solver(self, solver: Solver) -> None:
        reg_solver = cast("RegSolver", solver)
        if not (reg_solver.setting.is_tikhonov
                and reg_solver.setting.is_convex):
            raise RuntimeError(Errors.generic_message("It is not possible to compute the dual in the implementation of this setting. The setting needs to be convex and contain a regularization parameter!"))
        reg_solver.setting.duality_gap_warning_incr = 1.
        super()._complete_init_with_solver(solver)

    def _stop(self) -> bool:
        if self.solver is None:
            raise MissingValueError("No solver is attached to this rule.")
        solver = cast("RegSolver", self.solver)
        primal = solver.primal() if hasattr(solver,"primal") and callable(solver.primal) else None
        dual = solver.dual() if hasattr(solver,"dual") and callable(solver.dual) else None
        if primal is None and dual is None:
            raise RuntimeError(Errors.generic_message("The solver needs to provide at least one of the methods 'primal' or 'dual'."))        
        gap = solver.setting.duality_gap(primal = primal, dual = dual)
        self.history_dict["duality gap"].append(gap)
        triggered = (
            gap <= self.tol
            or np.isnan(gap)
            or (np.isinf(gap) and self.stop_at_infty)
        )
        if np.isnan(gap):
            self.log_info = 'duality gap is NaN'
        else:
            self.log_info ='duality gap:{:.2e} <= {:.1e}'.format(gap,self.tol)
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)    
        return triggered
