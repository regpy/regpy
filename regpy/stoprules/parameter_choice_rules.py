"""Stopping rules for choosing regularization parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from regpy.solvers.setting import Setting
    from regpy.solvers.solver import RegSolver

import logging
from collections.abc import Callable
from numbers import Real
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

import numpy as np

from regpy.operators import ObsDataT
from regpy.util import Errors
from regpy.vecsps import ReadOnlyVector, Vector
from .general import LoggingLevel, Schedule, MissingValueError, StopRule, _copy_vector


Norm: TypeAlias = Callable[[ReadOnlyVector], float]
DistanceFunction: TypeAlias = Callable[
    [ReadOnlyVector, ReadOnlyVector], float
]
ErrorPropagation: TypeAlias = Callable[[float], float]

__all__: list[str] = [
    "Discrepancy", "GeneralizedDiscrepancy", "LCurve", "QuasiOpt", "Lepskii", "Oracle",
]

class Discrepancy(StopRule):
    r"""Morozov's discrepancy principle.
    
    Stops at the first iterate at which the residual is smaller than a
    pre-determined multiple of the noise level::
    
        ||y - data|| < tau * noiselevel
    
    Parameters
    ----------
    noiselevel: float | None [default: None]
        Positive noise estimate. If omitted, read ``absolute_noise_level``
        from the attached setting's ``_noise_model_param``. Data must also
        be present in that setting; these requirements are checked on attachment.
    tau: float [default: 2.1]
        Multiplier of the noise level.
    norm: Norm | None [default: None]
        The norm with respect to which the discrepancy should be measured or
        ``None``.
        In the default case, ``setting.h_codomain.norm`` is used.
    noise_level_is_relative: bool [default: False]
        Whether an explicitly supplied ``noiselevel`` is relative to the norm of
        the observations. A noise level inferred from the setting is absolute.
    schedule: Schedule | None [default: None]
        Iterations at which the rule is evaluated. If ``None``, it
        is evaluated at every iteration.
    max_residual_increase: float | None [default: None]
        Optional divergence safeguard: stop without convergence if the residual
        exceeds this factor times its smallest recorded value and the discrepancy
        criterion is not satisfied. Must exceed one. None disables the safeguard,
        since residuals need not decrease monotonically for every method.
    logging_level: LoggingLevel [default: 'INFO']
        Logging level, as a level name or integer logging constant.
    """
    def __init__(
        self,
        noiselevel: float | None = None,
        *,
        tau: float = 2.1,
        norm: Norm | None = None,
        noise_level_is_relative: bool = False,
        schedule: Schedule | None = None,
        max_residual_increase: float | None = None,
        logging_level: LoggingLevel = "INFO",
    ) -> None:
        super().__init__(logging_level=logging_level, schedule=schedule)
        if not isinstance(noise_level_is_relative, bool):
            raise TypeError("noise_level_is_relative must be boolean.")
        if noiselevel is not None:
            self._validate_noiselevel(noiselevel)
        if isinstance(tau, (bool, np.bool_)) or not isinstance(tau, Real):
            raise TypeError("tau must be a finite real number.")
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive.")
        if tau <= 1:
            self.log.warning("The multiplier in the discrepancy principle should be bigger than one!")
        if norm is not None and not callable(norm):
            raise TypeError("norm must be callable or None.")
        if max_residual_increase is not None:
            if (isinstance(max_residual_increase, (bool, np.bool_))
                    or not isinstance(max_residual_increase, Real)):
                raise TypeError("max_residual_increase must be a real number or None.")
            if not np.isfinite(max_residual_increase) or max_residual_increase <= 1:
                raise ValueError("max_residual_increase must be finite and greater than one, or None.")
        self._requested_noiselevel = noiselevel
        self._requested_norm = norm
        self.noise_level_is_relative = noise_level_is_relative
        self.noiselevel = noiselevel
        self.norm = norm
        self.data = None
        self.tau = tau
        self.tol = tau
        self.max_residual_increase = max_residual_increase
        self._min_residual = np.inf
        self.history_dict["relative discrepancy"] = []

    @staticmethod
    def _validate_noiselevel(value: float) -> None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("noiselevel must be a finite positive real number.")
        if not np.isfinite(value) or value <= 0:
            raise ValueError("noiselevel must be finite and positive.")

    def _complete_init_with_solver(self, solver: RegSolver) -> None:
        """Resolve data, norm and noise from the attached solver's setting."""
        setting = getattr(solver, "setting", None)
        if setting is None:
            raise ValueError("Discrepancy requires a solver with a setting.")
        data = getattr(setting, "data", None)
        if data is None:
            raise ValueError("Discrepancy requires observation data in the attached setting.")
        norm = self._requested_norm
        if norm is None:
            norm = setting.h_codomain.norm
        noise = self._requested_noiselevel
        if noise is None:
            noise = getattr(setting, "_noise_model_param", {}).get("absolute_noise_level")
            if noise is None:
                raise ValueError("Discrepancy requires noiselevel or setting._noise_model_param['absolute_noise_level'].")
        elif self.noise_level_is_relative:
            noise = noise * norm(cast(ReadOnlyVector, data))
        self._validate_noiselevel(noise)
        super()._complete_init_with_solver(solver)
        self.data = data
        self.norm = norm
        self.noiselevel = float(noise)

    def reset(self) -> None:
        """Reset the decision, history and smallest observed residual."""
        super().reset()
        self._min_residual = np.inf

    def __repr__(self) -> str:
        return 'Discrepancy(noiselevel={}, tau={})'.format(
            self.noiselevel, self.tau)

    def _stop(self) -> bool:
        if self.solver is None or self.solver.y is None:
            raise MissingValueError
        residual = cast(ReadOnlyVector, self.data) - self.solver.y
        discrepancy = self.norm(residual)
        rel = discrepancy / self.noiselevel
        self.history_dict["relative discrepancy"].append(rel)
        threshold = self.tau * self.noiselevel
        # Report the physical residual and its absolute threshold, including
        # for non-triggering iterations. The stopping inequality is strict.
        self.log_info = (
            f"residual = {discrepancy:.3e}; "
            f"stop if residual < {threshold:.3e} "
            f"(tau={self.tau:g}, noiselevel={self.noiselevel:.3e})"
        )
        self.log.info(self.log_info)
        if not np.isfinite(discrepancy):
            return self._set_stop_status(False, "Nonfinite discrepancy.", failure=True)
        self._min_residual = min(self._min_residual, discrepancy)
        triggered = rel < self.tau
        if (not triggered and self.max_residual_increase is not None
                and discrepancy > self.max_residual_increase * self._min_residual):
            return self._set_stop_status(False,
                f"Residual increased by more than a factor {self.max_residual_increase} "
                "relative to its smallest recorded value.", failure=True)
        if triggered:
            self._set_stop_status(True, "Discrepancy principle satisfied: " + self.log_info)
        return triggered

class GeneralizedDiscrepancy(StopRule):
    r"""Stop at the first iterate satisfying S(y_k) < threshold.

    S is the attached solver's ``setting.data_fid`` and y_k is its current
    operator image ``solver.y``. The data fidelity is evaluated as configured,
    including data shifts and any additive constants; no regularization
    parameter or penalty term enters this criterion.

    Parameters
    ----------
    threshold: float
        Finite threshold for the data-fidelity value. Equality does not stop
        the iteration. Negative thresholds are allowed for shifted functionals.

    schedule: Schedule | None [default: None]
        Iterations at which the fidelity is evaluated and recorded.
        None checks every iteration.
    logging_level: LoggingLevel [default: 'INFO']
        Logging level, as a level name or integer logging constant.
    residual_increase: float | None [default: None]
        Optional divergence safeguard, finite and greater than one. Unless
        the threshold is satisfied, stop without convergence when the fidelity
        exceeds its recorded minimum m by more than
        (residual_increase - 1) * abs(m). For nonnegative fidelities this is
        equivalent to S(y_k) > residual_increase * m. None disables it.

    Notes
    -----
    Every evaluation is recorded in ``history_dict["data fidelity"]``.
    The base class records iteration numbers in ``history_step_dict`` and
    clears both histories on reset. The schedule controls whether the initial
    iterate is checked; without a schedule it is checked in a while-style run. An unavailable operator image raises MissingValueError,
    allowing a combined rule to request it from the solver.

    For S(y) = ||y - data||^2 / 2 and
    threshold = (tau * noiselevel)^2 / 2, the criterion agrees with
    Discrepancy using the same norm and positive noise level (without its
    optional divergence safeguard). To match a residual-norm increase factor
    c in Discrepancy, use residual_increase=c**2 for this quadratic fidelity.
    Nonfinite fidelity values stop with failure, as in Discrepancy.
    """

    def __init__(self, threshold: float, schedule: Schedule | None = None,
                 logging_level: LoggingLevel = "INFO",
                 residual_increase: float | None = None) -> None:
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
            raise TypeError("threshold must be a finite real number.")
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite.")
        if residual_increase is not None:
            if (isinstance(residual_increase, (bool, np.bool_))
                    or not isinstance(residual_increase, Real)):
                raise TypeError("residual_increase must be a real number or None.")
            if not np.isfinite(residual_increase) or residual_increase <= 1:
                raise ValueError("residual_increase must be finite and greater than one, or None.")
        super().__init__(logging_level=logging_level, schedule=schedule)
        self.residual_increase = residual_increase
        self._min_residual: float = np.inf
        self.threshold: float = float(threshold)
        self.history_dict["data fidelity"] = []

    def reset(self) -> None:
        """Clear the history and smallest observed fidelity value."""
        super().reset()
        self._min_residual = np.inf

    def __repr__(self) -> str:
        return f"GeneralizedDiscrepancy(threshold={self.threshold!r})"

    def _stop(self) -> bool:
        if self.solver is None or self.solver.y is None:
            raise MissingValueError
        solver = cast("RegSolver", self.solver)
        value = float(solver.setting.data_fid(solver.y))
        self.history_dict["data fidelity"].append(value)
        triggered = value < self.threshold
        self.log_info = f"data fidelity = {value:.6g}; threshold = {self.threshold:.6g}"
        if not np.isfinite(value):
            return self._set_stop_status(False, "Nonfinite data fidelity.", failure=True)
        self._min_residual = min(self._min_residual, value)
        if (not triggered and self.residual_increase is not None
                and value - self._min_residual
                > (self.residual_increase - 1) * abs(self._min_residual)):
            return self._set_stop_status(
                False,
                f"Data fidelity increased beyond factor {self.residual_increase} "
                f"relative to its smallest recorded value ({self._min_residual:.6g}).",
                failure=True)
        if triggered:
            self._set_stop_status(
                True, "Generalized discrepancy principle satisfied: " + self.log_info)
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG, self.log_info)
        return triggered


class LCurve(StopRule):
    r"""L Curve method.
    
    Computes ``||x||`` and ``||y-data||`` for all available parameters and
    returns the iterate minimizing their product (the currently implemented
    discrete L-curve criterion).
    
    Parameters
    ----------
    setting: Setting
        Tikhonov setting supplying the data and Hilbert norms.
    solver: RegSolver
        Solver used for computing the reconstructions.
    max_iter: int [default: 1000]
        Maximum number of regularization parameters.
    schedule: Schedule | None [default: None]
        Iterations at which the rule records and evaluates a parameter. If
        ``None``, every iteration is used.
    logging_level: LoggingLevel [default: 'INFO']
        Logging level, as a level name or integer logging constant.
    """
    def __init__(
        self,
        setting: Setting,
        solver: RegSolver,
        max_iter: int = 1000,
        schedule: Schedule | None = None,
        logging_level: LoggingLevel = "INFO",
    ) -> None:
        super().__init__(logging_level=logging_level, schedule=schedule)
        self.data: ReadOnlyVector | ObsDataT | None = setting.data
        self.setting: Setting = setting
        self.data_norm: Norm = setting.h_codomain.norm
        self.solution_norm: Norm = setting.h_domain.norm
        self.solver = solver
        self.history_dict["residual"] = []
        self.history_dict["norm"] = []
        self.history_dict["alphas"] = []
        self.recos: list[Vector] = []
        self.max_iter: int = max_iter
        self.it: int = 0

    def __repr__(self) -> str:
        return 'L Curve'

    def _stop(self) -> bool:
        self.it +=1
        if self.solver is None or self.solver.y is None:
            raise MissingValueError
        if self.solver.x is None:
            raise MissingValueError
        solver = cast("RegSolver", self.solver)
        if self.data is None:
            raise MissingValueError("The setting contains no observation data.")
        residual = cast(ReadOnlyVector, self.data) - solver.y
        norm_res = self.data_norm(residual)
        norm_x = self.solution_norm(solver.x)
        self.history_dict["residual"].append(norm_res)
        self.history_dict["norm"].append(norm_x)     
        self.history_dict["alphas"].append(self.setting.regpar)
        self.recos.append(_copy_vector(solver.op.domain, solver.x))
        self.log_info = 'res {:.3e},norm {:.3e}'.format(norm_res,norm_x)
        if self.it >= self.max_iter:
            return self._set_stop_status(False,
                f"{type(self).__name__}: parameter search cap reached ({self.max_iter}); "
                "best recorded iterate selected without a convergence certificate.")
        return False
   
    def best_stopping_index(self) -> int:
        r"""Best stopping index.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        res = self.history_dict["residual"]
        norm_x = self.history_dict["norm"]
        #alphas = self.history_dict["alphas"]
        #xi = np.log(res)
        #eta = np.log(norm_x)
        #dxi = np.gradient(xi,alphas)
        #d2xi = np.gradient(dxi,alphas)
        #deta = np.gradient(eta,alphas)
        #d2eta = np.gradient(deta,alphas)
        #kappa = (d2xi*deta - dxi*d2eta)/(dxi**2 + deta**2)**(3/2)
        #dydx = np.gradient(eta,xi)
        #d2yd2x = np.gradient(dydx,xi)
        #kappa = d2yd2x/(1+dydx**2)**(3/2)
        #return np.argmax(abs(kappa))
        return int(np.argmin(np.array(res)*np.array(norm_x)))
        
    def best_iterate(self) -> Vector:
        r"""Best iterate.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        return self.recos[self.best_stopping_index()]

class QuasiOpt(StopRule):
    r"""Quasi-optimality principle.
    
    Computes x for all available parameters
    and returns as best iterate that x_{k+1} where the ||x_{k+1} - x_{k}|| 
    is minimal
    
    Parameters
    ----------
    setting: Setting
        Tikhonov setting supplying the domain norm.
    solver: RegSolver
        Solver used for computing the reconstructions.
    max_iter: int [default: 1000]
        Maximum number of regularization parameters.
    schedule: Schedule | None [default: None]
        Iterations at which the rule records and evaluates a parameter. If
        ``None``, every iteration is used.
    logging_level: LoggingLevel [default: 'INFO']
        Logging level, as a level name or integer logging constant.
    """
    def __init__(
        self,
        setting: Setting,
        solver: RegSolver,
        max_iter: int = 1000,
        schedule: Schedule | None = None,
        logging_level: LoggingLevel = "INFO",
    ) -> None:
        super().__init__(logging_level=logging_level, schedule=schedule)
        self.data: ReadOnlyVector | ObsDataT | None = setting.data
        self.norm: Norm = setting.h_domain.norm
        self.solver = solver
        self.history_dict["norm_diff"] = []
        self.history_dict["alphas"] = []
        self.recos: list[Vector] = []
        self.max_iter: int = max_iter
        self.it: int = 0

    def __repr__(self) -> str:
        return 'Quasi optimality'

    def _stop(self) -> bool:
        self.it +=1
        if self.solver is None or self.solver.y is None:
            raise MissingValueError
        if self.solver.x is None:
            raise MissingValueError
        solver = cast("RegSolver", self.solver)
        if self.it > 1:
            norm_diff = self.norm(solver.x-self.recos[-1])
        else:
            norm_diff = 0
        self.history_dict["norm_diff"].append(norm_diff)
        self.history_dict["alphas"].append(solver.setting.regpar)
        self.recos.append(_copy_vector(solver.op.domain, solver.x))
        self.log_info = 'norm_diff {:.3e}'.format(norm_diff)
        if self.it >= self.max_iter:
            return self._set_stop_status(False,
                f"{type(self).__name__}: parameter search cap reached ({self.max_iter}); "
                "best recorded iterate selected without a convergence certificate.")
        return False
    
    def best_stopping_index(self) -> int:
        r"""Best stopping index.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        norm_diff = self.history_dict["norm_diff"]
        return int(np.argmin(norm_diff[1:])) + 1

    def best_iterate(self) -> Vector:
        r"""Best iterate.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        return self.recos[self.best_stopping_index()]
    
class Lepskii(StopRule):
    r"""Lepskiĭ's balancing principle.
    
    The rule computes reconstructions for all considered parameters. Before
    applying the balancing criterion, it sorts the regularization parameters
    increasingly. The supplied error-propagation estimate must be decreasing
    along that order.
    
    Parameters
    ----------
    setting: Setting
        Tikhonov setting supplying the domain norm.
    solver: RegSolver
        Solver used for computing the reconstructions.
    error_prop: ErrorPropagation | None [default: None]
        Error-propagation estimate as a function of the regularization
        parameter. Although the argument
    max_iter: int [default: 100]
        Maximum number of regularization parameters.
    schedule: Schedule | None [default: None]
        Iterations at which the rule records and evaluates a parameter. If
        ``None``, every iteration is used.
    logging_level: LoggingLevel [default: 'INFO']
        Logging level, as a level name or integer logging constant.
    """

    def __init__(
        self,
        setting: Setting,
        solver: RegSolver,
        error_prop: ErrorPropagation | None = None,
        max_iter: int = 100,
        schedule: Schedule | None = None,
        logging_level: LoggingLevel = "INFO",
    ) -> None:
        super().__init__(logging_level=logging_level, schedule=schedule)
        self.data: ReadOnlyVector | ObsDataT | None = setting.data
        self.norm: Norm = setting.h_domain.norm
        self.solver = solver
        self.history_dict["alphas"] = []
        self.recos: list[Vector] = []
        self.max_iter: int = max_iter
        self.it: int = 0
        if error_prop is None:
            raise ValueError(Errors.value_error(
                'Lepskii requires an error_prop callable; automatic '
                'estimation is not implemented.'
            ))
        self.Psi: ErrorPropagation = error_prop

    def __repr__(self) -> str:
        return 'Lepskii'

    def _stop(self) -> bool:
        self.it += 1
        if self.solver is None or self.solver.y is None:
            raise MissingValueError
        if self.solver.x is None:
            raise MissingValueError
        solver = cast("RegSolver", self.solver)
        self.history_dict["alphas"].append(solver.setting.regpar)
        self.recos.append(_copy_vector(solver.op.domain, solver.x))
        self.log_info = 'iter {:d}, alpha {:.3e}'.format(
            self.it, solver.setting.regpar
        )
        if self.it >= self.max_iter:
            return self._set_stop_status(False,
                f"{type(self).__name__}: parameter search cap reached ({self.max_iter}); "
                "best recorded iterate selected without a convergence certificate.")
        return False

    def best_stopping_index(self) -> int:
        r"""Best stopping index.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        alphas = np.asarray(self.history_dict["alphas"])
        if not np.all(alphas[:-1] <= alphas[1:]):
            idx = np.argsort(alphas)
        else:
            idx = np.arange(len(alphas))

        bark = 1
        while bark <= self.max_iter - 1:
            l = 0
            while l < bark:
                if self.norm(
                    self.recos[idx[bark]] - self.recos[idx[l]]
                ) >= 4 * self.Psi(float(alphas[idx[l]])):
                    break
                l += 1
            if l <= bark - 1:
                break
            bark += 1
        return int(idx[bark - 1])

    def best_iterate(self) -> Vector:
        r"""Best iterate.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        return self.recos[self.best_stopping_index()]


class Oracle(StopRule):
    r"""Oracle stoprule. Returns the iterate that is closest to the exact solution in terms of the given distance function.
    Useful for testing purposes and monitoring when the exact solution is known.
    
    
    Parameters
    ----------
    setting: Setting
        Setting whose operator domain contains the exact solution.
    exact_solution: ReadOnlyVector | None [default: None]
        Exact solution. If omitted, ``setting.exact_solution`` is used.
    distance_function: DistanceFunction | None [default: None]
        The distance function to measure the distance between the current iterate and the exact solution. 
        distance_function(x, exact_solution) -> float
    schedule: Schedule | None [default: None]
        Iterations at which errors and reconstructions are recorded. If
        ``None``, every iteration is used.
    logging_level: LoggingLevel [default: 'INFO']
        Logging level, as a level name or integer logging constant.
    record_operator_values: bool [default: False]
        Store a copy of the operator value with each recorded reconstruction.
        If the solver does not provide it, evaluate its operator at that point.
        The selected result is ``(x, y)`` when enabled and ``(x, None)``
        otherwise; disabled recording requires no additional operator calls.
    """
    selects_best_iterate: bool = True

    def __init__(
        self,
        setting: Setting,
        exact_solution: ReadOnlyVector | None = None,
        distance_function: DistanceFunction | None = None,
        schedule: Schedule | None = None,
        logging_level: LoggingLevel = "INFO",
        record_operator_values: bool = False,
    ) -> None:
        if not isinstance(record_operator_values, bool):
            raise TypeError('record_operator_values must be a bool.')
        self.record_operator_values = record_operator_values
        self.operator_values: list[Vector] = []
        self._recorded_steps: list[int] = []
        super().__init__(logging_level=logging_level, schedule=schedule)
        self.data: ReadOnlyVector | ObsDataT | None = setting.data
        self.history_dict["error"] = []
        self.history_dict["alphas"] = []
        self.recos: list[Vector] = []
        if not hasattr(setting,'exact_solution') and exact_solution is None:  
            raise ValueError(Errors.value_error('Oracle stoprule needs the exact solution to be provided in the setting!'))
        if exact_solution is not None:
            setattr(setting, 'exact_solution', exact_solution)
        self.exact_solution: ReadOnlyVector = cast(
            ReadOnlyVector, getattr(setting, 'exact_solution')
        )
        if distance_function is None:
            self.dist: Callable[[ReadOnlyVector], float] = (
                lambda x: setting.op.domain.norm(x - self.exact_solution)
            )
        else:
            self.dist = lambda x: distance_function(x, self.exact_solution)

    def reset(self) -> None:
        """Clear stored reconstructions and operator values along with history."""
        super().reset()
        self.recos.clear()
        self.operator_values.clear()
        self._recorded_steps.clear()

    def __repr__(self) -> str:
        return 'Oracle'

    def _stop(self) -> bool:
        if self.solver is None or self.solver.x is None:
            raise MissingValueError     
        solver = cast("RegSolver", self.solver)
        error = self.dist(solver.x)
        reconstruction = _copy_vector(solver.op.domain, solver.x)
        operator_value = None
        if self.record_operator_values:
            value = getattr(solver, 'y', None)
            if value is None:
                value = solver.op(solver.x)
            operator_value = _copy_vector(solver.op.codomain, value)
        # Append only after all computations succeed, keeping records aligned.
        self.history_dict["error"].append(error)
        self.history_dict["alphas"].append(
            getattr(solver, 'regpar', getattr(solver.setting, 'regpar', None))
        )
        self.recos.append(reconstruction)
        self._recorded_steps.append(solver.iteration_step_nr)
        if operator_value is not None:
            self.operator_values.append(operator_value)
        self.log_info = 'error {:.3e}'.format(error)
        return False

    def best_stopping_index(self) -> int:
        r"""Best stopping index.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        if not self.history_dict["error"]:
            raise RuntimeError('Oracle has not recorded any iterates.')
        best = np.argmin(self.history_dict["error"])
        self.log.info('selected iterate nr. %s as best iterate',
                      self._recorded_steps[best])

        return int(best)

    def best_iterate(self) -> tuple[Vector, Vector | None] | None:
        r"""Return the recorded reconstruction with the smallest error.

        Returns
        -------
        tuple[Vector, Vector | None] | None
            Selected reconstruction and its recorded operator value, or
            ``(x, None)`` if operator values were not recorded. Returns None
            if no iterates have been recorded. Selection is independent of
            which stopping rule triggered.
        """
        if not self.recos:
            return None
        best = self.best_stopping_index()
        return (self.recos[best],
                self.operator_values[best] if self.record_operator_values else None)
    


from .general import CountIterations_factory, _capped_factory


def Discrepancy_factory(setting: Setting, params: dict) -> StopRule:
    r"""Create Discrepancy plus an iteration cap (timing and counts enabled).
    
    Missing noise levels are resolved when a solver attaches; tau defaults
    to 2.1. rule_params can override constructor options, including noiselevel.
    The default selector handles unavailable noise and codomain-free settings.
    
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
    options = {key: params[key] for key in
               ('tau', 'norm', 'noise_level_is_relative', 'schedule',
                'max_residual_increase', 'logging_level')
               if key in params}
    options.setdefault('tau', 2.1)
    options.update(params.get('rule_params', {}))
    noise = options.pop('noiselevel', params.get('noiselevel', params.get('absolute_noise_level')))
    rule = Discrepancy(noise, **options)
    return rule + CountIterations_factory(setting, params)

GeneralizedDiscrepancy_factory = _capped_factory(GeneralizedDiscrepancy)
LCurve_factory = _capped_factory(LCurve)
QuasiOpt_factory = _capped_factory(QuasiOpt)
Lepskii_factory = _capped_factory(Lepskii)
Oracle_factory = _capped_factory(Oracle)
__all__ += ['Discrepancy_factory', 'GeneralizedDiscrepancy_factory', 'LCurve_factory', 'QuasiOpt_factory', 'Lepskii_factory', 'Oracle_factory']
