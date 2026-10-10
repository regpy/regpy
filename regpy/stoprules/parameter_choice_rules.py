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
    noiselevel: float
        An estimate of the distance from the noisy data to the exact data.
    tau: float [default: 2.0]
        Multiplier of the noise level.
    setting: Setting | None [default: None]
        Setting providing data and the codomain norm. If omitted, ``data`` and
        ``norm`` must be given explicitly.
    data: ReadOnlyVector | ObsDataT | None [default: None]
        The right-hand side (noisy data) or ``None``.
        By default, ``setting.data`` is used. The discrepancy calculation
        requires the selected representation to support subtraction from the
        solver's codomain vector.
    norm: Norm | None [default: None]
        The norm with respect to which the discrepancy should be measured or
        ``None``.
        In the default case, ``setting.h_codomain.norm`` is used.
    noise_level_is_relative: bool [default: False]
        Whether ``noiselevel`` is relative to the norm of the observations.
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
    @overload
    def __init__(
        self,
        noiselevel: float,
        *,
        tau: float = 2.,
        setting: Setting | None = None,
        data: ReadOnlyVector | ObsDataT | None = None,
        norm: Norm | None = None,
        noise_level_is_relative: bool = False,
        schedule: Schedule | None = None,
        max_residual_increase: float | None = None,
        logging_level: LoggingLevel = "INFO",
    ) -> None: ...

    @overload
    def __init__(
        self,
        norm: Norm,
        data: ReadOnlyVector | ObsDataT,
        noiselevel: float,
        *,
        tau: float = 2.,
        setting: Setting | None = None,
        noise_level_is_relative: bool = False,
        schedule: Schedule | None = None,
        max_residual_increase: float | None = None,
        logging_level: LoggingLevel = "INFO",
    ) -> None: ...

    @overload
    def __init__(
        self,
        norm: Norm,
        data: ReadOnlyVector | ObsDataT,
        *,
        noiselevel: float,
        tau: float = 2.,
        setting: Setting | None = None,
        noise_level_is_relative: bool = False,
        schedule: Schedule | None = None,
        max_residual_increase: float | None = None,
        logging_level: LoggingLevel = "INFO",
    ) -> None: ...

    def __init__(
        self, *args: Any, logging_level: LoggingLevel = "INFO", **kwargs: Any
    ) -> None:
        # Configure logging before emitting constructor/deprecation messages.
        super().__init__(logging_level=logging_level, schedule=kwargs.get('schedule'))
        defaults: dict[str, Any] = {
            'tau': 2., 'setting': None, 'data': None, 'norm': None,
            'noise_level_is_relative': False, 'schedule': None,
            'max_residual_increase': None,
        }
        if len(args)==3: 
            self.log.warning('Initialization of Discrepancy with three positional arguments (norm, data, noise_level) deprecated. Use one positional argument (noiselevel) and specifiy norm and data via a keyword argument setting!')
            norm,data,noiselevel = args
            p = {**defaults, **kwargs}
            tau,noise_level_is_relative,setting,schedule = p['tau'],p['noise_level_is_relative'],p['setting'],p['schedule']
        elif len(args)==2:
            self.log.warning('Initialization of Discrepancy with two positional arguments (norm, data) deprecated. Use one positional argument (noiselevel) and specifiy norm and data via a keyword argument setting!')
            norm,data = args
            p = {**defaults, **kwargs}
            tau,noise_level_is_relative,setting,noiselevel,schedule = p['tau'],p['noise_level_is_relative'],p['setting'],p['noiselevel'],p['schedule']
        elif len(args)==1:
            noiselevel = args[0]
            p = {**defaults, **kwargs}
            tau,noise_level_is_relative,setting,norm,data,schedule = p['tau'],p['noise_level_is_relative'],p['setting'],p['norm'],p['data'],p['schedule']
        else:
            raise ValueError(Errors.value_error(
                'Discrepancy must have one recommended positional argument '
                'or two/three deprecated positional arguments.'
            ))
        if not isinstance(noise_level_is_relative,bool):
            raise TypeError(Errors.type_error("noise_level_is_relative must be boolean."))
        if not isinstance(noiselevel,(int,float)):  
            raise TypeError(Errors.type_error(f"The noise level in the discrepancy principle should be real scalar! Got {noiselevel}"))
        if noiselevel<=0:
            raise ValueError(Errors.value_error(f"The noise level in the discrepancy principle needs to be greater than zero! Got {noiselevel}"))
        if not isinstance(tau,(int,float)):
            raise TypeError(Errors.type_error(f"The multiplier in the discrepancy principle should be real scalar! Got {tau}."))
        if tau<=1:
            self.log.warning("The multiplier in the discrepancy principle should be bigger than one!")
        if norm is not None:
            if not callable(norm):
                raise TypeError(Errors.type_error(f"The norm in the discrepancy principle needs to be a callable! Got {norm}."))
            self.norm: Norm = norm
        else:
            if setting is None:
                raise ValueError(Errors.value_error('If setting is not provided, then norm must be provided.')) 
            self.norm = setting.h_codomain.norm
        increase = p['max_residual_increase']
        if increase is not None and (not np.isfinite(increase) or increase <= 1):
            raise ValueError("max_residual_increase must be finite and greater than one, or None.")
        self.max_residual_increase: float | None = increase
        self._min_residual: float = np.inf
        if data is not None:
            self.data: ReadOnlyVector | ObsDataT = data
        else:
            if setting is None or setting.data is None:
                raise ValueError(Errors.value_error('If setting is not provided or setting has no data, then data must be provided.'))
            else:
                self.data = setting.data
            
        if noise_level_is_relative:
            self.noiselevel: float = noiselevel * self.norm(
                cast(ReadOnlyVector, self.data)
            )
        else:
            self.noiselevel = noiselevel
        self.tau: float = tau
        self.tol: float = self.tau
        self.history_dict["relative discrepancy"] = []


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
        self.log_info = 'discr./noiselevel = {:3.2f}< {:1.2f}'.format(rel, self.tau)
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
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)
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
    
    absolute_noise_level defaults to the setting's noise model; tau defaults
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
               ('tau', 'data', 'norm', 'noise_level_is_relative', 'schedule',
                'max_residual_increase', 'logging_level')
               if key in params}
    options.setdefault('tau', 2.1)
    options['setting'] = setting
    options.update(params.get('rule_params', {}))
    noise = options.pop('noiselevel', params.get('absolute_noise_level',
            setting._noise_model_param.get('absolute_noise_level')))
    rule = Discrepancy(noise, **options)
    return rule + CountIterations_factory(setting, params)

GeneralizedDiscrepancy_factory = _capped_factory(GeneralizedDiscrepancy)
LCurve_factory = _capped_factory(LCurve)
QuasiOpt_factory = _capped_factory(QuasiOpt)
Lepskii_factory = _capped_factory(Lepskii)
Oracle_factory = _capped_factory(Oracle)
__all__ += ['Discrepancy_factory', 'GeneralizedDiscrepancy_factory', 'LCurve_factory', 'QuasiOpt_factory', 'Lepskii_factory', 'Oracle_factory']
