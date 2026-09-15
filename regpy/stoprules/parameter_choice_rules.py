"""Stopping rules for choosing regularization parameters.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

import numpy as np

from regpy.operators import ObsDataT
from regpy.util import Errors
from regpy.vecsps import ReadOnlyVector, Vector
from .general import Schedule, MissingValueError, StopRule, _copy_vector

if TYPE_CHECKING:
    from regpy.solvers import RegSolver, Setting

Norm: TypeAlias = Callable[[ReadOnlyVector], float]
DistanceFunction: TypeAlias = Callable[
    [ReadOnlyVector, ReadOnlyVector], float
]
ErrorPropagation: TypeAlias = Callable[[float], float]

__all__: list[str] = [
    "Discrepancy", "LCurve", "QuasiOpt", "Lepskii", "Oracle",
]

class Discrepancy(StopRule):
    r"""Morozov's discrepancy principle.

    Stops at the first iterate at which the residual is smaller than a
    pre-determined multiple of the noise level::

        ||y - data|| < tau * noiselevel

    Parameters
    ----------
    noiselevel : float
        An estimate of the distance from the noisy data to the exact data.
    tau : float, optional
        Multiplier of the noise level. Defaults to ``2`` and should be greater
        than one.
    setting : Setting | None, optional
        Setting providing data and the codomain norm. If omitted, ``data`` and
        ``norm`` must be given explicitly.
    data : ReadOnlyVector | ObsDataT | None, optional
        The right-hand side (noisy data) or ``None`` (default).
        By default, ``setting.data`` is used. The discrepancy calculation
        requires the selected representation to support subtraction from the
        solver's codomain vector.
    norm : callable | None, optional
        The norm with respect to which the discrepancy should be measured or
        ``None`` (default).
        In the default case, ``setting.h_codomain.norm`` is used.
    noise_level_is_relative : bool, optional
        Whether ``noiselevel`` is relative to the norm of the observations.
        Defaults to ``False``.
    schedule : iterable of int or None, optional
        Iterations at which the rule is evaluated. If ``None`` (default), it
        is evaluated at every iteration.
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
        schedule: Schedule = None,
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
        schedule: Schedule = None,
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
        schedule: Schedule = None,
    ) -> None: ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        defaults: dict[str, Any] = {
            'tau': 2., 'setting': None, 'data': None, 'norm': None,
            'noise_level_is_relative': False, 'schedule': None,
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
        super().__init__(schedule=schedule)
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
        triggered =  rel < self.tau
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)
        return triggered

class LCurve(StopRule):
    r"""L Curve method.

    Computes ``||x||`` and ``||y-data||`` for all available parameters and
    returns the iterate minimizing their product (the currently implemented
    discrete L-curve criterion).

    Parameters
    ----------
    setting : Setting
        Tikhonov setting supplying the data and Hilbert norms.
    solver : RegSolver
        Solver used for computing the reconstructions.
    max_iter : int, optional
        Maximum number of regularization parameters. Defaults to ``1000``.
    schedule : iterable of int or None, optional
        Iterations at which the rule records and evaluates a parameter. If
        ``None`` (default), every iteration is used.
    """
    def __init__(
        self,
        setting: Setting,
        solver: RegSolver,
        max_iter: int = 1000,
        schedule: Schedule = None,
    ) -> None:
        super().__init__(schedule=schedule)
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
        return self.it >= self.max_iter
   
    def best_stopping_index(self) -> int:
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
        return self.recos[self.best_stopping_index()]

class QuasiOpt(StopRule):
    r"""Quasi-optimality principle.

    Computes x for all available parameters
    and returns as best iterate that x_{k+1} where the ||x_{k+1} - x_{k}|| 
    is minimal

    Parameters
    ----------
    setting : Setting
        Tikhonov setting supplying the domain norm.
    solver : RegSolver
        Solver used for computing the reconstructions.
    max_iter : int, optional
        Maximum number of regularization parameters. Defaults to ``1000``.
    schedule : iterable of int or None, optional
        Iterations at which the rule records and evaluates a parameter. If
        ``None`` (default), every iteration is used.
    """
    def __init__(
        self,
        setting: Setting,
        solver: RegSolver,
        max_iter: int = 1000,
        schedule: Schedule = None,
    ) -> None:
        super().__init__(schedule=schedule)
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
        return self.it >= self.max_iter
    
    def best_stopping_index(self) -> int:
        norm_diff = self.history_dict["norm_diff"]
        return int(np.argmin(norm_diff[1:])) + 1

    def best_iterate(self) -> Vector:
        return self.recos[self.best_stopping_index()]
    
class Lepskii(StopRule):
    r"""Lepskiĭ's balancing principle.

    The rule computes reconstructions for all considered parameters. Before
    applying the balancing criterion, it sorts the regularization parameters
    increasingly. The supplied error-propagation estimate must be decreasing
    along that order.

    Parameters
    ----------
    setting : Setting
        Tikhonov setting supplying the domain norm.
    solver : RegSolver
        Solver used for computing the reconstructions.
    error_prop : callable
        Error-propagation estimate as a function of the regularization
        parameter. Although the argument defaults to ``None`` for backward
        compatibility, a callable must currently be supplied because
        automatic estimation is not implemented.
    max_iter : int, optional
        Maximum number of regularization parameters. Defaults to ``100``.
    schedule : iterable of int or None, optional
        Iterations at which the rule records and evaluates a parameter. If
        ``None`` (default), every iteration is used.
    """

    def __init__(
        self,
        setting: Setting,
        solver: RegSolver,
        error_prop: ErrorPropagation | None = None,
        max_iter: int = 100,
        schedule: Schedule = None,
    ) -> None:
        super().__init__(schedule=schedule)
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
        return self.it >= self.max_iter

    def best_stopping_index(self) -> int:
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
        return self.recos[self.best_stopping_index()]


class Oracle(StopRule):
    r"""Oracle stoprule. Returns the iterate that is closest to the exact solution in terms of the given distance function.
    Useful for testing purposes and monitoring when the exact solution is known.


    Parameters
    ----------
    setting : Setting
        Setting whose operator domain contains the exact solution.
    exact_solution : ReadOnlyVector | None, optional
        Exact solution. If omitted, ``setting.exact_solution`` is used.
    distance_function : callable | None, optional
        The distance function to measure the distance between the current iterate and the exact solution. 
        distance_function(x, exact_solution) -> float
        Defaults to None, in which case the norm of the operator's domain is applied to x-exact_solution.
    schedule : iterable of int or None, optional
        Iterations at which errors and reconstructions are recorded. If
        ``None`` (default), every iteration is used.
    """
    def __init__(
        self,
        setting: Setting,
        exact_solution: ReadOnlyVector | None = None,
        distance_function: DistanceFunction | None = None,
        schedule: Schedule = None,
    ) -> None:
        super().__init__(schedule=schedule)
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

    def __repr__(self) -> str:
        return 'Oracle'

    def _stop(self) -> bool:
        if self.solver is None or self.solver.x is None:
            raise MissingValueError     
        solver = cast("RegSolver", self.solver)
        error = self.dist(solver.x)
        self.history_dict["error"].append(error)
        self.history_dict["alphas"].append(solver.setting.regpar)
        self.recos.append(_copy_vector(solver.op.domain, solver.x))
        self.log_info = 'error {:.3e}'.format(error)
        return False

    def best_stopping_index(self) -> int:
        return int(np.argmin(self.history_dict["error"]))

    def best_iterate(self) -> Vector:
        return self.recos[self.best_stopping_index()]
    


from .general import CountIterations_factory, _capped_factory


def Discrepancy_factory(setting, params: dict) -> StopRule:
    """Create Discrepancy plus an iteration cap (timing and counts enabled).

    absolute_noise_level defaults to the setting's noise model; tau defaults
    to 2.1. rule_params can override constructor options, including noiselevel.
    The default selector handles unavailable noise and codomain-free settings.
    """
    options = {key: params[key] for key in
               ('tau', 'data', 'norm', 'noise_level_is_relative', 'schedule')
               if key in params}
    options.setdefault('tau', 2.1)
    options['setting'] = setting
    options.update(params.get('rule_params', {}))
    noise = options.pop('noiselevel', params.get('absolute_noise_level',
            setting._noise_model_param.get('absolute_noise_level')))
    rule = Discrepancy(noise, **options)
    rule.log.setLevel(params.get('logging_level', 'INFO'))
    return rule + CountIterations_factory(setting, params)

LCurve_factory = _capped_factory(LCurve)
QuasiOpt_factory = _capped_factory(QuasiOpt)
Lepskii_factory = _capped_factory(Lepskii)
Oracle_factory = _capped_factory(Oracle)
__all__ += ['Discrepancy_factory', 'LCurve_factory', 'QuasiOpt_factory', 'Lepskii_factory', 'Oracle_factory']
