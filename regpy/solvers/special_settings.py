
from __future__ import annotations
import math as ma
import numpy as np
from typing import Callable
from copy import deepcopy

from regpy.util import ClassLogger, Errors
from regpy.vecsps import Vector, ReadOnlyVector
from regpy.operators import Operator, ObsDataT
from regpy.hilbert import HilbertSpace
from regpy.functionals.base import  as_functional, Composed
from regpy.functionals import Functional
from regpy.stoprules import (
    StopRule, DualityGapStopping, CountIterations, ErrorMonitor,
)
from regpy.hilbert import L2

from .setting import Setting

__all__ = ['HilbertSetting']

"""
Special settings are mainly convenience classes. 
Their main purpose of specialized setting is to define default inner and outer solvers 
that work "best" (or at least reasonably good) for certain combinations of data fidelity 
and penalty term. This default solver can be employed by the specialized setting's run 
method and allows to quickly get a solution to an inverse problem without having 
to perform tests of all applicable solvers. This holds rue in particular to nonlinear 
inverse problems where inner solvers are often needed. Specialized settings allow to 
to set up such combined solvers in a simple way.

Moreover, some specialized settings implement formulas for the duality gap that 
avoid loss of accuracy by cancellation effects.
"""

from regpy.solvers.solver import CombinedSolver
from regpy.solvers.linear import FISTA, SemiSmoothNewton
from regpy.stoprules import CountIterations

class dFISTA_SSNewton(CombinedSolver):
    """Dual FISTA followed by primal SemiSmoothNewton.

    During FISTA, x and y are dual coordinates and is_primal is False.
    At the transition, the selected dual iterate is converted once into
    a primal warm start. Thereafter is_primal is True.

    Parameters
    ----------
    setting : Setting
        Original primal Tikhonov setting.
    init : Vector or None
        Initial dual FISTA iterate.
    fista_stps : int
        Maximum number of FISTA steps. Defaults to 10.
    is_primal : bool
        Constructor flag supplied by Setting.run(). The method must be
        registered with primal=True because it receives the primal setting.
        Its current coordinate flag follows the active child instead.
    """

    def __init__(
        self,
        setting,
        init=None,
        fista_stps=10,
        logging_level="INFO",
        is_primal=True,
        fista_pars=None,
        ssn_pars=None,
    ):
        if is_primal is not True:
            raise ValueError(
                "Register dFISTA_SSNewton with primal=True."
            )
        if (
            isinstance(fista_stps, bool)
            or not isinstance(fista_stps, int)
            or fista_stps < 0
        ):
            raise ValueError("fista_stps must be a nonnegative integer.")

        self._original_setting = setting
        fista_pars = dict(fista_pars or {})
        ssn_pars = dict(ssn_pars or {})

        solver1 = FISTA(
            setting.get_dual_setting(),
            init=init,
            is_primal=False,
            logging_level=logging_level,
            **fista_pars,
        )

        super().__init__(
            solver1,
            lambda *, init: SemiSmoothNewton(
                setting,
                init=init,
                is_primal=True,
                logging_level=logging_level,
                **ssn_pars,
            ),
            stoprule=CountIterations(fista_stps),
        )

    def next(self):
        if self.is_converged():
            return False

        if self.solver2 is None:
            stopped = self.stoprule.stop()

            if stopped or self.solver1.is_converged():
                best = self.stoprule.best_iterate()

                if best is None:
                    p, Tstar_p = self.solver1.x, self.solver1.y
                elif isinstance(best, tuple):
                    p, Tstar_p = best
                else:
                    p, Tstar_p = best, None

                if p is None:
                    raise ValueError("No dual warm-start iterate available.")

                # Use the selected iterate, not necessarily the last one.
                if Tstar_p is None:
                    Tstar_p = self.solver1.op(p)

                x = self._original_setting.dual_to_primal(
                    (p, Tstar_p)
                )
                self.solver2 = self._solver2_factory(init=x.copy())
                self.active_solver = self.solver2
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

    @staticmethod
    def check_applicability(setting, **kwargs):
        dual_out, _ = FISTA.check_applicability(
            setting.get_dual_setting()
        )
        ssn_out, _ = SemiSmoothNewton.check_applicability(setting)

        out = dict(dual_out)
        out["applicable"] = (
            dual_out["applicable"] and ssn_out["applicable"]
        )
        # Child parameters are determined by their own constructors.
        return out, {}

class HilbertSetting(Setting):
    r""" Class of settings in which both the penalty and the data fidelity functional are squared Hilbert norms.
    """
    def  __init__(self,
            op: Operator, 
            penalty: Functional|HilbertSpace|Callable|None = None, 
            data_fid:Functional|HilbertSpace|Callable|None = None,
            regpar: float|None =None,
            data: ReadOnlyVector | ObsDataT | None = None, 
            exact_data: ReadOnlyVector | None = None, 
            init: ReadOnlyVector | None = None,
            incorporate_init_in_penalty: bool = True,
            codomain_free: bool = False,
            logging_level = "INFO",
            explicitDataWarning:bool = True,                 
            _primal_setting:Setting | None =None # intended only for internal use in get_dual_setting()
             )->None:
              
        penalty = L2(op.domain) if penalty is None else penalty
        data_fid = L2(op.codomain) if data_fid is None else data_fid
        super().__init__(op, penalty, data_fid, regpar, 
                              data = data, 
                              exact_data = exact_data, 
                              init=init, 
                              incorporate_init_in_penalty=incorporate_init_in_penalty,
                              codomain_free=codomain_free, 
                              logging_level=logging_level, 
                              explicitDataWarning=explicitDataWarning, 
                              _primal_setting=_primal_setting,
                              default_linear_regmethod = 'CGNE',
                              default_nonlinear_regmethod = 'NewtonCG',
                              default_inner_solver = 'TikhCG'
                              )
        if not self.penalty.is_norm_squared_type:
            raise RuntimeError(Errors.runtime_error('Penalty of HilbertSetting must be of squared norm type.',self.penalty))
        if not self.data_fid.is_norm_squared_type:
            raise RuntimeError(Errors.runtime_error('Data fidelity term of HilbertSetting must be of squared norm type.',self.data_fid))

    def duality_gap(self, primal = None, dual = None): #TODO: add specific implementation!
        return super().duality_gap(primal,dual)


class HilbertNonNegSetting(Setting):
    r""" Class of settings in which the data fidelity functional is a squared Hilbert norm and 
    the penalty functional is a squared L^2 norm with nonnegativity constraint.
    """
    def  __init__(self,
            op: Operator, 
            penalty: Functional|HilbertSpace|Callable|None = None, 
            data_fid:Functional|HilbertSpace|Callable|None = None,
            regpar: float|None =None,
            data: ReadOnlyVector | ObsDataT | None = None, 
            exact_data: ReadOnlyVector | None = None, 
            init: ReadOnlyVector | None = None,
            incorporate_init_in_penalty: bool = True,
            codomain_free: bool = False,
            logging_level = "INFO",
            explicitDataWarning:bool = True,                 
            _primal_setting:Setting | None =None # intended only for internal use in get_dual_setting()
             )->None:
        from regpy.functionals import QuadNonneg

        penalty = QuadNonneg(op.domain) if penalty is None else penalty
        data_fid = L2(op.codomain) if data_fid is None else data_fid
        super().__init__(op, penalty, data_fid, regpar, 
                              data = data, 
                              exact_data = exact_data, 
                              init=init, 
                              incorporate_init_in_penalty=incorporate_init_in_penalty,
                              codomain_free=codomain_free, 
                              logging_level=logging_level, 
                              explicitDataWarning=explicitDataWarning, 
                              _primal_setting=_primal_setting,
                              default_inner_solver = 'FISTA'
                              )
        if self.penalty.lb is None or (not self.op.domain(self.penalty.lb) == 0):
            raise RuntimeError(Errors.runtime_error('penalty must have lower bound 0',self.penalty.lb))
        if not self.data_fid.is_norm_squared_type:
            raise RuntimeError(Errors.runtime_error('Data fidelity term of HilbertSetting must be of squared norm type.',self.data_fid))

    def duality_gap(self, primal = None, dual = None): #TODO: add specific implementation!
        return super().duality_gap(primal,dual)


class HuberXSetting(Setting):
    def  __init__(self,
            op: Operator, 
            penalty: Functional|HilbertSpace|Callable|None = None, 
            data_fid:Functional|HilbertSpace|Callable|None = None,
            regpar: float|None =None,
            data: ReadOnlyVector | ObsDataT | None = None, 
            exact_data: ReadOnlyVector | None = None, 
            init: ReadOnlyVector | None = None,
            incorporate_init_in_penalty: bool = True,
            codomain_free: bool = False,
            logging_level = "INFO",
            explicitDataWarning:bool = True,                 
            _primal_setting:Setting | None =None # intended only for internal use in get_dual_setting()
             )->None:
              
        from regpy.functionals import L1
        penalty = L2(op.domain) if penalty is None else penalty
        data_fid = L1(op.codomain) if data_fid is None else data_fid
        super().__init__(op, penalty, data_fid, regpar, 
                              data = data, 
                              exact_data = exact_data, 
                              init=init, 
                              incorporate_init_in_penalty=incorporate_init_in_penalty,
                              codomain_free=codomain_free, 
                              logging_level=logging_level, 
                              explicitDataWarning=explicitDataWarning, 
                              _primal_setting=_primal_setting,
                              default_inner_solver = 'dual_FISTA'
                              )
  
    def duality_gap(self, primal = None, dual = None): #TODO: add specific implementation!
        return super().duality_gap(primal,dual)