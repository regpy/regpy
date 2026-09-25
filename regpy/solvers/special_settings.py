
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

__all__ = ['HilbertSetting', 'BoundConstrSetting']

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

class FISTA_SSNewton(CombinedSolver):
    """FISTA followed by SemiSmoothNewton.

    FISTA is stopped after fista_steps iterations. Its stopping rule's
    selected solution is copied and passed to SemiSmoothNewton as init.

    Both stages use the same setting and coordinate convention.
    An exterior stopping rule can stop the combined solver at any step,
    including before SemiSmoothNewton is constructed.
    """

    def __init__(
        self,
        setting,
        init=None,
        fista_stps: int = 10,
        is_primal: bool = True,
        logging_level="INFO",
        fista_params: dict | None = None,
        ssnewton_params: dict | None = None,
    ):
        if isinstance(fista_stps, bool) or not isinstance(fista_stps, int):
            raise TypeError("fista_steps must be an integer.")
        if fista_stps < 0:
            raise ValueError("fista_steps must be nonnegative.")

        fista_options = dict(fista_params or {})
        newton_options = dict(ssnewton_params or {})

        # The wrapper controls initialization and coordinate conventions.
        for options in (fista_options, newton_options):
            if "setting" in options or "init" in options:
                raise ValueError(
                    "Pass setting and init through the wrapper."
                )
            options["is_primal"] = is_primal
            options.setdefault("logging_level", logging_level)

        super().__init__(
            solver1=FISTA(setting, init=init, **fista_options),
            solver2=lambda *, init: SemiSmoothNewton(
                setting, init=init, **newton_options
            ),
            stoprule=CountIterations(
                max_iterations=fista_stps,
                logging_level=logging_level,
            ),
        )

    @staticmethod
    def check_applicability(setting, op_norm=None):
        fista_out, _ = FISTA.check_applicability(
            setting, op_norm=op_norm
        )
        newton_out, _ = SemiSmoothNewton.check_applicability(setting)

        out = {
            "applicable": (fista_out["applicable"] and newton_out["applicable"]),
            "info": ''
            }

        return out, {}

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
        super().__init__(op, penalty, data_fid, 
                              regpar = regpar, 
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

class BoundConstrSetting(Setting):
    r"""Settings with smooth convex penalty and data fidelity functionals, separable penalty, 
    and pointwise bound constraints on the unknown.

    The operator must be linear on a real domain. The data fidelity must be
    convex and provide a subgradient.

    Parameters:
    ------------------
    penalty: Quadratic | None [default: None]
        In the default case a Quadratic is used with defaults a=1, b=0, c=0 and given constraints.
    constr_l, constr_u: Vector | None [default: None]
        Constraints of the penalty if penalty is None. Otherwise, constr_l and constr_u are ignored.
    """

    def __init__(self, 
                 op: Operator, 
                 penalty: Functional,
                 data_fid: Functional | HilbertSpace | Callable | None,
                 constr_l: Vector | None = None, 
                 constr_u: Vector | None = None,
                 regpar: float | None = None, 
                 incorporate_init_in_penalty: bool = True,
                 data: Vector|None = None,
                 exact_data:Vector|None = None,
                 codomain_free: bool = False,
                 logging_level = "INFO",
                 explicitDataWarning:bool = True,                 
                 _primal_setting:Setting | None =None
                 ) -> None:
        from regpy.functionals import IntegralFunctionalBase

        if not np.issubdtype(op.domain.dtype, np.floating):
            raise ValueError("BoundConstrSetting requires a real domain.")
       
        super().__init__(
                        op, 
                        penalty, 
                        data_fid,
                        regpar = regpar, 
                        incorporate_init_in_penalty=incorporate_init_in_penalty,
                        exact_data=exact_data,
                        data = data,
                        codomain_free=codomain_free, 
                        logging_level=logging_level, 
                        explicitDataWarning=explicitDataWarning, 
                        _primal_setting=_primal_setting,
                        default_linear_regmethod = 'RunOnAlphaGrid',
                        default_nonlinear_regmethod = 'IRN',
                        default_inner_solver = 'PDHG'            
        )
        if not isinstance(self.penalty, IntegralFunctionalBase):
            raise TypeError(Errors.type_error("The penalty must be a subtype of IntegralFunctionalBase.",self))
        if constr_l is not None:
            self.penalty.constr_l = constr_l
        if constr_u is not None:
            self.penalty.constr_u = constr_u

        if any(getattr(self.penalty, name, None) is not None
               for name in ("lin_taylor_l", "lin_taylor_u",
                            "quad_taylor_l", "quad_taylor_u")):
            raise ValueError("The quadratic penalty must not have Taylor extensions.")

        if not self.data_fid.is_convex or "subgradient" not in self.data_fid.methods:
            raise ValueError("The data fidelity must be convex and provide its gradient.")
        self.add_method(method_abbrv="FISTA_SSN",
                             class_name= FISTA_SSNewton,
                             primal=True,
                             full_method_name="FISTA followed by SemiSmoothNewton"
                             )
        self.set_method_params('FISTA_SSN',{'fista_stps':50})
        self.set_stoprule_params('FISTA_SSN',
                                      {'factory':'DualityGapStopping',
                                       'schedule':None,
                                       'max_it':100
                                       }
                                       )


class Huber_XSetting(Setting):
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
                              default_inner_solver = 'dFISTA'
                              )
  
    def duality_gap(self, primal = None, dual = None): #TODO: add specific implementation!
        return super().duality_gap(primal,dual)