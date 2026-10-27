
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
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
    StopRule, DualityGapStopping, ErrorMonitor,
)
from regpy.hilbert import L2

from .setting import Setting

__all__ = ['HilbertSetting', 'BoundConstrSetting', 'Huber_XSetting',
           'FISTA_SSNewton', 'dFISTA_SSNewton', 'TV_Setting']

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

from regpy.solvers.sequences import Naturals
from regpy.solvers.solver import CombinedSolver
from regpy.solvers.linear import FISTA, SemiSmoothNewton

def _fista_gap_switch_rule(reltol: float, logging_level: int | str) -> DualityGapStopping:
    """Stop FISTA only when the relative primal duality-gap bound holds.

    ``tol=0`` deliberately disables positive absolute-gap thresholds for
    switching.  The rule is evaluated on every 5th FISTA iterate, including
    iteration zero.  A nonfinite gap must not be mistaken for convergence.
    """
    if (isinstance(reltol, (bool, np.bool_))
            or not isinstance(reltol, (int, float, np.floating))
            or not np.isfinite(reltol) or not 0. < reltol < 1.):
        raise ValueError("fista_reltol must be a finite real number in (0, 1).")
    return DualityGapStopping(
        tol=0., reltol=float(reltol), schedule=10*Naturals(),
        stop_at_infty=False, logging_level=logging_level,
    )


class _FISTA_SSNewtonBase(CombinedSolver):
    """Shared transition between primal/dual FISTA and primal/dual SSN.

    ``Setting.run`` registers the combined method as primal because its
    constructor receives the original primal Setting.  Its *final* coordinate
    label comes from the active child and is used by Setting.run to convert a
    dual SSN reconstruction back to primal coordinates.
    """

    def _initialize_stages(
        self, *, original_setting: Setting, fista_solver: FISTA,
        ssn_setting: Setting, ssn_primal: bool,
        ssn_options: dict, fista_reltol: float,
        logging_level: int | str,
    ) -> None:
        if not isinstance(ssn_primal, bool):
            raise TypeError("ssn_primal must be a bool.")
        self._original_setting = original_setting
        self._ssn_primal = ssn_primal
        self._ssn_setting = ssn_setting
        self._fista_is_primal = fista_solver.is_primal
        super().__init__(
            solver1=fista_solver,
            solver2=lambda *, init: SemiSmoothNewton(
                ssn_setting, init=init, is_primal=ssn_primal,
                **ssn_options,
            ),
            stoprule=_fista_gap_switch_rule(fista_reltol, logging_level),
        )
        # Setting.run() attaches the *outer* stopping rule to this wrapper.
        # CombinedSolver normally delegates .setting to its active child;
        # during dual FISTA/SSN that would be the dual setting.  The outer
        # rule, however, belongs to the original primal setting and must use
        # its duality gap and penalty when computing the relative bound.
        self.setting = original_setting

    def primal(self) -> tuple[ReadOnlyVector, ReadOnlyVector | None]:
        """Return the current reconstruction in the original primal space.

        During a dual stage, the child solver's ``primal()`` is its dual
        iterate (p, T*p). Map that pair through the original setting's
        R*-subgradient before exposing it to the *outer* gap stopping rule.
        """
        child = self.active_solver
        if child.is_primal:
            return child.primal()
        dual_pair = child.primal()
        return self._original_setting.dual_to_primal(dual_pair), None

    def dual(self) -> tuple[ReadOnlyVector, ReadOnlyVector | None]:
        """Return (p, T*p) in the original primal setting's dual coordinates.

        The primal/dual labels of a child refer to its own setting.  In the
        dual stage, ``child.primal()`` already is the desired original dual
        pair; returning ``child.dual()`` would return the wrong coordinates.
        """
        child = self.active_solver
        return child.dual() if child.is_primal else child.primal()

    def next(self) -> bool:
        """Advance the current stage, converting coordinates only on switching."""
        if self.is_converged():
            return False

        if self.solver2 is None:
            stopped = self.stoprule.stop()
            if stopped or self.solver1.is_converged():
                best = self.stoprule.best_iterate()
                if best is None:
                    current, image = self.solver1.x, self.solver1.y
                elif isinstance(best, tuple):
                    current, image = best
                else:
                    current, image = best, None
                if current is None:
                    raise ValueError("FISTA has no iterate available for SSNewton.")

                if self._fista_is_primal and not self._ssn_primal:
                    # x -> p=-S'(Tx)/alpha, in the original dual coordinates.
                    if image is None:
                        image = self._original_setting.op(current)
                    new_init = self._original_setting.primal_to_dual(
                        (current, image))
                elif not self._fista_is_primal and self._ssn_primal:
                    # p -> x in the original primal coordinates.
                    if image is None:
                        image = self._original_setting.op.adjoint(current)
                    new_init = self._original_setting.dual_to_primal(
                        (current, image))
                else:
                    new_init = current

                child = self._solver2_factory(init=new_init.copy())
                if child.setting is not self._ssn_setting:
                    raise RuntimeError("SSNewton was constructed with the wrong setting.")
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


class FISTA_SSNewton(_FISTA_SSNewtonBase):
    r"""Primal FISTA followed by primal or dual SemiSmoothNewton.

    FISTA switches to SSN once its *relative* duality-gap criterion is
    satisfied; no internal FISTA iteration cap is imposed. The enclosing
    ``Setting.run`` stopping rule may still impose a global iteration cap.

    Parameters
    ----------
    setting: Setting
        Original primal convex Tikhonov setting.
    init: ReadOnlyVector | None
        Initial primal FISTA iterate.
    fista_reltol: float [default: 0.5]
        Relative duality-gap tolerance for the FISTA-to-SSN transition.
        The positive absolute gap tolerance is disabled for this transition.
    ssn_primal: bool [default: True]
        If True, SSN solves the primal problem. If False, SSN solves the
        dual problem; its warm start is ``primal_to_dual`` of FISTA's iterate.
    is_primal: bool [default: True]
        Must be True: register the combined method on the primal Setting.
    fista_params: dict | None
        Options forwarded to primal FISTA.
    ssnewton_params: dict | None
        Options forwarded to SSN.
    """

    def __init__(
        self,
        setting: Setting,
        init: ReadOnlyVector | None = None,
        fista_reltol: float = 0.5,
        ssn_primal: bool = True,
        is_primal: bool = True,
        logging_level: int | str = "INFO",
        fista_params: dict | None = None,
        ssnewton_params: dict | None = None,
    ) -> None:
        if is_primal is not True:
            raise ValueError("Register FISTA_SSNewton with primal=True.")
        # Validate before constructing expensive nested Settings or solvers.
        _fista_gap_switch_rule(fista_reltol, logging_level)
        if not isinstance(ssn_primal, bool):
            raise TypeError("ssn_primal must be a bool.")
        fista_options = dict(fista_params or {})
        ssn_options = dict(ssnewton_params or {})
        for options in (fista_options, ssn_options):
            if any(key in options for key in ("setting", "init", "is_primal")):
                raise ValueError("Pass setting, init and is_primal through the wrapper.")
            options.setdefault("logging_level", logging_level)

        ssn_setting = setting if ssn_primal else setting.get_dual_setting()
        fista_solver = FISTA(setting, init=init, is_primal=True, **fista_options)
        self._initialize_stages(
            original_setting=setting, fista_solver=fista_solver,
            ssn_setting=ssn_setting, ssn_primal=ssn_primal,
            ssn_options=ssn_options, fista_reltol=fista_reltol,
            logging_level=logging_level,
        )

    @staticmethod
    def check_applicability(
        setting: Setting, op_norm: float | None = None,
        ssn_primal: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        fista_out, _ = FISTA.check_applicability(setting, op_norm=op_norm)
        try:
            ssn_setting = setting if ssn_primal else setting.get_dual_setting()
            ssn_out, _ = SemiSmoothNewton.check_applicability(ssn_setting)
        except (RuntimeError, ValueError) as error:
            ssn_out = {'applicable': False, 'info': str(error)}
        info = ' '.join(filter(None, (
            fista_out.get('info') if not fista_out['applicable'] else '',
            ssn_out.get('info') if not ssn_out['applicable'] else '',
        )))
        return {'applicable': bool(fista_out['applicable'] and ssn_out['applicable']),
                'info': info}, {}


class dFISTA_SSNewton(_FISTA_SSNewtonBase):
    r"""Dual FISTA followed by primal or dual SemiSmoothNewton.

    The FISTA iterate is a dual variable ``p``. With ``ssn_primal=False``
    the same dual variable initializes dual SSN. With ``ssn_primal=True``
    ``Setting.dual_to_primal`` produces the SSN warm start.

    FISTA switches at relative duality gap ``fista_reltol`` (default 0.5),
    rather than after a prescribed number of iterations. The current stage's
    coordinates are exposed by ``is_primal``. Register this wrapper with
    ``primal=True`` on the *original* Setting.

    Parameters
    ----------
    setting: Setting
        Original primal convex Tikhonov setting.
    init: ReadOnlyVector | None
        Initial dual FISTA iterate.
    fista_reltol: float [default: 0.5]
        Relative primal duality-gap threshold triggering the switch.
    ssn_primal: bool [default: True]
        If False, use SSN on the dual setting (no coordinate conversion).
    is_primal: bool [default: True]
        Registration flag, which must be True.
    fista_pars: dict | None
        Options forwarded to dual FISTA.
    ssn_pars: dict | None
        Options forwarded to SSN.
    """

    def __init__(
        self,
        setting: Setting,
        init: ReadOnlyVector | None = None,
        fista_reltol: float = 0.5,
        ssn_primal: bool = True,
        logging_level: int | str = "INFO",
        is_primal: bool = True,
        fista_pars: dict | None = None,
        ssn_pars: dict | None = None,
    ) -> None:
        if is_primal is not True:
            raise ValueError("Register dFISTA_SSNewton with primal=True.")
        _fista_gap_switch_rule(fista_reltol, logging_level)
        if not isinstance(ssn_primal, bool):
            raise TypeError("ssn_primal must be a bool.")
        fista_options = dict(fista_pars or {})
        ssn_options = dict(ssn_pars or {})
        for options in (fista_options, ssn_options):
            if any(key in options for key in ("setting", "init", "is_primal")):
                raise ValueError("Pass setting, init and is_primal through the wrapper.")
            options.setdefault("logging_level", logging_level)

        dual_setting = setting.get_dual_setting()
        ssn_setting = setting if ssn_primal else dual_setting
        fista_solver = FISTA(
            dual_setting, init=init, is_primal=False, **fista_options,
        )
        self._initialize_stages(
            original_setting=setting, fista_solver=fista_solver,
            ssn_setting=ssn_setting, ssn_primal=ssn_primal,
            ssn_options=ssn_options, fista_reltol=fista_reltol,
            logging_level=logging_level,
        )

    @staticmethod
    def check_applicability(
        setting: Setting, ssn_primal: bool = True, **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            dual_setting = setting.get_dual_setting()
            fista_out, _ = FISTA.check_applicability(dual_setting)
            ssn_out, _ = SemiSmoothNewton.check_applicability(
                setting if ssn_primal else dual_setting)
        except (RuntimeError, ValueError) as error:
            return {'applicable': False, 'info': str(error)}, {}
        info = ' '.join(filter(None, (
            fista_out.get('info') if not fista_out['applicable'] else '',
            ssn_out.get('info') if not ssn_out['applicable'] else '',
        )))
        return {'applicable': bool(fista_out['applicable'] and ssn_out['applicable']),
                'info': info}, {}

class HilbertSetting(Setting):
    r"""Class of settings in which both the penalty and the data fidelity functional are squared Hilbert norms.
        
    Parameters
    ----------
    op: Operator
        Operator used by this construction.
    penalty: Functional | HilbertSpace | Callable | None [default: None]
        Penalty.
    data_fid: Functional | HilbertSpace | Callable | None [default: None]
        Data fid.
    regpar: float | None [default: None]
        Regularization parameter.
    data: ReadOnlyVector | ObsDataT | None [default: None]
        Observed data.
    exact_data: ReadOnlyVector | None [default: None]
        Input vector; its coefficients are only read.
    init: ReadOnlyVector | None [default: None]
        Initial iterate in the operator domain.
    incorporate_init_in_penalty: bool [default: True]
        Incorporate init in penalty.
    codomain_free: bool [default: False]
        Codomain free.
    logging_level: int | str [default: 'INFO']
        Logging level for this object.
    explicitDataWarning: bool [default: True]
        Explicitdatawarning.
    _primal_setting: Setting | None [default: None]
         primal setting.
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
            logging_level: int | str = "INFO",
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
    
    Parameters
    ----------
    op: Operator
        Operator used by this construction.
    penalty: Functional
        In the default case a Quadratic is used with defaults a=1, b=0, c=0 and given constraints.
    data_fid: Functional | HilbertSpace | Callable | None
        Data fid.
    constr_l: ReadOnlyVector | None [default: None]
        Constraints of the penalty if penalty is None. Otherwise, constr_l and constr_u are ignored.
    constr_u: ReadOnlyVector | None [default: None]
        Constraints of the penalty if penalty is None. Otherwise, constr_l and constr_u are ignored.
    regpar: float | None [default: None]
        Regularization parameter.
    incorporate_init_in_penalty: bool [default: True]
        Incorporate init in penalty.
    data: ReadOnlyVector | None [default: None]
        Observed data.
    exact_data: ReadOnlyVector | None [default: None]
        Input vector; its coefficients are only read.
    codomain_free: bool [default: False]
        Codomain free.
    logging_level: int | str [default: 'INFO']
        Logging level for this object.
    explicitDataWarning: bool [default: True]
        Explicitdatawarning.
    _primal_setting: Setting | None [default: None]
         primal setting.
    """

    def __init__(self, 
                 op: Operator, 
                 penalty: Functional,
                 data_fid: Functional | HilbertSpace | Callable | None,
                 constr_l: ReadOnlyVector | None = None, 
                 constr_u: ReadOnlyVector | None = None,
                 regpar: float | None = None, 
                 incorporate_init_in_penalty: bool = True,
                 data: ReadOnlyVector | None = None,
                 init: ReadOnlyVector | None = None,
                 exact_data:ReadOnlyVector | None = None,
                 codomain_free: bool = False,
                 logging_level: int | str = "INFO",
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
                        init=init,
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
        self.set_method_params('FISTA_SSN', {'fista_reltol': 0.5, 'ssn_primal': True})
        self.set_stoprule_params('FISTA_SSN',
                                      {'factory':'DualityGapStopping',
                                       'max_it':100
                                       }
                                       )
        # This combined solver takes the original primal Setting and creates
        # the dual FISTA / SSNewton stages itself.  Registering with
        # primal=False makes Setting.run/evaluate_methods pass an already-dual
        # Setting, so the wrapper checks the wrong (dual-of-dual) problem.
        self.add_method(method_abbrv="dFISTA_SSN",
                             class_name= dFISTA_SSNewton,
                             primal=True,
                             full_method_name="dual FISTA followed by SemiSmoothNewton"
                             )  
        self.set_method_params('dFISTA_SSN', {'fista_reltol': 0.5, 'ssn_primal': False})
        self.set_stoprule_params('dFISTA_SSN',
                                      {'factory':'DualityGapStopping',
                                       'max_it':100
                                       }
                                       )

class Huber_XSetting(Setting):
    r"""Huber_XSetting interface.
    
    Parameters
    ----------
    op: Operator
        Operator used by this construction.
    penalty: Functional | HilbertSpace | Callable | None [default: None]
        Penalty.
    data_fid: Functional | HilbertSpace | Callable | None [default: None]
        Data fidelity; overrides the huber_sigma-dependent default when supplied.
    huber_sigma: float [default: 0.0]
        Use Huber data fidelity with this sigma if positive; otherwise use L1.
    regpar: float | None [default: None]
        Regularization parameter.
    data: ReadOnlyVector | ObsDataT | None [default: None]
        Observed data.
    exact_data: ReadOnlyVector | None [default: None]
        Input vector; its coefficients are only read.
    init: ReadOnlyVector | None [default: None]
        Initial iterate in the operator domain.
    incorporate_init_in_penalty: bool [default: True]
        Incorporate init in penalty.
    codomain_free: bool [default: False]
        Codomain free.
    logging_level: int | str [default: 'INFO']
        Logging level for this object.
    explicitDataWarning: bool [default: True]
        Explicitdatawarning.
    _primal_setting: Setting | None [default: None]
         primal setting.
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
            logging_level: int | str = "INFO",
            explicitDataWarning:bool = True,                 
            _primal_setting:Setting | None =None, # intended only for internal use in get_dual_setting()
            huber_sigma: float = 0.,
             )->None:
              
        from regpy.functionals import Huber, L1
        if (isinstance(huber_sigma, (bool, np.bool_))
                or not isinstance(huber_sigma, (int, float, np.floating))
                or not np.isfinite(huber_sigma)):
            raise ValueError("huber_sigma must be a finite real number.")
        self.huber_sigma = float(huber_sigma)
        penalty = L2(op.domain) if penalty is None else penalty
        if data_fid is None:
            data_fid = (Huber(op.codomain, sigma=self.huber_sigma, L1_type=True)
                        if self.huber_sigma > 0 else L1(op.codomain))
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
        self.add_method(
            method_abbrv='dFISTAdSSN',
            class_name=dFISTA_SSNewton,
            primal=True,
            full_method_name='Dual FISTA followed by dual SemiSmoothNewton',
        )
        self.set_method_params('dFISTAdSSN', {
            'fista_reltol': 0.5,
            'ssn_primal': False,
        })
        # Always monitor convergence of the *combined* method with the
        # original primal duality gap.  Previously, the default factory
        # could fall back to RelativeChangeSol because the CombinedSolver
        # class exposes primal()/dual() only by runtime delegation.
        # A recorded gap history also makes the method visible in
        # regpy.solvers.plotting.convergence_plots().
        self.set_stoprule_params('dFISTAdSSN', {
            'factory': 'DualityGapStopping',
            'schedule': None,
        })

  

class TV_Setting(Setting):
    r"""Rockafellar--Fenchel setting for (Huber-smoothed) isotropic TV.

    The minimized functional is

    .. math::
        \frac{1}{\alpha}S_{\rm data}(Tf)
        + H_\beta(\nabla f) + R(f),

    with ``H_0 = L1L2`` and ``H_beta = HuberL2(beta)`` for ``beta>0``.
    For a positive data-operator scale ``c`` it is represented equivalently as

    .. math::
        \frac{1}{\alpha}
        \bigl((S_{\rm data}\circ(\cdot/c))\oplus
        (\alpha H_\beta)\bigr)
        \bigl((cT,\nabla)f\bigr)+R(f).

    This changes the PDHG operator geometry but not the objective. Choose
    ``data_operator_scale='auto'`` to balance the norms of the forward and
    gradient blocks, or ``1.0`` to recover the unscaled formulation.

    Parameters
    ----------
    op : Operator
        **Linear** data-forward operator ``T``. Its domain must be a
        ``UniformGridFcts`` if ``grad`` is not supplied.
    data_fid : Functional | HilbertSpace | callable | None
        Data-fidelity functional on ``T.codomain``. Defaults to squared L2.
        An already data-dependent functional is supported. In particular,
        non-translation data fidelities (such as KL) receive their data
        through their *own* ``as_data_func`` method, not through a global
        shift of the direct sum.
    penalty : Functional | HilbertSpace | callable | None
        Additional penalty ``R`` on ``T.domain``; defaults to NullFunctional.
    regpar : float
        Positive regularization parameter ``alpha``. Changing ``regpar``
        subsequently automatically rescales the TV component.
    beta : float
        Nonnegative Huber transition parameter; zero gives exact TV.
    data : ReadOnlyVector | None
        Observed data in ``T.codomain``. Internally the observations are
        ``(c*data, zero)`` in the augmented codomain, where ``c`` is
        ``data_operator_scale``; ``forward_op`` and ``data_term`` retain the
        original (unscaled) physical data coordinates. Only the first
        *functional* component receives the raw observations via ``as_data_func``.
        The gradient component must never be shifted.
    grad : Operator | None
        Optional gradient operator with domain ``T.domain``. By default,
        ``ForwardFDGradient`` is used.
    data_operator_scale : float | str
        Positive scale ``c`` for the data-forward block. The augmented
        operator is ``(c*T, grad)`` and its first fidelity component is
        ``S_data(y/c)``. Defaults to ``1.0`` (backward-compatible). Set to
        ``'auto'`` to choose ``max(1, ||grad|| / ||T||)``; this estimates two
        operator norms and may be relatively expensive. The scale is fixed at
        construction and does not change when ``regpar`` or ``data`` changes.
    boundary_condition : str
        Boundary condition used to construct ``ForwardFDGradient``.
    init : ReadOnlyVector | None
        Initial reconstruction.
    incorporate_init_in_penalty : bool
        As in ``Setting``.
    logging_level : int | str
        As in ``Setting``.
    explicitDataWarning : bool
        As in ``Setting``.

    Notes
    -----
    * The stored ``Setting.op`` is ``VectorOfOperators((c*T, grad))``;
      ``forward_op`` still refers to the original ``T``.  Evaluation caps
      and histories count operations on ``forward_op``, not on ``grad`` or
      ``VectorOfOperators``.  Iterations are counted on the PDHG solver.
    * ``Setting.data`` stores ``(c*data, 0)``. Assigning raw data in
      ``T.codomain`` or scaled augmented data in the direct-sum codomain
      updates *only* the first fidelity component. This preserves KL semantics.
    * ``codomain_free=True`` is not supported for this explicit formulation.
    * The default solver is PDHG. If ``R=0``, the conjugate of the penalty
      is an indicator of zero, so the full duality gap is generally infinite
      at intermediate iterates. PDHG therefore defaults to
      ``DualityGapStopping(gap_part='data_fid')`` for primal PDHG and
      ``gap_part='penalty'`` for dPDHG. The latter evaluates the Bregman gap
      of the *dual setting's penalty*, which is the conjugated original data
      fidelity. It does not require recovering an original primal iterate
      through the (non-single-valued) subgradient of R*. Both partial-gap
      thresholds are heuristics and record ``converged=False`` on acceptance.
      The full gap remains available by overriding the stopping rule.
    """

    def __init__(
        self,
        op: Operator,
        data_fid: Functional | HilbertSpace | Callable | None = None,
        penalty: Functional | HilbertSpace | Callable | None = None,
        regpar: float = 1.,
        beta: float = 0.,
        data: ReadOnlyVector | None = None,
        grad: Operator | None = None,
        boundary_condition: str = "Neum",
        data_operator_scale: float | str = 'auto',
        init: ReadOnlyVector | None = None,
        incorporate_init_in_penalty: bool = True,
        logging_level: int | str = "INFO",
        explicitDataWarning: bool = True,
    ) -> None:
        from regpy.operators import ForwardFDGradient, VectorOfOperators
        from regpy.operators.base import LinearCombination
        from regpy.functionals.base import (
            FunctionalOnDirectSum, NullFunctional, as_functional,
        )
        from regpy.functionals.vector_integral_functionals import HuberL2, L1L2
        from regpy.vecsps import UniformGridFcts

        if not isinstance(op, Operator) or not op.linear:
            raise ValueError("TV_Setting requires a linear data-forward operator.")
        if isinstance(beta, (bool, np.bool_)) or not isinstance(
            beta, (int, float, np.floating)
        ) or not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be a finite nonnegative real number.")
        if (isinstance(regpar, (bool, np.bool_))
                or not isinstance(regpar, (int, float, np.floating))
                or not np.isfinite(regpar) or regpar <= 0):
            raise ValueError("regpar (alpha) must be finite and strictly positive.")
        if grad is None:
            if not isinstance(op.domain, UniformGridFcts):
                raise TypeError(
                    "The default ForwardFDGradient needs a UniformGridFcts "
                    "domain; provide grad explicitly for another domain."
                )
            grad = ForwardFDGradient(
                op.domain, boundary_condition=boundary_condition
            )
        if not isinstance(grad, Operator) or not grad.linear or grad.domain != op.domain:
            raise ValueError("grad must be a linear operator with domain T.domain.")

        # The physical model is the evaluation-accounting boundary. Instrument
        # it BEFORE forming any scaled or composite operator: operator algebra
        # may otherwise flatten an uninstrumented composition, bypassing the
        # physical model's counter. Never reset counts owned by outer solvers.
        op.enable_eval_count(reset=False)

        # Only the operator *coordinates* are changed, not the physical
        # fidelity S_data: F_c(y) = S_data(y/c), F_c(c*T(f)) = S_data(T(f)).
        # Automatic balancing uses norms of the original (unscaled) blocks.
        if isinstance(data_operator_scale, str):
            if data_operator_scale != "auto":
                raise ValueError(
                    "data_operator_scale must be a positive finite number "
                    "or 'auto'."
                )
            forward_norm = float(op.norm())
            gradient_norm = float(grad.norm())
            if (not np.isfinite(forward_norm) or forward_norm < 0
                    or not np.isfinite(gradient_norm) or gradient_norm < 0):
                raise ValueError(
                    "Cannot automatically balance data and gradient blocks: "
                    "their operator norms must be finite and nonnegative. "
                    "Pass an explicit data_operator_scale instead."
                )
            # An identically zero data operator cannot be balanced by scaling.
            scale = (max(1., gradient_norm / forward_norm)
                     if forward_norm > 0 else 1.)
        elif (isinstance(data_operator_scale, (bool, np.bool_))
              or not isinstance(data_operator_scale,
                                (int, float, np.integer, np.floating))
              or not np.isfinite(data_operator_scale)
              or data_operator_scale <= 0):
            raise ValueError(
                "data_operator_scale must be a positive finite number "
                "or 'auto'."
            )
        else:
            scale = float(data_operator_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                "Automatic data_operator_scale overflowed; "
                "pass an explicit positive finite scale."
            )

        alpha = float(regpar)
        self.data_operator_scale = scale
        self.forward_op = op
        # Do not use ``scale * op`` here.  Some forward operators (notably
        # ConvolutionOperator) override scalar multiplication to construct a
        # *new* operator with a scaled kernel.  Although mathematically equal
        # to scale*T, that new object bypasses the original physical T and
        # its evaluation counters.  The explicit one-term LinearCombination
        # retains T as a child and dispatches its forward/adjoint operations
        # through T, also when scale != 1.
        self.scaled_forward_op = (
            op if scale == 1. else LinearCombination((scale, op))
        )
        self.grad = grad
        self.beta = float(beta)
        self.tv_func = (L1L2(grad.codomain, conj_tol=1e-5)
                        if beta == 0 else HuberL2(grad.codomain, sigma=self.beta))
        self._tv_base_data_term = as_functional(
            L2(op.codomain) if data_fid is None else data_fid, op.codomain
        )
        if not self._tv_base_data_term.is_convex:
            raise ValueError("A Rockafellar--Fenchel TV_Setting requires convex data fidelity.")
        if data is not None and data not in op.codomain:
            raise ValueError("data must belong to T.codomain.")
        self.data_term = self._tv_base_data_term
        if data is not None:
            # Important: do this *before* forming FunctionalOnDirectSum.
            # A global direct-sum translation is wrong for KL and other
            # fidelities whose observations are not an argument shift.
            self.data_term = deepcopy(self._tv_base_data_term).as_data_func(data)
        self._tv_FuncOnSum = FunctionalOnDirectSum
        self._tv_registered = False

        augmented_op = VectorOfOperators((self.scaled_forward_op, grad))
        self._tv_augmented_op = augmented_op
        additional_penalty = (NullFunctional(op.domain)
                              if penalty is None else penalty)
        additional_penalty = as_functional(additional_penalty, op.domain)
        self._tv_null_penalty = isinstance(additional_penalty, NullFunctional)
        if not additional_penalty.is_convex:
            raise ValueError("A Rockafellar--Fenchel TV_Setting requires convex penalty.")

        # If the supplied fidelity already embeds observations, retain them
        # in the Setting.data metadata as well (when they are domain vectors).
        raw_data = data
        if raw_data is None and self.data_term.is_data_func:
            try:
                candidate = self.data_term.data
                if candidate in op.codomain:
                    raw_data = candidate
            except (AttributeError, TypeError):
                pass
        augmented_data = self._augmented_observations(
            raw_data, augmented_op, scale)

        # Do NOT pass augmented_data to Setting: Setting.change_data would
        # translate the *entire* FunctionalOnDirectSum and bypass the
        # data-dependent semantics of the first component.
        super().__init__(
            op=augmented_op,
            penalty=additional_penalty,
            data_fid=self._tv_data_fidelity(alpha),
            regpar=alpha,
            data=None,
            init=init,
            incorporate_init_in_penalty=incorporate_init_in_penalty,
            logging_level=logging_level,
            explicitDataWarning=(explicitDataWarning and augmented_data is None),
            default_inner_solver="PDHG",
        )
        if augmented_data is not None:
            self._data = augmented_data
        self._tv_registered = True
        if self._tv_null_penalty:
            self._protect_pdhg_from_infinite_gaps()

    @property
    def evaluation_operator(self) -> Operator:
        """Count physical forward-model work, not gradient/stack operations.

        The augmented operator ``(c*T, grad)`` calls the original ``T`` as
        its first block, including when ``c != 1``.  Counting the original
        operator avoids charging finite-difference gradient evaluations or
        book-keeping in VectorOfOperators to the inverse problem's budget.
        """
        return self.forward_op

    def _protect_pdhg_from_infinite_gaps(self) -> None:
        # R=0 entails R* = indicator_{0}. The full gap is typically
        # infinite because T* p != 0. For primal PDHG monitor the fidelity
        # contribution on this setting. For dual PDHG use the *penalty*
        # contribution on the transformed dual setting. This avoids the
        # original NullFunctional.conj subgradient and does not require
        # reconstructing the original primal iterate. Neither partial gap
        # certifies convergence; the iteration cap remains in place.
        params = {
            "factory": "DualityGapStopping",
            "stop_at_infty": False,
            "reltol": 1e-3,
            "tol": 1e-12,
            "rpow": 1    # actually the two components of the data term may have different rpow's, but 1 dominates.
        }
        self.set_stoprule_params("PDHG", {**params, "gap_part": "data_fid"})
        self.set_stoprule_params("dPDHG", {**params, "gap_part": "penalty"})

    @staticmethod
    def _augmented_observations(data, op, scale: float = 1.):
        if data is None:
            return None
        if data not in op.ops[0].codomain:
            raise ValueError("data must belong to the data-forward operator's codomain.")
        observations = op.codomain.zeros()
        op.ops[0].codomain.copyto(observations[0], scale * data)
        return observations

    def _tv_data_fidelity(self, alpha: float) -> Functional:
        scale = self.data_operator_scale
        # Apply dilation *after* incorporating observations in S_data.
        # This is necessary for KL and other non-translation fidelities.
        data_component = (self.data_term if scale == 1.
                          else self.data_term.dilation(1. / scale))
        return self._tv_FuncOnSum(
            [data_component, alpha * self.tv_func],
            domain=(self.op.codomain if hasattr(self, "_op")
                    else self._tv_augmented_op.codomain),
        )

    @Setting.regpar.setter
    def regpar(self, new_regpar: float) -> None:
        # Setting.__init__ calls this setter before _tv_registered becomes True.
        if (isinstance(new_regpar, (bool, np.bool_))
                or not isinstance(new_regpar, (int, float, np.floating))
                or not np.isfinite(new_regpar) or new_regpar <= 0):
            raise ValueError("TV_Setting.regpar must be finite and positive.")
        alpha = float(new_regpar)
        if getattr(self, "_tv_registered", False):
            # The measured-data dependence is contained *inside* the first
            # component. Rebuild the second component at the new alpha.
            self._data_fid = self._tv_data_fidelity(alpha)
        Setting.regpar.fset(self, alpha)
        if (getattr(self, "_tv_registered", False)
                and self._tv_null_penalty):
            # Setting.regpar refreshes the method registry: restore this
            # special setting's safe default on every alpha change.
            self._protect_pdhg_from_infinite_gaps()

    def change_data(self, new_data) -> None:
        """Update only S_data, accepting raw data or (c*data, zero).

        Raw observations are expressed in ``forward_op.codomain``;
        augmented observations are expressed in the scaled operator's
        codomain and therefore have first component ``c*data``.
        """
        if new_data in self.forward_op.codomain:
            raw_data = new_data
        elif new_data in self.op.codomain:
            # Nonzero second components would encode an unwanted shift of TV.
            if np.any(np.asarray(new_data[1]) != 0):
                raise ValueError("The augmented gradient-data component must be zero.")
            raw_data = new_data[0] / self.data_operator_scale
        else:
            raise ValueError("data must belong to T.codomain or augmented codomain.")
        # Updating a KL/Poisson fidelity is *not* the same as shifting it.
        self.data_term = deepcopy(self._tv_base_data_term).as_data_func(raw_data)
        self._data_fid = self._tv_data_fidelity(self.regpar)
        self._data = self._augmented_observations(
            raw_data, self.op, self.data_operator_scale)
