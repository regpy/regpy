
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
from regpy.functionals.base import NullFunctional, as_functional, Composed
from regpy.functionals import Functional
from regpy.stoprules import (
    StopRule, DualityGapStopping, ErrorMonitor,
)
from regpy.hilbert import L2

from .setting import Setting

__all__ = ['BoundConstrSetting', 'Huber_XSetting',
           'FISTA_SSNewton', 'dFISTA_SSNewton', 'TV_Setting', 'TVaug_Setting',
           'PDHGaug', 'TVaugOptimalityStopping',
           'TVaugFISTAGapStopping', 'TVaugPDHGGapStopping']

from regpy.solvers.sequences import Naturals
from regpy.solvers.solver import CombinedSolver
from regpy.solvers.linear import FISTA, SemiSmoothNewton, PDHG

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
                                       'max_it':1000
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
                                       'max_it':1000
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

  

class TVaug_Setting(Setting):
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
    * With ``R=0``, the primal penalty's conjugate is the indicator of zero,
      so the usual Fenchel duality gap is infinite unless the saddle dual
      variable is exactly adjoint-feasible. The default stopping rule uses
      :class:`TVaugOptimalityStopping`: the nonnegative fidelity Fenchel--Young
      gap plus the squared dual domain norm of ``(c*T, grad).adjoint(p)``.
      Both terms are recorded separately and a hard iteration cap remains.
      The new sum is an optimality residual, not a standard duality gap.
      With nonzero additional ``R`` the usual full-gap rule is retained.
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
            raise ValueError("TVaug_Setting requires a linear data-forward operator.")
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
            raise ValueError("A Rockafellar--Fenchel TVaug_Setting requires convex data fidelity.")
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
            raise ValueError("A Rockafellar--Fenchel TVaug_Setting requires convex penalty.")

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
        """Install the two-term optimality test when the extra penalty is zero.

        R*=indicator_{0} makes the ordinary full gap infinite at generic
        saddle iterates. The new criterion instead combines the fidelity's
        Fenchel--Young gap with the squared adjoint-feasibility residual.
        The dual PDHG method is interpreted in the original augmented
        primal/dual coordinates by TVaugOptimalityStopping.
        """
        for name in ("PDHG", "dPDHG"):
            self.set_stoprule_params(name, {
                "factory": _tvaug_pdhg_gap_factory,
                "tol": 1e-5,
                "schedule": None,
            })
        # FISTA may run directly on the augmented setting when the TV block
        # has a Lipschitz subgradient (e.g. Huber smoothing).
        self.set_stoprule_params("FISTA", {
            "factory": _tvaug_fista_gap_factory,
            "tol": 1e-5,
            "schedule": None,
        })

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
            raise ValueError("TVaug_Setting.regpar must be finite and positive.")
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


class PDHGaug(PDHG):
    r"""Solve a standard :class:`TV_Setting` through its augmented TV setting.

    With ``A = (c*T, grad)``, PDHG by default uses the blockwise upper bound

    .. math::
        L_A = \sqrt{c^2 L_T^2 + L_{\nabla}^2}

    for its step-size selection. Here the component norms use the augmented
    setting's **domain Hilbert geometry** and the appropriate component
    codomain geometries. This bound is at most a factor ``sqrt(2)`` above
    the true augmented norm, provided the component norms are accurate.

    ``op_norm`` (or the explicit alias ``norm_op``) is the norm of the
    *physical* operator ``T``, **not** the norm of ``(c*T, grad)``. If supplied,
    it is used rather than estimating ``T.norm()``. ``norm_grad`` similarly
    supplies the norm of the gradient. ``norm_op`` has precedence over
    ``op_norm`` when both are passed (e.g. when ``Setting.evaluate_methods``
    supplies its generic ``op_norm`` but the method configuration specifies
    ``norm_op``). If neither is given, each block norm is estimated separately.

    Set ``exact_augmented_norm=True`` to estimate the augmented norm directly,
    in the same way as ordinary PDHG. An explicit ``augmented_op_norm`` can
    also be supplied to use an independently known augmented norm; it takes
    precedence over all other norm options. These options affect automatic
    step-size selection, not the iteration formulas. Supplying all of
    ``tau``, ``sigma``, and ``theta`` bypasses automatic norm estimation.

    The solver returns the physical pair ``(x, T(x))`` from :meth:`run`;
    its ``primal`` and ``dual`` methods remain in augmented coordinates for
    the stopping rule. Evaluation counters are never reset here.
    """

    @staticmethod
    def _augmented_setting(setting: Setting):
        """Resolve the saddle formulation, including alpha-grid inner settings."""
        augmented = getattr(setting, 'tvaug_setting', None)
        if augmented is not None:
            return augmented
        cached = getattr(setting, '_tvaug_pdhg_cache', None)
        if cached is not None and cached.regpar == setting.regpar:
            return cached
        options = getattr(setting, '_tvaug_constructor', None)
        if options is None:
            return None
        augmented = TVaug_Setting(
            op=setting.op, data_fid=setting.data_fid,
            penalty=options['additional_penalty'], regpar=setting.regpar,
            beta=options['beta'], data=setting.data,
            grad=options['grad'], data_operator_scale=options['scale'],
            init=setting.init, incorporate_init_in_penalty=False,
            explicitDataWarning=False,
        )
        setting._tvaug_pdhg_cache = augmented
        return augmented

    @staticmethod
    def _validate_norm(value, name: str) -> float | None:
        """Reject invalid estimates rather than passing them to PDHG."""
        if value is None:
            return None
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be a finite nonnegative real number.')
        return float(value)

    @staticmethod
    def _norm_for_pdhg(
        augmented: TVaug_Setting, *,
        op_norm: float | None = None,
        norm_op: float | None = None,
        norm_grad: float | None = None,
        exact_augmented_norm: bool = False,
        augmented_op_norm: float | None = None,
    ) -> float:
        """Get the PDHG norm bound in the augmented Hilbert geometries.

        The codomain geometry of the scaled data term is the geometry of its
        original data fidelity: horizontal dilation preserves ``h_domain``.
        The TV coefficient likewise preserves the gradient-block geometry.
        """
        if not isinstance(exact_augmented_norm, (bool, np.bool_)):
            raise TypeError('exact_augmented_norm must be a bool.')
        op_norm = PDHGaug._validate_norm(op_norm, 'op_norm')
        norm_op = PDHGaug._validate_norm(norm_op, 'norm_op')
        norm_grad = PDHGaug._validate_norm(norm_grad, 'norm_grad')
        augmented_op_norm = PDHGaug._validate_norm(
            augmented_op_norm, 'augmented_op_norm')
        if augmented_op_norm is not None:
            bound = augmented_op_norm
        elif exact_augmented_norm:
            bound = float(augmented.op.norm(
                augmented.h_domain, augmented.h_codomain))
        else:
            if norm_op is None:
                norm_op = op_norm
            if norm_op is None:
                norm_op = float(augmented.forward_op.norm(
                    augmented.h_domain, augmented.data_term.h_domain))
            if norm_grad is None:
                norm_grad = float(augmented.grad.norm(
                    augmented.h_domain, augmented.tv_func.h_domain))
            bound = ma.hypot(augmented.data_operator_scale * norm_op,
                             norm_grad)
        if not np.isfinite(bound) or bound <= 0.:
            raise ValueError('The augmented operator norm bound must be '
                             'finite and strictly positive for automatic PDHG steps.')
        return bound

    def __init__(
        self, setting: Setting, *,
        op_norm: float | None = None,
        norm_op: float | None = None,
        norm_grad: float | None = None,
        exact_augmented_norm: bool = False,
        augmented_op_norm: float | None = None,
        **kwargs,
    ) -> None:
        augmented = self._augmented_setting(setting)
        if augmented is None:
            raise ValueError('PDHGaug requires a convex TV Tikhonov setting.')
        # When all three PDHG step parameters are given, the base solver does
        # not evaluate an operator norm at all.
        prescribed_steps = (kwargs.get('tau', 0.) > 0
                            and kwargs.get('sigma', 0.) > 0
                            and kwargs.get('theta', -1.) >= 0)
        bound = None if prescribed_steps else self._norm_for_pdhg(
            augmented, op_norm=op_norm, norm_op=norm_op,
            norm_grad=norm_grad, exact_augmented_norm=exact_augmented_norm,
            augmented_op_norm=augmented_op_norm)
        super().__init__(augmented, op_norm=bound, **kwargs)
        self._physical_tv_setting = setting
        self.augmented_op_norm_bound = bound

    @staticmethod
    def check_applicability(
        setting: Setting,
        op_norm: float | None = None,
        norm_op: float | None = None,
        norm_grad: float | None = None,
        exact_augmented_norm: bool = False,
        augmented_op_norm: float | None = None,
        tau: float = 0., sigma: float = 0., theta: float = -1.,
        **kwargs,
    ):
        if not setting.is_tikhonov or not setting.is_convex:
            return {'applicable': False, 'info':
                    'PDHGaug requires a convex TV Tikhonov setting.'}, {}
        augmented = PDHGaug._augmented_setting(setting)
        if augmented is None:
            return {'applicable': False, 'info':
                    'PDHGaug requires a TV_Setting or TV alpha-grid inner setting.'}, {}
        prescribed_steps = tau > 0 and sigma > 0 and theta >= 0
        bound = None if prescribed_steps else PDHGaug._norm_for_pdhg(
            augmented, op_norm=op_norm, norm_op=norm_op,
            norm_grad=norm_grad, exact_augmented_norm=exact_augmented_norm,
            augmented_op_norm=augmented_op_norm)
        # PDHG.check_applicability always interprets op_norm as the norm of
        # the operator in *its* setting: here that is the augmented operator.
        return PDHG.check_applicability(
            augmented, op_norm=bound, tau=tau, sigma=sigma,
            theta=theta, **kwargs)

    def run(self, stoprule):
        x, _ = super().run(stoprule)
        return x, self._physical_tv_setting.op(x)


class TVaugOptimalityStopping(DualityGapStopping):
    r"""Two-term optimality residual for augmented TV with zero extra penalty.

    Write ``A=(cT, grad)``, ``S=setting.data_fid``, and let ``alpha`` be the
    regularization parameter. For a primal/dual pair ``(x,p)``, this evaluates

    ``gap_S = (S(Ax) + S.conj(-alpha*p) + alpha*<p,Ax>) / alpha``

    and

    ``feasibility = <A.adjoint(p), G_X^{-1} A.adjoint(p)>``.

    ``gap_S`` is the nonnegative Fenchel--Young gap of the *entire augmented*
    data fidelity (including the TV term). The second term is the squared
    dual Hilbert norm on the original domain. Both vanish exactly at the
    first-order optimality conditions when the additional penalty is zero.
    Their **sum** is tested against ``tol``. It is an optimality residual,
    not a conventional Fenchel duality gap or a calibrated error estimate.

    A primal PDHG solver provides its saddle variable ``p`` directly. For a
    standard FISTA run on TV_Setting, ``p`` is computed *at the current
    iterate* using ``-S.subgradient(Ax)/alpha``; the physical FISTA dual
    buffers are not in augmented coordinates. A dual PDHG solver is mapped
    back from the transformed setting using its primal and dual saddle
    variables. All three routes use the same two numerical criteria.

    History keys:
        - ``'fidelity gap'`` (first term),
        - ``'adjoint feasibility squared'`` (second term),
        - ``'augmented optimality residual'`` (their sum),
        - ``'duality gap'`` (compatibility alias of their sum for plotting;
          it is NOT the usual Fenchel duality gap).

    ``dual_candidate`` may override the FISTA dual choice; by default the
    requested augmented data-functional subgradient is always used.
    """

    def __init__(self, tol: float = 1e-5, schedule=None,
                 logging_level: int | str = 'INFO',
                 dual_candidate=None, stop_at_infty: bool = False) -> None:
        if (isinstance(tol, (bool, np.bool_)) or
                not isinstance(tol, (int, float, np.integer, np.floating))
                or not np.isfinite(tol) or tol < 0):
            raise ValueError('tol must be finite and nonnegative.')
        if dual_candidate is not None and not callable(dual_candidate):
            raise TypeError('dual_candidate must be callable or None.')
        if not isinstance(stop_at_infty, (bool, np.bool_)):
            raise TypeError('stop_at_infty must be boolean.')
        # Subclass DualityGapStopping for existing convergence_plots() support.
        # Never invoke its full-duality-gap _stop() or its reference-vector
        # setup: NullFunctional.conj is an indicator with infinite values.
        super().__init__(tol=float(tol), reltol=0., schedule=schedule,
                         logging_level=logging_level,
                         stop_at_infty=bool(stop_at_infty))
        self.dual_candidate = dual_candidate
        self._augmented: TVaug_Setting | None = None
        self._tv_source: str | None = None
        self.additional_gap_keys = ('fidelity gap',
                                    'adjoint feasibility squared')
        for key in (*self.additional_gap_keys, 'augmented optimality residual'):
            self.history_dict[key] = []

    def __repr__(self) -> str:
        return f'{type(self).__name__}(tol={self.tol})'

    def _complete_init_with_solver(self, solver) -> None:
        setting = solver.setting
        original = getattr(setting, '_primal_setting', None)
        if isinstance(original, TVaug_Setting):
            augmented, source = original, 'dual-pdhg'
        elif isinstance(setting, TVaug_Setting):
            augmented, source = setting, 'pdhg'
        else:
            augmented = getattr(setting, 'tvaug_setting', None)
            source = 'fista'
        if not isinstance(augmented, TVaug_Setting):
            raise ValueError('TVaugOptimalityStopping requires an augmented TV setting.')
        if not augmented._tv_null_penalty:
            raise ValueError('This two-term criterion requires a null additional penalty. '
                             'Use the full duality gap for nonzero R.')
        self._augmented = augmented
        self._tv_source = source
        # Skip DualityGapStopping._complete_init_with_solver(): it attempts
        # the full R*-based dual gap and computes a penalty reference value.
        StopRule._complete_init_with_solver(self, solver)

    def reset(self) -> None:
        super().reset()
        self._augmented = None
        self._tv_source = None

    def _current_pair(self):
        """Return x, A(x), and p in the *original* augmented coordinates."""
        solver = self.solver
        augmented = self._augmented
        if solver is None or augmented is None:
            raise RuntimeError('The TVaug stopping rule has not been attached.')
        if self._tv_source == 'fista':
            x = solver.x
            y = augmented.op(x)
            if self.dual_candidate is None:
                # In particular, the TV-block subgradient must come from
                # alpha*H_beta(grad(x)), not from physical FISTA's dual.
                q = augmented.data_fid.subgradient(y)
                p = (-1. / augmented.regpar) * q
            else:
                p = self.dual_candidate(x, augmented)
        elif self._tv_source == 'dual-pdhg':
            # On the dual Setting, solver.primal()[0] is original p, while
            # solver.dual()[0] = -x / alpha (see Setting.get_dual_setting).
            p, _ = solver.primal()
            pstar, _ = solver.dual()
            if p is None or pstar is None:
                raise ValueError('Dual PDHG must provide both saddle variables.')
            x = -pstar / augmented.regpar
            y = augmented.op(x)
        else:
            (x, y), (p, _) = solver.primal(), solver.dual()
            if y is None and x is not None:
                y = augmented.op(x)
        if x is None or y is None or p is None:
            raise ValueError('TVaug optimality requires primal and dual iterates.')
        if x not in augmented.op.domain or p not in augmented.op.codomain:
            raise ValueError('TVaug saddle vectors belong to the wrong spaces.')
        return x, y, p

    def _error_terms(self):
        """Evaluate the two mathematical terms without changing solver state."""
        augmented = self._augmented
        if augmented is None:
            raise RuntimeError('The TVaug stopping rule has not been attached.')
        _, y, p = self._current_pair()
        alpha = augmented.regpar
        q = -alpha * p
        fidelity = augmented.data_fid
        # <q,y> uses the canonical duality pairing of the augmented codomain.
        fy = (fidelity(y) + fidelity.conj(q)
              - augmented.op.codomain.vdot(q, y).real) / alpha
        adj = augmented.op.adjoint(p)
        riesz = augmented.h_domain.gram_inv(adj)
        feas = augmented.op.domain.vdot(adj, riesz).real
        if np.isfinite(fy):
            fy = max(0., float(fy))  # suppress tiny negative cancellation
        if np.isfinite(feas):
            feas = max(0., float(feas))
        return float(fy), float(feas)

    def _stop(self) -> bool:
        fidelity, feasibility = self._error_terms()
        total = fidelity + feasibility
        self.history_dict['fidelity gap'].append(fidelity)
        self.history_dict['adjoint feasibility squared'].append(feasibility)
        self.history_dict['augmented optimality residual'].append(total)
        self.history_dict['duality gap'].append(total)
        self.log_info = ('Augmented TV: fidelity gap={:.3e}, '
                         'adjoint feasibility squared={:.3e}, sum={:.3e} '
                         '(tol={:.3e})').format(
                             fidelity, feasibility, total, self.tol)
        if np.isnan(total) or (np.isinf(total) and self.stop_at_infty):
            return self._set_stop_status(False, self.log_info, failure=True)
        if np.isfinite(total) and total <= self.tol:
            if self.is_main_rule:
                self.log.info(self.log_info)
            return self._set_stop_status(True, self.log_info)
        if self.is_main_rule:
            self.log.debug(self.log_info)
        return False


class TVaugPDHGGapStopping(TVaugOptimalityStopping):
    """Default two-term TVaug optimality monitor for primal or dual PDHG.

    The public name is retained for compatibility with previous TV settings.
    """


class TVaugFISTAGapStopping(TVaugOptimalityStopping):
    """Default TVaug optimality monitor for standard TV_Setting FISTA.

    Its dual variable is reconstructed from the subgradient of the *augmented*
    data functional at the current primal FISTA iterate, even when the
    physical FISTA dual buffers are stale or initialized to zero.
    """

    def _complete_init_with_solver(self, solver) -> None:
        super()._complete_init_with_solver(solver)
        # Direct FISTA on TVaug_Setting uses the same augmented subgradient
        # reconstruction as FISTA on the physical TV_Setting.
        if self._tv_source == 'pdhg':
            self._tv_source = 'fista'
        if self._tv_source != 'fista':
            raise ValueError('TVaugFISTAGapStopping requires a primal TV FISTA solver.')


def _tvaug_pdhg_gap_factory(setting: Setting, params: dict) -> StopRule:
    """TVaug optimality residual and an independent hard iteration cap."""
    from regpy.stoprules.general import CountIterations_factory
    opts = dict(params.get('rule_params', {}))
    for key in ('tol', 'schedule', 'logging_level', 'stop_at_infty'):
        if key in params:
            opts.setdefault(key, params[key])
    opts.setdefault('tol', 1e-5)
    opts.setdefault('schedule', None)
    return TVaugPDHGGapStopping(**opts) + CountIterations_factory(setting, params)


def _tvaug_fista_gap_factory(setting: Setting, params: dict) -> StopRule:
    """The same two-term criterion, using the augmented FISTA subgradient."""
    from regpy.stoprules.general import CountIterations_factory
    opts = dict(params.get('rule_params', {}))
    for key in ('tol', 'schedule', 'logging_level', 'dual_candidate',
                'stop_at_infty'):
        if key in params:
            opts.setdefault(key, params[key])
    opts.setdefault('tol', 1e-5)
    # Evaluate every FISTA iterate, including iteration zero.
    opts['schedule'] = None
    return TVaugFISTAGapStopping(**opts) + CountIterations_factory(setting, params)


class TV_Setting(Setting):
    r"""Ordinary TV-penalized setting with the original forward operator.

    The primal problem (when ``regpar`` is positive) is

    .. math::
       S_{data}(T x)/\alpha + TV_\beta(x) + R(x).

    ``R`` is an optional additional convex penalty; by default it is zero.
    Passing ``penalty=TV`` or a concrete ``TVUniformGridFcts`` supplies the
    TV penalty itself (without a second copy of TV). Other ``penalty`` values
    are interpreted as an *additional* functional R.
    Unlike :class:`TVaug_Setting`, ``op`` is the *physical* forward operator
    ``T`` and ``penalty`` contains TV explicitly.  The augmented
    :class:`TVaug_Setting` is created only for convex, linear Tikhonov
    settings and stored in :attr:`tvaug_setting`. Its saddle solver is
    registered here as ``PDHGaug`` and selected as the default minimizer
    (also the default inner minimizer).

    The optional ``exact_data`` is retained for synthetic data generation.
    Data updates on the regular setting also update the augmented setting.

    A standard FISTA solve is supported whenever the TV penalty exposes an
    applicable proximal operator. For zero additional R, its default rule
    evaluates the **augmented fidelity Fenchel--Young gap plus the squared
    adjoint-feasibility residual** on every outer iterate. The saddle dual
    variable is reconstructed as ``p=-S_aug.subgradient(Ax)/alpha`` at each
    FISTA iterate; it is not the physical FISTA solver's stored dual vector.
    The two terms have separate histories. A nonzero additional penalty
    retains the ordinary full-gap rule. The iteration cap is always present.
    A chosen TV subgradient at a nondifferentiable point may be conservative
    even when another subgradient would satisfy the stationarity equation.

    To ensure that the two formulations have the same objective, the default
    ``incorporate_init_in_penalty`` is False: changing an initial guess must
    not shift the TV seminorm. A nonzero additional penalty R can prevent
    FISTA from having a simple proximal operator; ``PDHGaug`` remains usable
    provided R has a proximal operator.
    """

    def __init__(self,
                 op: Operator,
                 data_fid: Functional | HilbertSpace | Callable | None = None,
                 penalty: Functional | HilbertSpace | Callable | None = None,
                 regpar: float | None = None,
                 beta: float = 0.,
                 data: ReadOnlyVector | None = None,
                 exact_data: ReadOnlyVector | None = None,
                 grad: Operator | None = None,
                 boundary_condition: str = 'Neum',
                 data_operator_scale: float | str = 'auto',
                 init: ReadOnlyVector | None = None,
                 incorporate_init_in_penalty: bool = False,
                 logging_level: int | str = 'INFO',
                 explicitDataWarning: bool = True) -> None:
        from regpy.functionals import TV
        from regpy.functionals.base import NullFunctional, as_functional
        from regpy.functionals.vector_integral_functionals import TVUniformGridFcts
        from regpy.vecsps import UniformGridFcts

        if not isinstance(op.domain, UniformGridFcts):
            raise TypeError('TV_Setting requires a UniformGridFcts domain.')
        if (isinstance(beta, (bool, np.bool_))
                or not isinstance(beta, (int, float, np.floating))
                or not np.isfinite(beta) or beta < 0):
            raise ValueError('beta must be finite and nonnegative.')
        if incorporate_init_in_penalty:
            raise ValueError('TV_Setting requires incorporate_init_in_penalty=False '
                             'to keep TV(x) unchanged by the starting point.')
        if grad is not None:
            raise ValueError("TV_Setting uses the TV functional's own gradient; "
                             'for custom gradient operators use TVaug_Setting.')
        self.data_operator_scale = data_operator_scale
        # ``penalty=TV`` or a concrete TVUniformGridFcts denotes *the* TV
        # functional.  Other supplied penalties retain the older augmented
        # TV interface's meaning: an additional regularizer R.
        if penalty is TV or isinstance(penalty, TVUniformGridFcts):
            self.tv = (TV(op.domain, beta=float(beta),
                          boundary_condition=boundary_condition)
                       if penalty is TV else penalty)
            self._tv_additional_penalty = NullFunctional(op.domain)
            standard_penalty = self.tv
            self.beta = float(self.tv.beta)
        else:
            self.beta = float(beta)
            self.tv = TV(op.domain, beta=self.beta,
                         boundary_condition=boundary_condition)
            self._tv_additional_penalty = (
                NullFunctional(op.domain) if penalty is None
                else as_functional(penalty, op.domain))
            standard_penalty = (
                self.tv if penalty is None
                else self.tv + self._tv_additional_penalty)
        self.boundary_condition = boundary_condition
        self._tv_ready = False
        self.tvaug_setting: TVaug_Setting | None = None
        super().__init__(
            op=op, penalty=standard_penalty,
            data_fid=L2(op.codomain) if data_fid is None else data_fid,
            # TV.reference_vector() falls back to TV.proximal(0), whose
            # default implementation runs *dual FISTA*. The zero image is
            # already a feasible TV minimizer. Supply it explicitly to
            # avoid running a full inner solver just to construct a Setting.
            regpar=regpar, data=data,
            init=(op.domain.zeros() if init is None else init),
            incorporate_init_in_penalty=False,
            logging_level=logging_level,
            explicitDataWarning=explicitDataWarning,
            exact_data = exact_data,
            default_linear_regmethod='alGr',
            default_nonlinear_regmethod='IRN',
            default_inner_solver='PDHGaug',
        )
        self._tv_ready = True
        self._sync_tvaug_setting()

    def _sync_tvaug_setting(self) -> None:
        """Build, update, or remove the optional convex saddle-point form."""
        if not self.is_tikhonov or not self.is_convex:
            self.tvaug_setting = None
            # An unregularized TV_Setting uses an alpha-grid regularizer.
            # Its inner method is carried into a fresh generic Tikhonov
            # Setting via RunOnAlphaGrid/Bregman; the wrapper rebuilds the
            # augmented saddle formulation for that inner problem.
            if not self.is_tikhonov:
                from regpy.solvers.setting import get_convex_solver_dictionary
                self._convex_solvers = getattr(self, '_convex_solvers', {})
                if 'PDHGaug' not in self._convex_solvers:
                    template = deepcopy(get_convex_solver_dictionary()['PDHG'])
                    template['class'] = PDHGaug
                    template['full'] = 'PDHG on the augmented TV inner setting'
                    # Inner alpha-grid settings are ordinary Setting instances;
                    # they must not fall back to RelativeChangeSol either.
                    if isinstance(self._tv_additional_penalty, NullFunctional):
                        template['stoprule_params'].update({
                            'factory': _tvaug_pdhg_gap_factory,
                            'tol': 1e-5,
                            'schedule': None,
                        })
                    self._convex_solvers['PDHGaug'] = template
            return
        if self.tvaug_setting is None:
            self.tvaug_setting = TVaug_Setting(
                op=self.op,
                data_fid=self.data_fid,
                penalty=self._tv_additional_penalty,
                regpar=self.regpar,
                beta=self.beta,
                data=self.data,
                grad=self.tv.grad,
                boundary_condition=self.boundary_condition,
                data_operator_scale=self.data_operator_scale,
                init=self.init,
                incorporate_init_in_penalty=False,
                logging_level=self.log.level,
                explicitDataWarning=False,
            )
        else:
            self.tvaug_setting.regpar = self.regpar

        # Only Tikhonov convex settings have a method registry containing
        # PDHG and FISTA.  Running PDHGaug is a call to PDHG on TVaug.
        self.add_method('PDHGaug', PDHGaug, primal=True,
                        full_method_name='PDHG on the augmented TV setting')
        self._default_inner_solver = 'PDHGaug'
        # The outer PDHGaug solver is registered on TV_Setting but executes
        # on TVaug_Setting. Install the two-term optimality monitor on the
        # outer registry as well. With nonzero R, retain the full gap.
        if self.tvaug_setting._tv_null_penalty:
            pdhg_gap_factory = _tvaug_pdhg_gap_factory
        else:
            pdhg_gap_factory = 'DualityGapStopping'
        self.set_stoprule_params('PDHGaug', {
            'factory': pdhg_gap_factory,
            'max_it': 1000,
            'tol': 1e-5,
            'reltol': 0.,
            'stop_at_infty': False,
            'schedule': None,
        })
        if 'FISTA' in self._methods and self.tvaug_setting._tv_null_penalty:
            self.set_stoprule_params('FISTA', {
                'factory': _tvaug_fista_gap_factory,
                'max_it': 1000,
                'schedule': None,
            })

    @Setting.regpar.setter
    def regpar(self, new_regpar: float | None) -> None:
        Setting.regpar.fset(self, new_regpar)
        if getattr(self, '_tv_ready', False):
            self._sync_tvaug_setting()

    @regpar.deleter
    def regpar(self) -> None:
        Setting.regpar.fdel(self)
        if getattr(self, '_tv_ready', False):
            self._sync_tvaug_setting()

    def get_linearized_tikh_setting(self, *args, **kwargs):
        """Retain the TV split for alpha-grid inner Tikhonov problems.

        Other linearization points (and explicitly different Bregman or
        shifted penalties) are delegated unchanged to the base class.
        """
        inner_penalty = kwargs.get('inner_penalty', False)
        inner = super().get_linearized_tikh_setting(*args, **kwargs)
        point = kwargs.get('x0', args[0] if args else None)
        if (inner_penalty is self.penalty and point is not None
                and np.all(np.asarray(point) == 0.)
                and inner.op.linear and inner.is_convex):
            inner._tvaug_constructor = {
                'grad': self.tv.grad, 'beta': self.beta,
                'additional_penalty': self._tv_additional_penalty,
                'scale': self.data_operator_scale,
            }
        return inner

    def set_data_fid(self, data_fid) -> None:
        """Refresh the augmented representation when fidelity is replaced."""
        super().set_data_fid(data_fid)
        if getattr(self, '_tv_ready', False):
            self.tvaug_setting = None
            self._sync_tvaug_setting()

    def set_penalty(self, penalty) -> None:
        """Keep the explicit TV decomposition immutable after construction."""
        if getattr(self, '_tv_ready', False):
            raise RuntimeError(
                'TV_Setting has a fixed TV penalty decomposition; create a '
                'new TV_Setting to replace TV or the additional penalty.')
        super().set_penalty(penalty)

    def change_data(self, new_data) -> None:
        """Update the physical observations and the augmented data term."""
        super().change_data(new_data)
        if getattr(self, '_tv_ready', False) and self.tvaug_setting is not None:
            self.tvaug_setting.change_data(new_data)

    def change_init(self, new_init) -> None:
        """Keep both starting points equal without translating the TV penalty."""
        super().change_init(new_init)
        if getattr(self, '_tv_ready', False) and self.tvaug_setting is not None:
            self.tvaug_setting.change_init(new_init)
