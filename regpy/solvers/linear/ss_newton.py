from math import sqrt,inf
from collections.abc import Iterable
from numbers import Real
import numpy as np
import logging as log
from copy import deepcopy

from regpy.util import Errors
from regpy.vecsps import NumPyVectorSpace
from regpy.operators import CoordinateMask
from regpy.operators.base import Composition
from regpy.hilbert import GramHilbertSpace, HilbertSpace
from regpy.functionals.base import Functional, SquaredNorm, Composed, Conj, HorizontalShiftDilation
from regpy.stoprules import CountIterations
from regpy.functionals.integral_functionals import IntegralFunctionalBase

from ..sequences import GeometricSequence
from ..solver import RegSolver
from ..setting import Setting
from .tikhonov import TikhonovCG
__all__ = ["SemiSmoothNewton"]

def quadTaylor(func:Functional,x):
    """returns the quadratic Taylor polynomial of a functional at a given point as a SquaredNorm functional.
    The Gram matrix of the Hilbert space underlying the new functional is set to the Hessian of the given functional.
    """
    if not isinstance(func,Functional):
        raise TypeError(Errors.type_error("func must be a Functional",func))
    if not x in func.domain:
        raise ValueError(Errors.not_in_vecsp("x must be in the domain of func",x))
    hess = func.hessian(x)
    grad = func.subgradient(x)
    fx = func(x)
    h_space = GramHilbertSpace(hess)
    b=hess.inverse(grad)-x
    c= fx-np.vdot(0.5*x+b,hess(x))
    return SquaredNorm(h_space,a=1.,b=b,c=c)

class SemiSmoothNewton(RegSolver):
    r"""Semi-smooth Newton method for minimizing smooth Tikhonov functionals

    .. math::
        \mathcal{S}_{g^{\delta}}(Tf) + \alpha * \mathcal{R}(f) 

    subject to bilateral constraints 

    ..math::
        psi_{minus} \leq f \leq psi_{plus} 
    
    for a separable smooth penalty functional :math:`\mathcal{R}` and a smooth, strictly convex data functional :math:`\mathcal{S}_{g^{\delta}}` 
    and a linear operator :math:`T`.
    
    Parameters
    ----------
    setting: Setting
        The setting. The operator domain must be a real NumPyVectorSpace
        equipped with a Hilbert space. 
    init: np.ndarray or None, default: None
        Zeroth iterate. If None, it is taken from the setting. 
        init must be feasible.
    psi_plus, psi_minus : real scalar or array, optional
        Bound constraints, defaulting to the penalty's essential-domain bounds
        (including constr_u/constr_l). Finite bounds must be in its essential
        domain; infinite bounds require an unbounded domain in that direction.

    cg_pars: dictionary, default: None
        Additional CG parameters. The dictionary is copied. The reltolx value
        is supplied by cg_reltols, overriding any reltolx entry here; other
        stopping tolerances, such as tol and reltoly, remain unchanged.
    cg_reltols : iterable of float or None, optional
        Positive finite relative domain tolerances for successive CG solves.
        None creates a fresh GeometricSequence(1.0, 1.0/3.0), yielding
        1, 1/3, 1/9, ... . The first value is used for initialization, then
        one value is consumed per inactive-set CG solve. Steps with all
        indices active consume no value. Values are validated when consumed.
        Exhaustion raises ValueError if another CG solve is needed; it does
        not imply convergence of the Newton iteration.
    logging_level: Loglevel
        default: logging:INFO
    cg_logging_level: Loglevel
        default: logging.INFO

    Notes
    -----
    A penalty with restricted essential domain must be an IntegralFunctionalBase
    or the conjugate of one, optionally wrapped in HorizontalShiftDilation.
    Positive and negative dilations (including the dual scale -regpar) and
    horizontal/data shifts are preserved in the private extended penalty.
    Bounds are always extracted in the original solver coordinates.
    The operator domain must have real Hilbert geometry.
    Hard bounds are retained for the active-set method. In the private copy of
    a direct integral penalty, hard constraints are removed and finite natural
    endpoints are extended quadratically at the actual constraint (not at an
    interior midpoint). Finite values and first and second derivatives are
    required at that point, even when it is a natural endpoint.

    For a conjugate integral penalty, linear extensions of its private primal
    functional are replaced by quadratic extensions at the same primal joins.
    This extends the dual penalty quadratically beyond its original bounds.
    The primal curvature must be finite and strictly positive at each join;
    otherwise an error is raised. Any remaining restricted dual domain also
    raises an error. The original data fidelity is not extended: changing it
    would change the objective rather than merely extend the penalty outside
    the active-set constraints.
    The original setting penalty is not modified.

    """

    def __init__(self, 
                 setting: Setting,
                 init: np.ndarray | None = None,
                 psi_plus: np.ndarray | None = None, 
                 psi_minus: np.ndarray | None = None,
                 cg_pars = None, 
                 logging_level = "INFO", 
                 cg_logging_level = "WARNING",
                 cg_reltols: Iterable[float] | None = None,
                 **kwargs
                 ):
        super().__init__(setting)

        out, _ = SemiSmoothNewton.check_applicability(self.setting)
        if not out['applicable']:
            raise RuntimeError('SemismoothNewton_bilateral not applicable to this setting. '+out['info'])
        self.S = self.setting.data_fid
        original_penalty = self.setting.penalty
        self.psi_plus = self._constraint_vector(
            original_penalty.dom_u if psi_plus is None else psi_plus, 'psi_plus'
        )
        self.psi_minus = self._constraint_vector(
            original_penalty.dom_l if psi_minus is None else psi_minus, 'psi_minus'
        )
        if np.any(self.psi_minus >= self.psi_plus):
            raise ValueError("psi_minus must be strictly below psi_plus.")
        self._validate_constraints(original_penalty)
        self.R = self._extended_penalty(original_penalty)
        regpar= setting.regpar
        x0 = setting.get_or_update_initial_guess(init, update=False)
        if not np.all(x0<=self.psi_plus) or not np.all(x0>=self.psi_minus):
            raise ValueError(Errors.generic_message('x0 not feasible.'))
        y0 = self.op(x0)
        quad_setting = Setting(self.op, quadTaylor(self.R,x0),quadTaylor(self.S,y0),regpar=regpar)

        cg_pars = dict(cg_pars) if cg_pars is not None else {}
        self.cg_pars = cg_pars
        self.cg_reltols = iter(
            GeometricSequence(1.0, 1.0 / 3.0)
            if cg_reltols is None else cg_reltols
        )
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""

        self.log.setLevel(logging_level)
        self.cg_logging_level = cg_logging_level

        """Prepare first iteration step"""
        self.lam_plus = self.op.domain.zeros()
        self.lam_minus = self.op.domain.zeros()

        self.tikhcg=TikhonovCG(quad_setting, 
                               x0 = x0, 
                               logging_level=self.cg_logging_level, 
                               **self._next_cg_parameters()
                               )
        self.x, self.y = self.tikhcg.run()

        self.active_plus = self.op.domain.IfPos(self.lam_plus +self.regpar*(self.x-self.psi_plus ))
        self.active_minus = self.op.domain.IfPos(self.lam_minus-self.regpar*(self.x-self.psi_minus))
        if not (self.active_plus).any() and not (self.active_minus).any() and self.data_fid.is_quadratic:
            self.log.info('Stopped at 0th iterate.')
            self.converge()
        self.log.info('it {}: CG its {}; changes active sets +{},-{}'.format(
            self.iteration_step_nr,self.tikhcg.iteration_step_nr,
            (1.*self.active_minus+self.active_plus).sum(),0 )
        )

    def _next_cg_parameters(self) -> dict:
        """Consume one relative tolerance for an actual CG solve."""
        try:
            tolerance = next(self.cg_reltols)
        except StopIteration as error:
            raise ValueError(
                "cg_reltols exhausted before the next CG solve; "
                "supply more tolerances or an infinite iterator."
            ) from error
        if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real):
            raise TypeError("CG relative tolerances must be positive finite real numbers.")
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("CG relative tolerances must be positive and finite.")
        parameters = dict(self.cg_pars)
        parameters['reltolx'] = tolerance
        return parameters

    def _constraint_vector(self, value, name):
        array = np.asarray(value)
        if np.iscomplexobj(array):
            raise TypeError(f"{name} must be real.")
        if array.ndim != 0 and array.shape != self.op.domain.shape:
            raise ValueError(f"{name} must be scalar or have the domain shape.")
        result = np.array(np.broadcast_to(array, self.op.domain.shape), dtype=float, copy=True)
        if np.any(np.isnan(result)):
            raise ValueError(f"{name} must not contain NaN.")
        return result

    def _validate_constraints(self, penalty):
        lower, upper = np.asarray(penalty.dom_l), np.asarray(penalty.dom_u)
        for name, bound in [('psi_minus', self.psi_minus), ('psi_plus', self.psi_plus)]:
            if np.any(bound < lower) or np.any(bound > upper):
                raise ValueError(f"{name} is outside the penalty's essential domain.")
            if name == 'psi_minus' and np.any(np.isposinf(bound)):
                raise ValueError("psi_minus cannot be +infinity.")
            if name == 'psi_plus' and np.any(np.isneginf(bound)):
                raise ValueError("psi_plus cannot be -infinity.")
            # Check finite boundary values, including open natural endpoints.
            # Unbounded coordinates are filled by a feasible reference.
            finite = np.isfinite(bound)
            if np.any(finite):
                probe = np.array(penalty.reference_vector(), copy=True)
                probe[finite] = bound[finite]
                try:
                    valid = np.isfinite(penalty(probe))
                except (ValueError, NotImplementedError) as error:
                    raise ValueError(f"{name} is not in the penalty's essential domain.") from error
                if not valid:
                    raise ValueError(f"{name} is not in the penalty's essential domain.")

    @staticmethod
    def _check_extension_point(functional, threshold, mask, *, positive=False):
        """Require finite Taylor data; dual quadratic tails need positive curvature."""
        for method in ('_f', '_f_deriv', '_f_second_deriv'):
            values = functional.domain.zeros()
            with np.errstate(all='ignore'):
                getattr(functional, method)(
                    threshold, out=values, mask=mask, **functional.kwargs
                )
            if not np.all(np.isfinite(values[mask])):
                raise ValueError(
                    f"Cannot construct quadratic extension: {method} "
                    "is nonfinite at the extension point."
                )
            if positive and method == '_f_second_deriv' and np.any(values[mask] <= 0):
                raise ValueError(
                    "A quadratic extension of the conjugate requires a "
                    "strictly positive primal second derivative at the join."
                )

    @staticmethod
    def _is_integral_conjugate(penalty) -> bool:
        """Recognize conjugates through horizontal shift/dilation wrappers."""
        while isinstance(penalty, HorizontalShiftDilation):
            penalty = penalty.func
        return isinstance(penalty, Conj) and isinstance(penalty.func, IntegralFunctionalBase)

    def _extended_penalty(self, penalty):
        """Extend a private penalty while preserving the original active-set bounds."""
        if (isinstance(penalty, HorizontalShiftDilation)
                and self._is_integral_conjugate(penalty)):
            # The active-set bounds were extracted from the original wrapper,
            # including the exchange of lower/upper bounds for negative scale.
            # Keep the same affine coordinates for the extended functional.
            result = deepcopy(penalty)
            result.func = self._extended_penalty(penalty.func)
            result.__dict__.pop('conj', None)
            return result
        if isinstance(penalty, Conj) and isinstance(penalty.func, IntegralFunctionalBase):
            primal = deepcopy(penalty.func)
            for side in ('l', 'u'):
                linear = getattr(primal, 'lin_taylor_' + side)
                if linear is None:
                    continue
                threshold = np.array(
                    np.broadcast_to(linear, primal.domain.shape), copy=True
                )
                natural = getattr(primal, '_nat_dom_' + side)
                mask = np.isfinite(threshold) & (
                    threshold >= natural if side == 'l' else threshold <= natural
                )
                if np.any(mask):
                    self._check_extension_point(primal, threshold, mask, positive=True)
                # The conjugate of a positive-curvature quadratic tail is
                # quadratic beyond the former dual constraint f'(threshold).
                setattr(primal, 'lin_taylor_' + side, None)
                if np.any(mask):
                    setattr(primal, 'quad_taylor_' + side, threshold)
            if not primal.conj_everywhere_finite:
                raise ValueError(
                    "The conjugate penalty still has a restricted domain after "
                    "replacing primal linear extensions by quadratic extensions."
                )
            return primal.conj
        if not isinstance(penalty, IntegralFunctionalBase):
            return penalty
        result = deepcopy(penalty)
        # Constraints belong to the active-set method, not the smooth model.
        result.constr_l = None
        result.constr_u = None
        for side, bound in [('l', self.psi_minus), ('u', self.psi_plus)]:
            natural = np.asarray(getattr(result, '_nat_dom_' + side))
            mask = np.isfinite(natural) & np.isfinite(bound)
            if not np.any(mask):
                continue
            existing = getattr(result, 'quad_taylor_' + side)
            threshold = (np.array(existing, copy=True) if existing is not None
                         else np.full(self.op.domain.shape, -np.inf if side == 'l' else np.inf))
            threshold[mask] = bound[mask]
            self._check_extension_point(result, threshold, mask)
            setattr(result, 'lin_taylor_' + side, None)
            setattr(result, 'quad_taylor_' + side, threshold)
        if not result.everywhere_finite:
            raise ValueError("The penalty could not be extended to the full domain space.")
        return result

    def _next(self):
        """compute active and inactive sets"""
        self.active_plus_old=self.active_plus
        self.active_minus_old=self.active_minus
        self.active  = self.active_plus | self.active_minus
        self.inactive= self.op.domain.logical_not(self.active)

        """On the active sets the solution takes the values of the constraints."""
        self.x[self.active_plus]=self.psi_plus[self.active_plus]
        self.x[self.active_minus]=self.psi_minus[self.active_minus]

        """Lagrange parameters are 0 where the corresponding constraints are not active."""
        self.lam_plus[self.inactive]=0
        self.lam_plus[self.active_minus]=0
        self.lam_minus[self.inactive]=0
        self.lam_minus[self.active_plus]=0

        projection = CoordinateMask(self.h_domain.vecsp, self.inactive)
        # Clamping changes x; linearize the fidelity at its current image.
        self.y = self.op(self.x)
        S_taylor = quadTaylor(self.S, self.y)
        full_R_taylor = quadTaylor(self.R, self.x)
        if self.active.all():
            self.log.info('all indices active!')
        else:
            """Define quadratic Taylor approximation of the data fidelity functional at the current iterate"""
            """... and a quadratic Taylor approximation of the full penalty functional"""
            """ ... as well as its restriction to the inactive set.""" 
            hess_R_full =  full_R_taylor.gram
            constr_R_taylor = Composed(
                full_R_taylor,
                op = projection.adjoint,
                hessian = lambda x: Composition(projection.adjoint, 
                    hess_R_full, projection,
                    inverse = projection.adjoint * hess_R_full.inverse * projection
                    ),
                h_domain=self.h_domain,
                )
            """Define a quadratic approximation of the setting at the current iterate restricted to the inactive set, 
            which respects the solution on the active set that we have already computed.
            Since we assume separability of the penalty term, the solution on the active set only affects the data term.
            """
            quad_setting = Setting(self.op * projection, 
                                  constr_R_taylor,
                                  S_taylor,
                                  regpar=self.regpar,
                                  data = self.op(projection(self.x)-self.x)
                                  )         
            tikhcg = TikhonovCG(quad_setting, 
                                init=projection(full_R_taylor.reference_vector()),                                
                                x0 = projection(self.x), 
                                logging_level=self.cg_logging_level,
                                **self._next_cg_parameters()
                                )
            aux, _ = tikhcg.run()
            self.x[self.inactive] = aux[self.inactive]
        self.y = self.op(self.x)

        S_taylor.data = self.op.codomain.zeros() # Has been changed by data in quad_setting!
        residual = self.op.adjoint(S_taylor.subgradient(self.y))
        residual += self.regpar*full_R_taylor.subgradient(self.x)
        residual = self.setting.h_domain.gram_inv(residual)
        
        self.lam_plus[self.active_plus]  = -residual[self.active_plus]
        self.lam_minus[self.active_minus]= +residual[self.active_minus]

        """Update active and inactive sets"""
        self.active_plus  = (self.lam_plus +self.regpar*(self.x-self.psi_plus )) >=0 
        self.active_minus = (self.lam_minus-self.regpar*(self.x-self.psi_minus)) >=0
        added_ind = (self.op.domain.logical_and(self.active_plus,  self.op.domain.logical_not(self.active_plus_old ))).sum() \
                  + (self.op.domain.logical_and(self.active_minus, self.op.domain.logical_not(self.active_minus_old))).sum() 
        removed_ind = (self.op.domain.logical_and(self.active_plus_old, self.op.domain.logical_not(self.active_plus))).sum() \
                + (self.op.domain.logical_and(self.active_minus_old, self.op.domain.logical_not(self.active_minus))).sum()
        self.log.info('it {}: CG its {}, changes active sets +{},-{}'.format(
            self.iteration_step_nr,
            tikhcg.iteration_step_nr if not self.active.all() else 0,
            added_ind, 
            removed_ind
            )
        )
        if added_ind+removed_ind==0 and self.data_fid.is_quadratic:
            self.converge()

    def primal(self):
        projected_x = self.x.copy()
        projected_x[self.active_minus] = self.psi_minus[self.active_minus]
        projected_x[self.active_plus] = self.psi_plus[self.active_plus]
        return (projected_x, self.op(projected_x))

    @staticmethod
    def check_applicability(setting,op_norm=None):
        out = {'info':''}
        if not setting.op.linear:
            out['info'] += 'Operator must be linear. '
        if (not isinstance(setting.h_domain, HilbertSpace)
                or setting.h_domain.vecsp != setting.op.domain
                or not isinstance(setting.op.domain, NumPyVectorSpace)
                or not np.issubdtype(setting.op.domain.dtype, np.floating)):
            out['info'] += 'Operator domain must carry a real Hilbert space. '
        if not setting.is_tikhonov:
            out['info'] += 'Setting has no regularization parameter.'
        if not setting.is_convex:
            out['info'] += 'Setting has not convex.'    
        if setting.codomain_free:
            out['info'] += 'SemiSmoothNewton requires codomain vectors. '
        else:
            if not 'hessian' in setting.data_fid.methods:
                out['info'] += 'Data functional not twice differentiable. '
            if not setting.data_fid.is_strictly_convex:
                out['info'] += 'Data fidelity term is not strictly convex.'
            if setting.data_fid.Lipschitz==np.inf:
                out['info'] += 'Data functional not Lipschitz continuous. '                
        penalty = setting.penalty
        if not penalty.is_separable:
            out['info'] += 'Penalty term not separable. '
        else:
            # everywhere_finite is not part of the general Functional API.
            # Essential-domain bounds are available for separable functionals.
            lower, upper = penalty.dom_l, penalty.dom_u
            full_domain = (
                lower is not None
                and upper is not None
                and np.all(np.isneginf(lower))
                and np.all(np.isposinf(upper))
            )
            integral_penalty = (
                isinstance(penalty, IntegralFunctionalBase)
                or SemiSmoothNewton._is_integral_conjugate(penalty)
            )
            if not full_domain and not integral_penalty:
                out['info'] += (
                    'A penalty with restricted or unspecified essential domain '
                    'must be an IntegralFunctionalBase, its conjugate, or a '
                    'horizontal shift/dilation of such a conjugate. '
                )
        if not 'hessian' in setting.penalty.methods:
            out['info'] += 'Penalty term not twice differentiable. '
        if not isinstance(setting.op.domain, NumPyVectorSpace):
            out['info'] += 'Operator domain not a NumPyVectorSpace. '
        out['applicable'] = out['info']==''
        out['rate'] = np.nan
        return out, None

    def _get_second_R_derivatives_at_bound_constraints(self):
        """
        Since the (separable!) penalty functional R may not be defined outside of the feasible set and SemiSmoothNewton may produce infeasible iterates 
        for which the gradient of the penalty must be evaluated, we extend the R by a quadratic function outside of the feasible interval in each variable.
        For the evaluation of the gradient of R, evaluations at the boundaries and knowledge of the second derivatives suffice. 

        Returns: 
        Rpp_plus, Rpp_minus:
            Arrays of the the of the grid containing the second derivatives of the component functions of R at the upper and lower bound.  
        """
        ub = self.psi_plus.copy()
        ub[ub==np.inf] = self.psi_minus[ub==np.inf]
        ub[ub==-np.inf] = 0.

        lb = self.psi_minus.copy()
        lb[lb==-np.inf] = self.psi_plus[lb==-np.inf]
        lb[lb==np.inf] = 0

        ones = self.h_domain.vecsp.ones()
        return self.R.hessian(ub)(ones), self.R.hessian(lb)(ones)
