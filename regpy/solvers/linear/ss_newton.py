from math import sqrt,inf
from collections.abc import Iterable
from numbers import Real
import numpy as np
import logging as log
from copy import deepcopy

from regpy.util import Errors
from regpy.vecsps.maskable import MaskableVectorSpace, MaskableDirectSum
from regpy.vecsps.base import TupleVector, ReadOnlyVector
from regpy.operators import CoordinateMask, CoordinateProjection, PtwMultiplication, Identity
from regpy.operators.base import Composition
from regpy.hilbert import GramHilbertSpace, HilbertSpace
from regpy.functionals.base import Functional, SquaredNorm, Composed, Conj, HorizontalShiftDilation, FunctionalOnDirectSum, LinearCombination
from regpy.stoprules import CountIterations
from regpy.functionals.integral_functionals import IntegralFunctionalBase

from ..sequences import Naturals
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
    c= fx-func.domain.vdot(0.5*x+b,hess(x))
    return SquaredNorm(h_space,a=1.,b=b,c=c)

class SemiSmoothNewton(RegSolver):
    r"""Semi-smooth Newton method for minimizing smooth Tikhonov functionals

    .. math::
        \mathcal{S}_{g^{\delta}}(Tf) + \alpha * \mathcal{R}(f) 

    subject to bilateral constraints 

    ..math::
        psi_{minus} \leq f \leq psi_{plus} 
    
    for a separable smooth penalty functional :math:`\mathcal{R}` and a smooth convex data functional (strictly convex or separable) :math:`\mathcal{S}_{g^{\delta}}` 
    and a linear operator :math:`T`.
    
    Parameters
    ----------
    is_primal : bool, optional
        Whether iterates represent the original primal problem (default True).
        Pass False when solving its dual. This labels coordinates and does not
        change the iteration formulas or construct a dual setting.
    setting: Setting
        The setting. The operator domain must be a real ordered MaskableVectorSpace
        equipped with a Hilbert space. 
    init: Vector or None, default: None
        Zeroth iterate. If None, it is taken from the setting. 
        init must be feasible. It is exposed as iteration zero through x, y
        and primal(), before any CG solve. Boundary multipliers are initialized
        from the objective gradient at this point, and the next active sets
        are computed from these multipliers. The first call to next() performs
        an ordinary active-set Newton step; there is no preliminary
        unconstrained Taylor solve. Iteration drivers retain their usual
        convention of yielding only completed steps.
    psi_plus, psi_minus : real scalar or domain vector, optional
        Bound constraints, defaulting to the penalty's essential-domain bounds
        (including constr_u/constr_l). Finite bounds must be in its essential
        domain; infinite bounds require an unbounded domain in that direction.

    cg_pars: dictionary, default: None
        Additional CG parameters. The dictionary is copied. The reltolx value
        is supplied by cg_reltols, overriding any reltolx entry here; other
        stopping tolerances, such as tol and reltoly, remain unchanged.
    cg_reltols : iterable of float or None, optional
        Positive finite relative domain tolerances for successive CG solves.
        None creates a fresh (1.0/3.0) ** Naturals(), yielding
        1, 1/3, 1/9, ... . The first value is used for the first Newton step, then
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
    Direct sums of these functionals are handled componentwise, recursively.
    The original setting penalty is not modified.

    For separable fidelities, each Hessian must be represented as pointwise
    multiplication or a scalar multiple of Identity, with finite nonnegative
    entries. Positive-curvature coordinates define the inner CG data space.
    Zero-curvature gradient contributions are added to the quadratic penalty
    with scale 1/regpar. An entirely affine fidelity model is solved directly
    using the shifted penalty reference, without consuming a CG tolerance.
    The penalty Taylor Hessian must still be invertible. Inner duals are lifted
    to the full codomain, using -S'/regpar on zero-curvature coordinates.

    Initial multipliers use the same Hilbert-gradient coordinates as the
    subsequent steps: G^{-1}(T* S'(T x0) + regpar R'(x0)), where G is the
    domain Gram operator. On upper bounds lam_plus is minus this gradient;
    on lower bounds lam_minus is plus this gradient. Other entries are zero.
    Multipliers are not clipped to be nonnegative: a negative boundary
    multiplier releases that constraint in the next active set.
    Equal initial and predicted active sets do not mark the solver converged;
    an interior starting point may still require a Newton solve.

    """

    def __init__(self, 
                 setting: Setting,
                 init: ReadOnlyVector | None = None,
                 psi_plus: ReadOnlyVector | float | None = None, 
                 psi_minus: ReadOnlyVector | float | None = None,
                 cg_pars = None, 
                 logging_level = "INFO", 
                 cg_logging_level = "WARNING",
                 cg_reltols: Iterable[float] | None = None,
                 is_primal: bool = True,
                 **kwargs
                 ):
        super().__init__(setting, is_primal=is_primal)

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
        if any(a >= b for a, b in zip(
                self.op.domain.iter_coefficients(self.psi_minus),
                self.op.domain.iter_coefficients(self.psi_plus), strict=True)):
            raise ValueError("psi_minus must be strictly below psi_plus.")
        self._validate_constraints(original_penalty)
        self.R = self._extended_penalty(original_penalty)
        x0 = self.op.domain.empty()
        self.op.domain.copyto(x0, setting.get_or_update_initial_guess(init, update=False))
        if any(not l <= v <= u for l, v, u in zip(
                self.op.domain.iter_coefficients(self.psi_minus),
                self.op.domain.iter_coefficients(x0),
                self.op.domain.iter_coefficients(self.psi_plus), strict=True)):
            raise ValueError(Errors.generic_message('x0 not feasible.'))
        self.x, self.y = x0, self.op(x0)
        self._cg_dual = None
        self._cg_dual_adjoint = None

        cg_pars = dict(cg_pars) if cg_pars is not None else {}
        self.cg_pars = cg_pars
        self.cg_reltols = iter(
            (1.0 * (1.0 / 3.0) ** Naturals())
            if cg_reltols is None else cg_reltols
        )
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""

        self.log.setLevel(logging_level)
        self.cg_logging_level = cg_logging_level

        # A^0 records the constraints attained by the supplied feasible x0.
        initial_plus = self.op.domain.eq_mask(self.x, self.psi_plus)
        initial_minus = self.op.domain.eq_mask(self.x, self.psi_minus)
        residual = self.op.adjoint(self.S.subgradient(self.y))
        residual = residual + self.regpar * self.R.subgradient(self.x)
        residual = self.setting.h_domain.gram_inv(residual)

        self.lam_plus = self.op.domain.zeros()
        self.lam_minus = self.op.domain.zeros()
        self.op.domain.copyto(self.lam_plus, -residual, where=initial_plus)
        self.op.domain.copyto(self.lam_minus, residual, where=initial_minus)

        # Store A^1 for the first ordinary Newton step. Do not infer
        # stationarity merely from A^0 == A^1: inactive gradients may be nonzero.
        self.active_plus = self.op.domain.geq_mask(self.lam_plus + self.regpar * (self.x - self.psi_plus), self.op.domain.zeros())
        self.active_minus = self.op.domain.geq_mask(self.lam_minus - self.regpar * (self.x - self.psi_minus), self.op.domain.zeros())

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
        space = self.op.domain
        if value is None:
            raise ValueError(f"{name} must be specified.")
        result = space.empty()
        if np.isscalar(value):
            if np.iscomplexobj(value):
                raise TypeError(f"{name} must be real.")
            space.fill(result, value)
        elif value in space:
            space.copyto(result, value)
        else:
            raise ValueError(f"{name} must be scalar or a domain vector.")
        if any(np.isnan(v) for v in space.iter_coefficients(result)):
            raise ValueError(f"{name} must not contain NaN.")
        return result

    def _validate_constraints(self, penalty):
        space = self.op.domain
        lower = self._constraint_vector(penalty.dom_l, "penalty lower bound")
        upper = self._constraint_vector(penalty.dom_u, "penalty upper bound")
        for name, bound in [('psi_minus', self.psi_minus), ('psi_plus', self.psi_plus)]:
            coefficients = list(space.iter_coefficients(bound))
            if any(v < l or v > u for v, l, u in zip(
                    coefficients, space.iter_coefficients(lower),
                    space.iter_coefficients(upper), strict=True)):
                raise ValueError(f"{name} is outside the penalty's essential domain.")
            if any(np.isposinf(v) if name == 'psi_minus' else np.isneginf(v)
                   for v in coefficients):
                raise ValueError(f"{name} has an invalid infinite bound.")
            finite = space.mask_from_iterable(np.isfinite(v) for v in coefficients)
            if space.nrTrues(finite):
                probe = space.empty()
                space.copyto(probe, penalty.reference_vector())
                space.copyto(probe, bound, where=finite)
                if not np.isfinite(penalty(probe)):
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

    def _extended_penalty(self, penalty, lower=None, upper=None):
        """Extend a private penalty while preserving the original active-set bounds."""
        lower = self.psi_minus if lower is None else lower
        upper = self.psi_plus if upper is None else upper
        if isinstance(penalty, FunctionalOnDirectSum):
            result = deepcopy(penalty)
            result.funcs = [
                self._extended_penalty(f, l, u)
                for f, l, u in zip(penalty.funcs, lower, upper, strict=True)
            ]
            result.__dict__.pop('conj', None)
            return result
        if (isinstance(penalty, HorizontalShiftDilation)
                and self._is_integral_conjugate(penalty)):
            # The active-set bounds were extracted from the original wrapper,
            # including the exchange of lower/upper bounds for negative scale.
            # Keep the same affine coordinates for the extended functional.
            result = deepcopy(penalty)
            result.func = self._extended_penalty(penalty.func, lower, upper)
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
        for side, bound in [('l', lower), ('u', upper)]:
            natural = np.asarray(getattr(result, '_nat_dom_' + side))
            mask = np.isfinite(natural) & np.isfinite(bound)
            if not np.any(mask):
                continue
            existing = getattr(result, 'quad_taylor_' + side)
            threshold = (np.array(existing, copy=True) if existing is not None
                         else np.full(penalty.domain.shape, -np.inf if side == 'l' else np.inf))
            threshold[mask] = bound[mask]
            self._check_extension_point(result, threshold, mask)
            setattr(result, 'lin_taylor_' + side, None)
            setattr(result, 'quad_taylor_' + side, threshold)
        if not result.everywhere_finite:
            raise ValueError("The penalty could not be extended to the full domain space.")
        return result

    def _fidelity_model(self):
        """Return (quadratic fidelity, projected operator, projection, linear gradient).

        The final vector is the full codomain covector on zero-curvature
        coordinates. No Hilbert Gram inverse is applied to this covector.
        """
        if not self.S.is_separable:
            return quadTaylor(self.S, self.y), self.op, None, None
        space = self.op.codomain
        hessian = self.S.hessian(self.y)
        curvature = hessian(space.ones())
        values = list(space.iter_coefficients(curvature))
        if any(not np.isfinite(v) or np.iscomplexobj(v) or v < 0 for v in values):
            raise ValueError("Fidelity Hessian must have finite nonnegative real entries.")
        positive = space.mask_from_iterable(v > 0 for v in values)
        gradient = self.S.subgradient(self.y)
        linear = space.zeros()
        space.copyto(linear, gradient, where=space.logical_not(positive))
        if space.nrTrues(positive) == 0:
            return None, None, None, linear
        projection = CoordinateProjection(space, positive)
        reduced_hessian = PtwMultiplication(projection.codomain, projection(curvature))
        reduced_y = projection(self.y)
        b = reduced_hessian.inverse(projection(gradient)) - reduced_y
        # This constant makes quadratic + <linear,z> equal the full Taylor
        # polynomial at y (and hence everywhere).
        constant = (self.S(self.y) - space.vdot(linear, self.y)
                    - projection.codomain.vdot(0.5 * reduced_y + b,
                                              reduced_hessian(reduced_y)))
        model = SquaredNorm(GramHilbertSpace(reduced_hessian), a=1., b=b, c=constant)
        return model, projection * self.op, projection, linear

    def _next(self):
        """compute active and inactive sets"""
        self.active_plus_old=self.active_plus
        self.active_minus_old=self.active_minus
        self.active = self.op.domain.logical_or(self.active_plus, self.active_minus)
        self.inactive= self.op.domain.logical_not(self.active)

        """On the active sets the solution takes the values of the constraints."""
        self.op.domain.copyto(self.x, self.psi_plus, where=self.active_plus)
        self.op.domain.copyto(self.x, self.psi_minus, where=self.active_minus)

        """Lagrange parameters are 0 where the corresponding constraints are not active."""
        self.op.domain.fill(self.lam_plus, 0, where=self.inactive)
        self.op.domain.fill(self.lam_plus, 0, where=self.active_minus)
        self.op.domain.fill(self.lam_minus, 0, where=self.inactive)
        self.op.domain.fill(self.lam_minus, 0, where=self.active_plus)

        projection = CoordinateMask(self.h_domain.vecsp, self.inactive)
        S_taylor, model_op, data_projection, linear = self._fidelity_model()
        cg_iterations = 0
        full_R_taylor = quadTaylor(self.R, self.x)
        if linear is not None:
            shift = full_R_taylor.gram.inverse(self.op.adjoint(linear) / self.regpar)
            full_R_taylor = SquaredNorm(full_R_taylor.h_domain, a=1.,
                                       b=full_R_taylor.b + shift, c=full_R_taylor.c)
        if (self.op.domain.nrTrues(self.active) == self.op.domain.dimension):
            self.log.info('all indices active!')
        elif S_taylor is None:
            # The full fidelity model is affine: separability of R makes
            # the inactive minimizer its shifted reference vector.
            self.op.domain.copyto(self.x, full_R_taylor.reference_vector(),
                                  where=self.inactive)
            self._cg_dual = -linear / self.regpar
            self._cg_dual_adjoint = None
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
            quad_setting = Setting(model_op * projection, 
                                  constr_R_taylor,
                                  S_taylor,
                                  regpar=self.regpar,
                                  data = model_op(projection(self.x)-self.x)
                                  )         
            tikhcg = TikhonovCG(quad_setting, 
                                init=projection(full_R_taylor.reference_vector()),                                
                                x0 = projection(self.x), 
                                logging_level=self.cg_logging_level,
                                **self._next_cg_parameters()
                                )
            aux, _ = tikhcg.run()
            # Capture before resetting S_taylor.data below: CG's dual belongs
            # to the shifted, restricted quadratic subproblem.
            self._store_cg_dual(tikhcg, data_projection, linear)
            cg_iterations = tikhcg.iteration_step_nr
            self.tikhcg = tikhcg
            self.op.domain.copyto(self.x, aux, where=self.inactive)
        self.y = self.op(self.x)

        #S_taylor.data = self.op.codomain.zeros() # Has been changed by data in quad_setting!
        #residual = self.op.adjoint(S_taylor.subgradient(self.y))
        #residual += self.regpar*full_R_taylor.subgradient(self.x)
        residual = self.op.adjoint(self.setting.data_fid.subgradient(self.y))
        residual += self.regpar * self.R.subgradient(self.x)
        residual = self.setting.h_domain.gram_inv(residual)
        
        self.op.domain.copyto(self.lam_plus, -residual, where=self.active_plus)
        self.op.domain.copyto(self.lam_minus, residual, where=self.active_minus)

        """Update active and inactive sets"""
        self.active_plus  = self.op.domain.geq_mask(self.lam_plus +self.regpar*(self.x-self.psi_plus ), self.op.domain.zeros()) 
        self.active_minus = self.op.domain.geq_mask(self.lam_minus-self.regpar*(self.x-self.psi_minus), self.op.domain.zeros())
        space = self.op.domain
        added_ind = sum(space.nrTrues(space.logical_and(new, space.logical_not(old)))
                        for new, old in ((self.active_plus, self.active_plus_old),
                                         (self.active_minus, self.active_minus_old)))
        removed_ind = sum(space.nrTrues(space.logical_and(old, space.logical_not(new)))
                          for new, old in ((self.active_plus, self.active_plus_old),
                                           (self.active_minus, self.active_minus_old)))
        self.log.info('it {}: CG its {}, changes active sets +{},-{}'.format(
            self.iteration_step_nr,
            tikhcg.iteration_step_nr if not (self.op.domain.nrTrues(self.active) == self.op.domain.dimension) else 0,
            added_ind, 
            removed_ind
            )
        )
        if added_ind+removed_ind==0 and self.data_fid.is_quadratic and self.R.is_quadratic:
            self.converge()

    def _store_cg_dual(self, solver, projection=None, linear=None) -> None:
        """Snapshot the latest inner dual before its Taylor functional changes."""
        pair = solver.dual()
        if pair is None or pair[0] is None:
            raise RuntimeError("The inner CG solver did not provide a dual iterate.")
        self._cg_dual = self.op.codomain.empty()
        value = pair[0] if projection is None else projection.adjoint(pair[0])
        if linear is not None:
            value = value - linear / self.regpar
        self.op.codomain.copyto(self._cg_dual, value)
        self._cg_dual_adjoint = None

    def dual(self):
        """Return the dual pair used by duality-based stopping rules.

        At iteration zero, use the neutral dual point p=0. Afterward,
        reconstruct the dual pair exactly as the setting previously did
        when this solver exposed only primal(). The CG iteration is unchanged.
        """
        if self.iteration_step_nr == 0:
            return (
                self.op.codomain.zeros(),
                self.op.domain.zeros(),
            )

        _, dual = self.setting._complete_primal_dual_tuples(
            primal=self.primal(),
            dual=None,
        )
        return dual
        
    def dual_cg(self):
        """Return the latest inner CG dual vector and its full adjoint image.

        At initialization use the neutral pair (0, 0). Inactive-set CG uses
        T composed with a coordinate mask, so its second dual component is
        projected and cannot be reused here. Compute T* p lazily and cache it.
        Steps with all coordinates active perform no CG solve and retain the
        previous dual pair. For nonquadratic fidelities p comes from the
        quadratic Taylor model; a smaller or finite original gap is not
        guaranteed. No projection or alteration of p is performed.
        """
        if self._cg_dual is None:
            self._cg_dual = self.op.codomain.zeros()
            self._cg_dual_adjoint = self.op.domain.zeros()
        elif self._cg_dual_adjoint is None:
            self._cg_dual_adjoint = self.op.adjoint(self._cg_dual)
        return self._cg_dual, self._cg_dual_adjoint

    def primal(self):
        projected_x = self.op.domain.empty()
        self.op.domain.copyto(projected_x, self.x)
        self.op.domain.copyto(projected_x, self.psi_minus, where=self.active_minus)
        self.op.domain.copyto(projected_x, self.psi_plus, where=self.active_plus)
        return (projected_x, self.op(projected_x))

    @staticmethod
    def check_applicability(setting,op_norm=None):
        out = {'info':''}
        if not setting.op.linear:
            out['info'] += 'Operator must be linear. '
        if (not isinstance(setting.h_domain, HilbertSpace)
                or setting.h_domain.vecsp != setting.op.domain
                or not isinstance(setting.op.domain, MaskableVectorSpace)
                or not setting.op.domain.is_ordered):
            out['info'] += 'Operator domain must carry a real Hilbert space, which is ordered and maskable. '
        if not setting.is_tikhonov:
            out['info'] += 'Setting has no regularization parameter.'
        if not setting.is_convex:
            out['info'] += 'Setting has not convex.'    
        if setting.codomain_free:
            out['info'] += 'SemiSmoothNewton requires codomain vectors. '
        else:
            if not 'hessian' in setting.data_fid.methods:
                out['info'] += 'Data functional not twice differentiable. '
            if not (setting.data_fid.is_strictly_convex or setting.data_fid.is_separable):
                out['info'] += 'Data fidelity must be strictly convex or separable. '
            if setting.data_fid.is_separable and (
                    not isinstance(setting.op.codomain, MaskableVectorSpace)
                    or not setting.op.codomain.is_ordered):
                out['info'] += 'Separable fidelity requires a real ordered maskable codomain. ' 
            if setting.data_fid.Lipschitz==np.inf:
                out['info'] += 'Data functional not Lipschitz continuous. '                
        def check_penalty(penalty):
            if isinstance(penalty, FunctionalOnDirectSum):
                return all(check_penalty(f) for f in penalty.funcs)
            if not penalty.is_separable or 'hessian' not in penalty.methods:
                return False
            lower, upper = penalty.dom_l, penalty.dom_u
            def infinite(bound, sign):
                if bound is None:
                    return False
                vals = (bound,) if np.isscalar(bound) else penalty.domain.iter_coefficients(bound)
                return all(np.isneginf(v) if sign < 0 else np.isposinf(v) for v in vals)
            return ((infinite(lower, -1) and infinite(upper, 1))
                    or isinstance(penalty, IntegralFunctionalBase)
                    or SemiSmoothNewton._is_integral_conjugate(penalty))
        if not check_penalty(setting.penalty):
            out['info'] += ('Penalty must have a separable Hessian and support '
                            'extension, componentwise for direct sums. ')
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
        space = self.op.domain
        lower = list(space.iter_coefficients(self.psi_minus))
        upper = list(space.iter_coefficients(self.psi_plus))
        ub, lb = space.empty(), space.empty()
        space.set_coefficients(ub, (u if np.isfinite(u) else l if np.isfinite(l) else 0.
                                    for l, u in zip(lower, upper)))
        space.set_coefficients(lb, (l if np.isfinite(l) else u if np.isfinite(u) else 0.
                                    for l, u in zip(lower, upper)))
        ones = space.ones()
        return self.R.hessian(ub)(ones), self.R.hessian(lb)(ones)
