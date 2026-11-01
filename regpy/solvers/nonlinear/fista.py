from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from regpy.vecsps.base import ReadOnlyVector
from collections.abc import Iterable, Iterator
from math import sqrt
from numbers import Real
from typing import Any

import numpy as np

from ..linear.proximal_gradient import _unregularized_penalty
from regpy.vecsps import Vector
from regpy.util import Errors
from .. import RegSolver, Setting, Naturals

__all__ = ["FISTA"]


class FISTA(RegSolver):
    r"""The generalized FISTA algorithm for minimization of Tikhonov functionals
    
    .. math::
        \mathcal{S}_{g^{\delta}}(F(f)) + \alpha \mathcal{R}(f).
    
    Gradient steps are performed on the first term, and proximal steps on the second term.
    Without a regularization parameter, FISTA minimizes the data-fidelity
    term subject to admissible box constraints. As in the linear FISTA,
    the setting's penalty can be a NullFunctional, a squared-norm-type
    functional, a Quadratic integral functional with optional lower/upper
    bounds, or a FunctionalOnDirectSum of these. Unconstrained penalties
    contribute only the domain Hilbert geometry; for a Quadratic only its
    bounds remain active (the quadratic, linear and constant coefficients
    are discarded). Taylor extensions are not supported in this mode.
    This also covers unregularized problems without constraints. The
    unconstrained case is a regularization method studied by Hubmer and
    Ramlau, Inverse Problems, 2017, under the tangential cone condition.
    
    The step sizes for the gradient steps are determined using a backtracking method introduced in
    A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding algorithm for
    linear inverse problems. SIAM J. Imaging Sci., 2(1):183–202, 2009.
    
    Parameters
    ----------
    setting: Setting
        The setting of the forward problem. Includes the penalty and data
        fidelity functionals. If ``setting.regpar`` is not ``None``, FISTA
        minimizes the Tikhonov functional displayed above. If
        ``setting.regpar`` is ``None``, the penalty may be a squared-norm-type
        functional, :class:`~regpy.functionals.base.NullFunctional`, a
        :class:`~regpy.functionals.integral_functionals.Quadratic` with
        optional box constraints, or a
        :class:`~regpy.functionals.base.FunctionalOnDirectSum` of these.
        The unregularized objective retains only the box constraints; it
        does not contain any quadratic, linear or constant penalty terms.
    data: Vector | Any | None [default: None]
        The data to be fitted. If None, the data is taken from the setting.
    init: ReadOnlyVector | None [default: None]
        The initial guess
    tau: float [default: 10 ** 16]
        Initial step size for backtracking when the data-fidelity gradient has
        no finite Lipschitz constant. Has to be sufficiently large.
    eta: float [default: 0.8]
        Step size reduction constant.
    stepsize_selection_steps: Iterable[int] | None [default: None]
        Iterates at which the derivative norm is reestimated. Norm estimation
        is always performed once at the initial iterate. By default, it is
        repeated at iterations ``1, 2, 4, 8, ...``. Passing ``[0]`` therefore
        estimates the norm only initially. Values must be nonnegative and
        strictly increasing. Whenever reestimation changes the step size, the
        accelerated three-term recursion is restarted. This option is ignored
        when backtracking is used.
    stepsize_safety: float [default: 0.9]
        Safety factor for norm-based step sizes. Must lie in ``(0, 1]`` and9``.
    op_norm_method: str [default: 'lanczos']
        Method used to estimate the derivative norm.
        Norm estimates use ``reltol=0.05 / sqrt(1 + iteration_step_nr)``.
        Thus the initial tolerance is 0.05 and decreases with iteration count.
    op_lower_bound: float [default: 0.0]
        lower bound of the operator: :math:`\|op(f)\|\geq op_lower_bound * \|f\|`\.
        Used to define convexity parameter of data functional.
    proximal_pars: dict | None [default: None]
        Parameter dictionary passed to the computation of the prox-operator for the penalty term.
    logging_level: int | str [default: 'INFO']
        logging level
    update_setting: bool [default: True]
        If True, the setting is updated with the initial guess and data. This can be used to save memory if the initial guess and data are very large.
    
    Notes
    -----
    In the unregularized mode, the original setting penalty still defines
    ``setting.h_domain`` for gradients, norm estimates, and step-size
    selection. The solver uses a separate constraint-only penalty for its
    proximal step; the original setting penalty is not modified.
    
    Whenever norm estimation or backtracking changes the step size, the next
    accelerated step starts afresh from the accepted current iterate. In
    particular, the previous iterate is discarded as a momentum state and the
    acceleration parameters are reset to their initial values.
    """

    def __init__(
            self, 
            setting: Setting, 
            data: Vector | Any | None =None,
            init: ReadOnlyVector | None =None, 
            tau: float = 10**16,
            eta: float = 0.8, 
            stepsize_selection_steps: Iterable[int] | None = None,
            stepsize_safety: float = 0.9,
            op_norm_method: str = "lanczos",
            op_lower_bound: float = 0.,
            proximal_pars: dict | None = None, 
            logging_level: int | str = "INFO",
            update_setting: bool = True
            ) -> None:
        out, _ = self.check_applicability(setting)
        if not out["applicable"]:
            raise ValueError(
                "FISTA not applicable to this setting. " + out["info"]
            )
        super().__init__(setting)
        self.log.setLevel(logging_level)
        if out["warning"]:
            self.log.warning(out["warning"])

        initial = setting.get_or_update_initial_guess(init, update_setting)
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, initial)
        self.data = setting.get_or_update_data(data, update_setting)
        if self.data is None:
            raise ValueError(Errors.value_error(
                "FISTA requires data either in the setting or as an argument."
            ))

        if self.op.linear:
            self.log.warning(
                "Using non-linear FISTA with a linear Operator! Consider using "
                "the linear FISTA in solvers.linear."
            )
        if self.setting.is_tikhonov:
            self.regpar = self.setting.regpar
            self.penalty = self.setting.penalty
        else:
            # Use the same unregularized penalty conversion as linear FISTA:
            # keep box constraints, but discard all other penalty terms.
            # The setting's original penalty retains its Hilbert geometry.
            self.regpar = 1.
            self.penalty = _unregularized_penalty(self.setting.penalty)
        self.codomain_free = setting.codomain_free
        self.mu_penalty = self.regpar * self.penalty.convexity_param
        self.mu_data_fidelity = (
            0. if self.codomain_free
            else self.data_fid.convexity_param * op_lower_bound**2
        )
        self.mu = self.mu_data_fidelity + self.mu_penalty
        self.proximal_pars = (
            {} if proximal_pars is None else dict(proximal_pars)
        )
        self.eta = eta
        if not 0 < eta < 1:
            raise ValueError(Errors.value_error(
                "The step size reduction constant must be between 0 and 1!"
            ))
        if (isinstance(stepsize_safety, bool)
                or not isinstance(stepsize_safety, Real)):
            raise TypeError(Errors.type_error(
                "stepsize_safety must be a real scalar."
            ))
        self.stepsize_safety = float(stepsize_safety)
        if not 0 < self.stepsize_safety <= 1:
            raise ValueError(Errors.value_error(
                "stepsize_safety must lie in (0, 1]."
            ))
        self.op_norm_method = op_norm_method

        domain, codomain = self.op.domain, self.op.codomain
        self.x_old = domain.empty()
        domain.copyto(self.x_old, self.x)
        self._h = domain.empty()
        self._gradient = domain.empty()
        self._domain_work = domain.empty()

        if self.codomain_free:
            self._adjoint_eval_h = domain.empty()
            self._adjoint_data = domain.empty()
            _, self.deriv = self.op.linearize(
                self.x, out=self._adjoint_eval_h,
                return_adjoint_eval=True
            )
            self.y = None
            self.backtracking = False
        else:
            self.y = codomain.empty()
            self._data_gradient = codomain.empty()
            if np.isfinite(self.data_fid.Lipschitz):
                _, self.deriv = self.op.linearize(self.x, out=self.y)
                self.backtracking = False
            else:
                self.op(self.x, out=self.y)
                self.tau = tau
                self.backtracking = True

        self._stepsize_selection_steps: Iterator[int] | None = None
        self._last_stepsize_selection_step = -1
        self._next_stepsize_selection_step: int | None = None
        if not self.backtracking:
            try:
                self._stepsize_selection_steps = iter(
                    (2 ** Naturals())
                    if stepsize_selection_steps is None
                    else stepsize_selection_steps
                )
            except TypeError as error:
                raise TypeError(Errors.type_error(
                    "stepsize_selection_steps must be an iterable of integers."
                )) from error
            self._advance_stepsize_selection_step()
            self._estimate_stepsize()
            if self._next_stepsize_selection_step == 0:
                self._advance_stepsize_selection_step()
        elif stepsize_selection_steps is not None:
            self.log.warning(
                "stepsize_selection_steps is ignored because backtracking "
                "is used."
            )
        if self.tau <= 0 or not np.isfinite(self.tau):
            raise ValueError(Errors.value_error(
                "The step size must be positive and finite!",self.tau
            ))

        self.t = self.t_old = 0.
        self.q = self.tau * self.mu / (1 + self.tau * self.mu_penalty)

    def primal(self) -> tuple[Vector | None, Vector | None]:
        r"""Return the current primal iterate and operator image, when available for monitoring and stopping.
        
        
        Returns
        -------
        tuple[Vector | None, Vector | None]
            Result described above.
        """
        return (self.x, self.y)

    @staticmethod
    def check_applicability(setting: Setting) -> tuple[dict, dict]:
        r"""Check whether ``setting`` provides the operations FISTA needs.
        
        Both the use as minimizer of possibly non-convex Tikhonov functionals 
        with externally chosen regularization parameter and the use as regularization 
        method are possible. 
        
        Parameters
        ----------
        setting: Setting
            Problem setting containing the operator, penalty, and data fidelity.
        
        
        Returns
        -------
        tuple[dict, dict]
            Result described above.
        """
        out = {"info": "", "warning": ""}
        par = {}

        without_regpar = not setting.is_tikhonov
        if without_regpar:
            try:
                # Shared with linear FISTA, including direct sums and
                # rejection of non-constraint Taylor extensions.
                _unregularized_penalty(setting.penalty)
            except ValueError as error:
                out["info"] += str(error) + " "
        if not without_regpar and "proximal" not in setting.penalty.methods:
            out["info"] += "Missing prox of penalty functional. "
        if not setting.codomain_free:
            if "subgradient" not in setting.data_fid.methods:
                out["info"] += "Missing subgradient of data fidelity functional. "
            if not np.isfinite(setting.data_fid.Lipschitz):
                out["warning"] += (
                    "Lipschitz constant of data fidelity functional is not "
                    "finite; backtracking will be used for step-size "
                    "selection."
                )

        out["applicable"] = out["info"] == ""
        return out, par

    def _advance_stepsize_selection_step(self) -> None:
        r"""Read and validate the next norm-reestimation step.
        """
        if self._stepsize_selection_steps is None:
            self._next_stepsize_selection_step = None
            return
        try:
            step = next(self._stepsize_selection_steps)
        except StopIteration:
            self._next_stepsize_selection_step = None
            return
        if (isinstance(step, bool)
                or not isinstance(step, (int, np.integer))):
            raise TypeError(Errors.type_error(
                "stepsize_selection_steps must contain integers."
            ))
        step = int(step)
        if step < 0:
            raise ValueError(Errors.value_error(
                "Step-size selection steps must be nonnegative."
            ))
        if step <= self._last_stepsize_selection_step:
            raise ValueError(Errors.value_error(
                "Step-size selection steps must be strictly increasing."
            ))
        self._last_stepsize_selection_step = step
        self._next_stepsize_selection_step = step

    def _estimate_stepsize(self) -> None:
        r"""Estimate the current derivative norm and update ``tau``.
        """
        norm = self.deriv.norm(
            self.setting.h_domain,
            None if self.codomain_free else self.setting.h_codomain,
            method=self.op_norm_method,
            reltol=0.05 / np.sqrt(1 + self.iteration_step_nr),
            codomain_free=self.codomain_free,
        )
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(Errors.value_error(
                "The derivative norm must be finite and positive."
            ))
        lipschitz = 1. if self.codomain_free else self.data_fid.Lipschitz
        self.tau = self.stepsize_safety / (norm**2 * lipschitz)
        self.log.info(
            f"it. {self.iteration_step_nr}: Step size changed by norm "
            f"estimation to {self.tau}"
        )

    def _update_norm_based_stepsize(self) -> bool:
        r"""Refresh the norm-based step size and report whether it changed.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        step = self._next_stepsize_selection_step
        if step is None:
            return False
        if step < self.iteration_step_nr:
            raise RuntimeError(Errors.runtime_error(
                "A scheduled step-size selection step was skipped."
            ))
        if step == self.iteration_step_nr:
            old_tau = self.tau
            self._estimate_stepsize()
            self._advance_stepsize_selection_step()
            return self.tau != old_tau
        return False

    def _restart_three_term_recursion(self) -> None:
        r"""Discard momentum and restart FISTA at the current iterate.
        """
        self.op.domain.copyto(self.x_old, self.x)
        self.t = self.t_old = 0.
        self.q = self.tau * self.mu / (
            1 + self.tau * self.mu_penalty
        )

    def _write_candidate(self) -> None:
        r"""Write ``prox(h - tau * gradient)`` into ``x_old``.
        """
        self.op.domain.copyto(self.x_old, self._h)
        self.op.domain.copyto(self._domain_work,self._gradient)
        self._domain_work *= self.tau
        self.x_old -= self._domain_work
        self.penalty.proximal(
            self.x_old, self.tau * self.regpar, out=self.x_old,
            **self.proximal_pars
        )

    def _next(self) -> None:
        tau_at_iteration_start = self.tau
        if self.mu == 0:
            self.t = (1 + sqrt(1 + 4 * self.t_old**2)) / 2
            beta = (self.t_old - 1) / self.t
        else:
            self.q = self.tau * self.mu / (
                1 + self.tau * self.mu_penalty
            )
            self.t = (
                1 - self.q * self.t_old**2
                + sqrt((1 - self.q * self.t_old**2)**2 + 4 * self.t_old**2)
            ) / 2
            beta = (
                (self.t_old - 1) / self.t
                * (1 + self.tau * self.mu_penalty
                   - self.t * self.tau * self.mu)
                / (1 - self.tau * self.mu_data_fidelity)
            )

        # h = x + beta * (x - x_old)
        self.op.domain.copyto(self._h, self.x)
        self.op.domain.copyto(self._domain_work, self.x)
        self._domain_work -= self.x_old
        self._domain_work *= beta
        self._h += self._domain_work
        self.t_old = self.t

        if self.codomain_free:
            _, deriv = self.op.linearize(
                self._h, out=self._adjoint_eval_h,
                return_adjoint_eval=True
            )
            deriv.adjoint_data(self.data, out=self._adjoint_data)
            self._adjoint_eval_h -= self._adjoint_data
            self.h_domain.gram_inv(
                self._adjoint_eval_h, out=self._gradient
            )
        else:
            _, deriv = self.op.linearize(self._h, out=self.y)
            if self.backtracking:
                data_fid_of_h = self.data_fid(self.y)
            self.data_fid.subgradient(self.y, out=self._data_gradient)
            deriv.adjoint(self._data_gradient, out=self._gradient)
            self.h_domain.gram_inv(self._gradient, out=self._gradient)

        self._write_candidate()
        while self.backtracking:
            self.op(self.x_old, out=self.y)
            self.op.domain.copyto(self._domain_work, self.x_old)
            self._domain_work -= self._h
            upper_bound = (
                data_fid_of_h
                + self.setting.h_domain.inner(
                    self._domain_work, self._gradient
                )
                + self.setting.h_domain.inner(
                    self._domain_work, self._domain_work
                ) / (2 * self.tau)
            )
            if self.data_fid(self.y) <= upper_bound:
                break
            self.tau *= self.eta
            self._write_candidate()

        # ``x_old`` now contains the accepted next iterate.
        self.x, self.x_old = self.x_old, self.x
        stepsize_changed = self.tau != tau_at_iteration_start
        if self.codomain_free:
            _, self.deriv = self.op.linearize(
                self.x, out=self._adjoint_eval_h,
                return_adjoint_eval=True,
            )
            stepsize_changed |= self._update_norm_based_stepsize()
        elif self.backtracking:
            self.op(self.x, out=self.y)
        else:
            _, self.deriv = self.op.linearize(self.x, out=self.y)
            stepsize_changed |= self._update_norm_based_stepsize()

        if stepsize_changed:
            self._restart_three_term_recursion()
