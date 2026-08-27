from collections.abc import Iterable, Iterator
from numbers import Real
from typing import cast

import numpy as np

from regpy.operators import ObsDataT
from regpy.util import Errors
from regpy.vecsps import ReadOnlyVector, Vector

from ..solver import RegSolver
from ..setting import Setting, check_Hilbert_applicability

__all__ = ["Landweber"]


def _powers_of_two() -> Iterator[int]:
    """Yield ``1, 2, 4, 8, ...`` indefinitely."""
    n = 0
    while True:
        yield 2**n
        n += 1


class Landweber(RegSolver):
    r"""The nonlinear Landweber method for the ill-posed equation

    .. math::
        F(x) = data.

    It applies gradient descent to :math:`\lVert F(x)-data\rVert_Y^2` given by

    .. math::
            x_{k+1} = x_k  -  \mu  G_X^{-1}  F'[x_k]^*  G_Y  (F(x_k) - data).
    
    If ``setting.codomain_free`` is true, this iteration formula is evaluated
    through ``adjoint_eval`` and ``adjoint_data``, and no codomain vector is
    allocated.

    There are three mutually exclusive step-length modes:

    - By default, the derivative norm is estimated initially and again at the
      iterations specified by ``stepsize_selection_steps``.
    - Supplying ``stepsize`` selects a fixed step length.
    - Setting ``backtracking=True`` selects adaptive backtracking. It is
      applied initially and at ``stepsize_selection_steps`` only. Each search
      starts from one initially and from the previous step length afterwards,
      and may both increase and decrease it.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Setting of the inverse problem.
    data : ReadOnlyVector | ObsDataT | None, optional
        Right-hand side. In the default case, ``setting.data`` is used. For a
        codomain-free setting, it may be any object accepted by
        ``deriv.adjoint_data``.
    init : Vector | None, optional
        Initial guess. In the default case, it is taken from the setting.
    stepsize : float | None, optional
        Positive fixed step length. If supplied, ``backtracking`` must be
        ``False`` and derivative-norm estimates are not used.
    backtracking : bool, optional
        Whether to choose the step length by backtracking. Defaults to
        ``False``. Backtracking is unavailable for codomain-free settings
        because its acceptance test requires a codomain residual.
    eta : float, optional
        Geometric step-length factor used by backtracking; must lie in
        ``(0, 1)``. Rejected steps are multiplied by ``eta`` and accepted steps
        are enlarged by division by ``eta``. Defaults to ``0.5``.
    max_backtracking_steps : int, optional
        Maximum number of trial evaluations in one backtracking iteration.
        Defaults to ``100``.
    stepsize_selection_steps : iterable of int | None, optional
        Iterates at which an adaptive step length is selected. Selection is
        always performed once at the initial iterate (iteration zero). By
        default, it is then repeated at iterations ``1, 2, 4, 8, ...``.
        Passing ``[0]`` therefore selects the step length only initially.
        Values must be nonnegative and strictly increasing. In norm-based
        mode, the derivative norm at :math:`x_k` determines the next update.
        In backtracking mode, a search at :math:`x_k` determines the update
        from :math:`x_k` to :math:`x_{k+1}`. This argument cannot be combined
        with a fixed ``stepsize``.
    stepsize_safety : float, optional
        Safety factor for adaptively selected step lengths. In norm-based mode
        the selected value is ``stepsize_safety / norm(deriv)**2``. It is also
        applied to a new step-length scale selected by backtracking. It must
        lie in ``(0, 1]`` and defaults to ``0.9``.
    op_norm_method : str, optional
        Method used to estimate the derivative norm.
        Defaults to ``"lanczos"``.
    logging_level : int | str, optional
        Logging level.
    """

    def __init__(
        self,
        setting: Setting,
        data: ReadOnlyVector | ObsDataT | None = None,
        init: Vector | None = None,
        stepsize: float | None = None,
        backtracking: bool = False,
        eta: float = 0.5,
        max_backtracking_steps: int = 100,
        stepsize_selection_steps: Iterable[int] | None = None,
        stepsize_safety: float = 0.9,
        op_norm_method: str = "lanczos",
        logging_level: int | str = "INFO",
    ) -> None:
        out, par = self.check_applicability(setting, user_data=data, user_init=init)
        if not out['applicable']:
            raise ValueError(
                'Nonlinear Landweber not applicable to this setting. '
                + out['info']
            )
        super().__init__(setting)
        self.log.setLevel(logging_level)
        if self.op.linear:
            self.log.warning(
                "Using nonlinear Landweber with a linear operator. Consider "
                "using the linear Landweber implementation."
            )

        domain = self.op.domain
        self.x: Vector = par['init']  # use memory of par['init']
        self.data: ReadOnlyVector | ObsDataT = par['data']

        if not isinstance(backtracking, (bool, np.bool_)):
            raise TypeError(Errors.type_error(
                "backtracking must be boolean."
            ))
        self.backtracking: bool = bool(backtracking)
        if self.backtracking and stepsize is not None:
            raise ValueError(Errors.value_error(
                "stepsize and backtracking are mutually exclusive."
            ))
        if setting.codomain_free and self.backtracking:
            self.log.warning(
                "Backtracking is unavailable without codomain vectors and "
                "has been disabled."
            )
            self.backtracking = False

        if isinstance(eta, bool) or not isinstance(eta, Real):
            raise TypeError(Errors.type_error(
                "The backtracking factor eta must be a real scalar."
            ))
        self.eta: float = float(eta)
        if not 0 < self.eta < 1:
            raise ValueError(Errors.value_error(
                "The backtracking factor eta must be between 0 and 1."
            ))
        if (isinstance(max_backtracking_steps, bool)
                or not isinstance(max_backtracking_steps, int)):
            raise TypeError(Errors.type_error(
                "max_backtracking_steps must be an integer."
            ))
        if max_backtracking_steps <= 0:
            raise ValueError(Errors.value_error(
                "max_backtracking_steps must be positive."
            ))
        self.max_backtracking_steps: int = max_backtracking_steps

        if (isinstance(stepsize_safety, bool)
                or not isinstance(stepsize_safety, Real)):
            raise TypeError(Errors.type_error(
                "stepsize_safety must be a real scalar."
            ))
        self.stepsize_safety: float = float(stepsize_safety)
        if not 0 < self.stepsize_safety <= 1:
            raise ValueError(Errors.value_error(
                "stepsize_safety must lie in (0, 1]."
            ))
        self.op_norm_method: str = op_norm_method

        self._update: Vector = domain.empty()
        if setting.codomain_free:
            self.y = None
            self._adjoint_data: Vector = domain.empty()
            _, self.deriv = self.op.linearize(
                self.x,
                out=self._update,
                return_adjoint_eval=True,
            )
        else:
            codomain = self.op.codomain
            self.y = codomain.empty()
            _, self.deriv = self.op.linearize(self.x, out=self.y)
            self._residual: Vector = codomain.empty()
            self._gy_residual: Vector = codomain.empty()
            if self.backtracking:
                self._trial_x: Vector = domain.empty()
                self._scaled_update: Vector = domain.empty()

        self._stepsize_selection_steps: Iterator[int] | None = None
        self._last_stepsize_selection_step: int = -1
        self._next_stepsize_selection_step: int | None = None
        self.stepsize: float

        if stepsize is None:
            try:
                self._stepsize_selection_steps = iter(
                    _powers_of_two()
                    if stepsize_selection_steps is None
                    else stepsize_selection_steps
                )
            except TypeError as error:
                raise TypeError(Errors.type_error(
                    "stepsize_selection_steps must be an iterable of "
                    "integers."
                )) from error
            self._advance_stepsize_selection_step()
            if self.backtracking:
                self.stepsize = 1.
                self._backtracking_initialized: bool = False
                self._old_err: float = 0.
            else:
                self._estimate_stepsize()
            if self._next_stepsize_selection_step == 0:
                self._advance_stepsize_selection_step()
        else:
            if stepsize_selection_steps is not None:
                raise ValueError(Errors.value_error(
                    "stepsize_selection_steps cannot be used with a fixed "
                    "stepsize."
                ))
            if isinstance(stepsize, bool) or not isinstance(stepsize, Real):
                raise TypeError(Errors.type_error(
                    "stepsize must be a real scalar."
                ))
            self.stepsize = float(stepsize)
            if not np.isfinite(self.stepsize) or self.stepsize <= 0:
                raise ValueError(Errors.value_error(
                    "The step length must be finite and positive."
                ))

    check_applicability = staticmethod(check_Hilbert_applicability)

    def _advance_stepsize_selection_step(self) -> None:
        """Read and validate the next step-length selection step."""
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
                "Step-length selection steps must be nonnegative."
            ))
        if step <= self._last_stepsize_selection_step:
            raise ValueError(Errors.value_error(
                "Step-length selection steps must be strictly increasing."
            ))
        self._last_stepsize_selection_step = step
        self._next_stepsize_selection_step = step

    def _estimate_stepsize(self) -> None:
        """Estimate the derivative norm and update the step length."""
        norm = self.deriv.norm(
            self.setting.h_domain,
            None if self.setting.codomain_free else self.setting.h_codomain,
            method=self.op_norm_method,
            codomain_free=self.setting.codomain_free,
        )
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(Errors.value_error(
                "The derivative norm must be finite and positive."
            ))
        self.stepsize = self.stepsize_safety / norm**2
        self.log.info(f'it. {self.iteration_step_nr:}:stepsize change by norm estimation to {self.stepsize}')

    def _update_norm_based_stepsize(self) -> None:
        """Refresh a norm-based step length at the current iteration."""
        step = self._next_stepsize_selection_step
        if step is None:
            return
        if step < self.iteration_step_nr:
            raise RuntimeError(Errors.runtime_error(
                "A scheduled step-length selection step was skipped."
            ))
        if step == self.iteration_step_nr:
            self._estimate_stepsize()
            self._advance_stepsize_selection_step()

    def _backtracking_is_scheduled(self) -> bool:
        """Return whether backtracking is due at the current iterate."""
        iterate = self.iteration_step_nr - 1
        if iterate == 0 and not self._backtracking_initialized:
            return True
        step = self._next_stepsize_selection_step
        if step is None:
            return False
        if step < iterate:
            raise RuntimeError(Errors.runtime_error(
                "A scheduled step-length selection step was skipped."
            ))
        return step == iterate

    def _update_residual(self) -> None:
        self.op.codomain.copyto(self._residual, self.y)
        self._residual -= cast(ReadOnlyVector, self.data)
        self.h_codomain.gram(self._residual, out=self._gy_residual)

    def _write_trial(self) -> None:
        self.op.domain.copyto(self._trial_x, self.x)
        self.op.domain.copyto(self._scaled_update, self._update)
        self._scaled_update *= self.stepsize
        self._trial_x -= self._scaled_update

    def _backtracking_step(self) -> None:
        """Choose and apply a decreasing-residual trial step."""
        initial_stepsize = self.stepsize
        trial_stepsize = initial_stepsize
        accepted_stepsize: float | None = None
        decreasing = False

        for _ in range(self.max_backtracking_steps):
            if not np.isfinite(trial_stepsize):
                break
            self.stepsize = trial_stepsize
            self._write_trial()
            self.op(self._trial_x, out=self.y)
            self._update_residual()
            new_error = float(self.op.codomain.vdot(
                self._residual, self._gy_residual
            ).real)

            if np.isfinite(new_error) and new_error < self._old_err:
                accepted_stepsize = trial_stepsize
                if decreasing:
                    break
                trial_stepsize /= self.eta
            elif accepted_stepsize is not None:
                break
            else:
                decreasing = True
                trial_stepsize *= self.eta

        if accepted_stepsize is None:
            raise RuntimeError(Errors.runtime_error(
                "Backtracking failed to find a decreasing step within "
                f"{self.max_backtracking_steps} trial evaluations."
            ))

        if not self._backtracking_initialized:
            self.stepsize = self.stepsize_safety * accepted_stepsize
        elif accepted_stepsize < initial_stepsize:
            self.stepsize = self.stepsize_safety * accepted_stepsize
        elif accepted_stepsize > initial_stepsize:
            self.stepsize = max(
                initial_stepsize,
                self.stepsize_safety * accepted_stepsize,
            )
        else:
            # The previous step was already safety-adjusted.
            self.stepsize = initial_stepsize
        self.log.info(f'it. {self.iteration_step_nr}: Changed stepsize to {self.stepsize} by backtracking.')

        self._write_trial()
        self.x, self._trial_x = self._trial_x, self.x
        self._backtracking_initialized = True

    def _next(self) -> None:
        if self.setting.codomain_free:
            self._next_codomain_free()
            return

        self._update_residual()
        if self.backtracking:
            self._old_err = float(self.op.codomain.vdot(
                self._residual, self._gy_residual
            ).real)

        self.deriv.adjoint(self._gy_residual, out=self._update)
        self.h_domain.gram_inv(self._update, out=self._update)

        if self.backtracking and self._backtracking_is_scheduled():
            self._backtracking_step()
            if (self._next_stepsize_selection_step
                    == self.iteration_step_nr - 1):
                self._advance_stepsize_selection_step()
        else:
            self._update *= self.stepsize
            self.x -= self._update

        _, self.deriv = self.op.linearize(self.x, out=self.y)
        if not self.backtracking:
            self._update_norm_based_stepsize()

    def _next_codomain_free(self) -> None:
        self.deriv.adjoint_data(self.data, out=self._adjoint_data)
        self._update -= self._adjoint_data
        self.h_domain.gram_inv(self._update, out=self._update)

        self._update *= self.stepsize
        self.x -= self._update
        _, self.deriv = self.op.linearize(
            self.x,
            out=self._update,
            return_adjoint_eval=True,
        )
        self._update_norm_based_stepsize()
