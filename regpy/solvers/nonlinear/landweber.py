from math import sqrt
from typing import Any

import numpy as np

from regpy.util import Errors
from regpy.vecsps import Vector

from ..solver import RegSolver
from ..setting import Setting, check_Hilbert_applicability

__all__ = ["Landweber"]


class Landweber(RegSolver):
    r"""The nonlinear Landweber method for the ill-posed equation

    .. math::
        F(x) = data.

    It applies gradient descent to :math:`\lVert F(x)-data\rVert_Y^2` given by

    .. math::
            x_{k+1} = x_k  -  \mu  G_X^{-1}  F'[x_k]^*  G_Y  (F(x_k) - data).
    
    If ``setting.codomain_free`` is true, this iteration formala is evaluated
    through ``adjoint_eval`` and ``adjoint_data`` and no codomain vector is allocated.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Setting of the inverse problem.
    data : Vector or Any or None [default: None]
        Right-hand side. In the default case, ``setting.data`` is used. For a
        codomain-free setting, it may be any object accepted by
        ``deriv.adjoint_data``.
    init : Vector or None [default: None]
        Initial guess. In the default case, it is taken from the setting.
    stepsize : float or None [default: None]
        Positive step size. Without backtracking, the default is
        ``0.9 / norm(deriv)**2``. With backtracking, the default initial value
        is ``10**16``.
    backtracking : bool [default: True]
        Whether to reduce the step size until the residual decreases.
        Backtracking is disabled for codomain-free settings because its test
        requires a codomain residual.
    eta : float [default: 0.5]
        Step-size reduction factor used by backtracking; must lie in ``(0, 1)``.
    op_norm_method : str [default: "lanczos"]
        Method used to estimate the derivative norm.
    update_setting : bool [default: True]
        If true, explicitly supplied data and initial guess are stored in the
        setting.
    logging_level : int or str [default: "INFO"]
        Logging level.
    """

    def __init__(
        self,
        setting: Setting,
        data: Vector | Any | None = None,
        init: Vector | None = None,
        stepsize: float | None = None,
        backtracking: bool = True,
        eta: float = 0.5,
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
        self.x = par['init'] # use memory of par['init']
        self.data = par['data']

        self.backtracking = bool(backtracking)
        if setting.codomain_free and self.backtracking:
            self.log.warning(
                "Backtracking is unavailable without codomain vectors and "
                "has been disabled."
            )
            self.backtracking = False

        self.eta = eta
        if not 0 < eta < 1:
            raise ValueError(Errors.value_error(
                "The step size reduction constant must be between 0 and 1."
            ))

        self._update = domain.empty()
        if setting.codomain_free:
            self.y = None
            self._adjoint_data = domain.empty()
            _, self.deriv = self.op.linearize(
                self.x,
                out=self._update,
                return_adjoint_eval=True,
            )
        else:
            codomain = self.op.codomain
            self.y = codomain.empty()
            _, self.deriv = self.op.linearize(self.x, out=self.y)
            self._residual = codomain.empty()
            self._gy_residual = codomain.empty()
            self._trial_x = domain.empty()
            self._scaled_update = domain.empty()

        if self.backtracking:
            self.stepsize = 10**16 if stepsize is None else stepsize
        elif stepsize is None:
            norm = self.deriv.norm(
                setting.h_domain,
                None if setting.codomain_free else setting.h_codomain,
                method=op_norm_method,
                codomain_free=setting.codomain_free,
            )
            if norm <= 0:
                raise ValueError(Errors.value_error(
                    "The derivative norm must be positive."
                ))
            self.stepsize = 0.9 / norm**2
        else:
            self.stepsize = stepsize
        if self.stepsize <= 0:
            raise ValueError(Errors.value_error(
                "The step size must be positive."
            ))

        if self.backtracking:
            self._update_residual()
            self._old_err = self.op.codomain.vdot(
                self._residual, self._gy_residual
            ).real

    check_applicability = staticmethod(check_Hilbert_applicability)

    def _update_residual(self) -> None:
        self.op.codomain.copyto(self._residual, self.y)
        self._residual -= self.data
        self.h_codomain.gram(self._residual, out=self._gy_residual)

    def _write_trial(self) -> None:
        self.op.domain.copyto(self._trial_x, self.x)
        np.multiply(self._update, self.stepsize, out=self._scaled_update)
        self._trial_x -= self._scaled_update

    def _next(self) -> None:
        if self.setting.codomain_free:
            self._next_codomain_free()
            return

        if not self.backtracking:
            self._update_residual()

        self.deriv.adjoint(self._gy_residual, out=self._update)
        self.h_domain.gram_inv(self._update, out=self._update)

        if self.backtracking:
            while True:
                self._write_trial()
                self.op(self._trial_x, out=self.y)
                self._update_residual()
                new_err = self.op.codomain.vdot(
                    self._residual, self._gy_residual
                ).real
                if new_err < self._old_err:
                    self._old_err = new_err
                    break
                self.stepsize *= self.eta
            self.x, self._trial_x = self._trial_x, self.x
        else:
            self._update *= self.stepsize
            self.x -= self._update

        _, self.deriv = self.op.linearize(self.x, out=self.y)

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
