from math import sqrt

import numpy as np

from regpy.util import Errors
from ..general import RegSolver

__all__ = ["Landweber", "LandweberOnlyDomain"]


class Landweber(RegSolver):
    r"""The Landweber method. Solves the potentially non-linear, ill-posed equation

    .. math::
        F(x) = data,

    where :math:`F` is a Frechet-differentiable operator, by gradient descent applied 
    to :math:`\Vert F(x) - g^\delta\Vert^2`:

    .. math::

        x_{k+1} = x_{k} - \mu F'[x_k]^*(F(x_k)-data)


    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem.
    init : array-like, optional
        The initial guess and starting point of the iteration. If None (default), it is taken from setting.
    data : array-like, default None
        The measured data/right hand side. If None it is taken from setting.
    stepsize : float, optional
        The step length; must be chosen not too large. If omitted, it is guessed from the norm of
        the derivative at the initial guess. Alternatively, if backtracking is used, stepsize defines an
        initial guess for the step length which has to be sufficiently large. If omitted the initial step
        length will be 10**16.
    backtracking : boolean, optional
        Wether or not to use backtracking for finding a sufficient step length. Default: True.
    """

    def __init__(self, setting,init=None,data=None, stepsize=None, backtracking=True, eta = 0.5, op_norm_method = "lanczos",update_setting = True):
        super().__init__(setting)
        if self.op.linear:
            self.log.warning(
                "Using non-linear Landweber with a linear Operator! Consider "
                "using the linear Landweber implementation."
            )

        initial = setting.get_or_update_initial_guess(init, update_setting)
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, initial)
        self.rhs = setting.get_or_update_data(data, update_setting)

        self.y = self.op.codomain.empty()
        _, self.deriv = self.op.linearize(self.x, out=self.y)

        self.backtracking = bool(backtracking)
        if self.backtracking:
            self.stepsize = 10**16 if stepsize is None else stepsize
        else:
            self.stepsize = (
                0.9 / self.deriv.norm(
                    setting.h_domain, setting.h_codomain,
                    method=op_norm_method
                )**2
                if stepsize is None else stepsize
            )
        if self.stepsize <= 0:
            raise ValueError(Errors.value_error(
                "The step size must be positive!"
            ))
        self.eta = eta
        if not 0 < eta < 1:
            raise ValueError(Errors.value_error(
                "The step size reduction constant must be between 0 and 1!"
            ))

        domain, codomain = self.op.domain, self.op.codomain
        self._residual = codomain.empty()
        self._gy_residual = codomain.empty()
        self._update = domain.empty()
        self._trial_x = domain.empty()
        self._scaled_update = domain.empty()

        if self.backtracking:
            self._update_residual()
            self._old_err = codomain.vdot(
                self._residual, self._gy_residual
            ).real

    def _update_residual(self):
        self.op.codomain.copyto(self._residual, self.y)
        self._residual -= self.rhs
        self.h_codomain.gram(
            self._residual, out=self._gy_residual
        )

    def _write_trial(self):
        self.op.domain.copyto(self._trial_x, self.x)
        np.multiply(
            self._update, self.stepsize, out=self._scaled_update
        )
        self._trial_x -= self._scaled_update

    def _next(self):
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

        # Compute the derivative for the next step directly into ``self.y``.
        _, self.deriv = self.op.linearize(self.x, out=self.y)

        if self.log.isEnabledFor(20):
            err = (
                self._old_err if self.backtracking
                else self.op.codomain.vdot(
                    self._residual, self._gy_residual
                ).real
            )
            self.log.info("|residual| = {}".format(sqrt(err)))


class LandweberOnlyDomain(RegSolver):
    r"""Nonlinear Landweber iteration without constructing codomain vectors.

    The ordinary Landweber gradient can be written as

    .. math::

        G_X^{-1}\left(
            F'[x]^*G_YF(x) - F'[x]^*G_Ydata
        \right).

    This variant obtains the two terms directly in the operator domain through
    ``linearize(..., return_adjoint_eval=True)`` and
    ``deriv.adjoint_data(data)``. Consequently, it does not allocate codomain work vectors or store
    an iterate image in ``op.codomain``. The supplied operator must implement optimized
    in-place versions of adjoint evaluation, adjoint data, and adjoint
    derivative that incorporate the codomain Gram matrix.

    Backtracking is deliberately not supported because its acceptance test
    requires evaluating codomain residual norms.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Setting of the nonlinear forward problem.
    init : array-like, optional
        Initial guess. If omitted, it is taken from ``setting``.
    data
        Measured data or another object accepted by
        ``deriv.adjoint_data``. If omitted, ``setting.data`` is used.
        The solver retains this object but creates no codomain work vector.
    stepsize : float, optional
        Positive Landweber step size. If omitted, ``0.9 / norm(deriv)**2`` is
        used, with the derivative norm computed in codomain-free mode.
    op_norm_method : str, optional
        Method used for derivative norm estimation.
    update_setting : bool, optional
        Whether the initial guess should be updated in ``setting``.
        Data are intentionally not copied into the setting by this variant.
    logging_level
        Logging level.
    """

    def __init__(
        self,
        setting,
        init=None,
        data=None,
        stepsize=None,
        op_norm_method="lanczos",
        update_setting=True,
        logging_level="INFO",
    ):
        super().__init__(setting)
        if self.op.linear:
            self.log.warning(
                "Using nonlinear LandweberOnlyDomain with a linear operator. "
                "Consider using the linear domain-only Landweber implementation."
            )
        self.log.setLevel(logging_level)

        initial = setting.get_or_update_initial_guess(init, update_setting)
        domain = self.op.domain

        self.x = domain.empty()
        domain.copyto(self.x, initial)
        self.y = None

        if data is None:
            data = setting.data
        if data is None:
            raise ValueError(Errors.value_error(
                "Data have to be included in the setting or supplied directly."
            ))
        self.data = data

        self._adjoint_eval = domain.empty()
        self._adjoint_data = domain.empty()
        self._normal_residual = domain.empty()
        self._update = domain.empty()

        _, self.deriv = self.op.linearize(
            self.x,
            out=self._adjoint_eval,
            return_adjoint_eval=True,
        )

        if stepsize is None:
            norm = self.deriv.norm(
                setting.h_domain,
                setting.h_codomain,
                method=op_norm_method,
                without_codomain_vectors=True,
            )
            if norm <= 0:
                raise ValueError(Errors.value_error(
                    "The derivative norm must be positive."
                ))
            stepsize = 0.9 / norm**2
        if stepsize <= 0:
            raise ValueError(Errors.value_error(
                "The step size must be positive!"
            ))
        self.stepsize = stepsize

    def primal(self):
        """Return the current domain iterate and no codomain image."""
        return self.x, None

    def _next(self):
        domain = self.op.domain

        self.deriv.adjoint_data(
            self.data,
            out=self._adjoint_data,
        )
        domain.copyto(self._normal_residual, self._adjoint_eval)
        self._normal_residual -= self._adjoint_data

        self.h_domain.gram_inv(
            self._normal_residual,
            out=self._update,
        )

        if self.log.isEnabledFor(20):
            norm_sq = domain.vdot(
                self._normal_residual,
                self._update,
            ).real
            self.log.info(
                "|normal residual| = {}".format(
                    sqrt(max(norm_sq, 0.0))
                )
            )

        self._update *= self.stepsize
        self.x -= self._update

        _, self.deriv = self.op.linearize(
            self.x,
            out=self._adjoint_eval,
            return_adjoint_eval=True,
        )
