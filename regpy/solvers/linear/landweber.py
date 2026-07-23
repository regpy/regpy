from math import sqrt

from regpy.util import Errors
from ..general import RegSolver

__all__ = ["Landweber", "LandweberOnlyDomain"]


class Landweber(RegSolver):
    r"""The linear Landweber method. Solves the linear, ill-posed equation

    .. math::
        Tx = data

    by gradient descent for
    :math:`\lVert T(x)-g^\delta\rVert_{Y}^{2}`:

    .. math::
        x_{k+1}
        = x_k-\mu G_X^{-1}T^*G_Y(Tx_k-data).

    The reciprocal of the number of iterations takes the role of the
    regularization parameter.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Setting of the forward problem.
    init : array-like
        Initial guess.
    change_init : bool, optional
        If true, ``init`` itself is used as iteration vector. Otherwise it is
        copied. Defaults to false.
    data : array-like, optional
        Right hand side. Defaults to ``setting.data``.
    stepsize : float, optional
        Step size :math:`\mu`. If omitted, ``1 / norm(T)**2`` is used.
    norm_method
        Method passed to the operator-norm computation.
    """

    def __init__(
        self,
        setting,
        init,
        change_init=False,
        data=None,
        stepsize=None,
        norm_method=None,
        logging_level="INFO",
    ):
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(
                Errors.not_linear_op(
                    self.op,
                    add_info=(
                        "The linear Landweber method requires a linear "
                        "operator. Use the nonlinear Landweber solver instead."
                    ),
                )
            )

        if data is None:
            data = setting.data
        if data is None:
            raise ValueError(
                Errors.value_error(
                    "Data has to be included in setting or given directly."
                )
            )
        if data not in self.op.codomain:
            raise ValueError(
                Errors.not_in_vecsp(
                    data,
                    self.op.codomain,
                    vec_name="data",
                    space_name="codomain",
                )
            )
        if init not in self.op.domain:
            raise ValueError(
                Errors.not_in_vecsp(
                    init,
                    self.op.domain,
                    vec_name="initial guess",
                    space_name="domain",
                )
            )

        self.log.setLevel(logging_level)
        self.data = data
        self.x = init if change_init else init.copy()

        self.y = self.op.codomain.empty()
        self.op(self.x, out=self.y)

        self._update = self.op.domain.empty()
        self._gy_residual = self.op.codomain.empty()
        # After the Gram application this stores G_Y(Tx-data).

        if stepsize is None:
            norm = setting.op.norm(
                setting.h_domain,
                setting.h_codomain,
                method=norm_method,
            )
            self.stepsize = 1 / norm**2
        else:
            self.stepsize = stepsize
        if self.stepsize <= 0:
            raise ValueError(Errors.value_error("stepsize must be positive."))

    def _next(self):
        codomain = self.op.codomain

        codomain.copyto(self._gy_residual, self.y)
        self._gy_residual -= self.data
        self.h_codomain.gram(
            self._gy_residual,
            out=self._gy_residual,
        )
        self.op.adjoint(self._gy_residual, out=self._update)
        self.h_domain.gram_inv(self._update, out=self._update)

        self._update *= self.stepsize
        self.x -= self._update
        self.op(self.x, out=self.y)

        if self.log.isEnabledFor(20):  # logging.INFO
            # Reuse the same buffer to report the norm of the new residual.
            codomain.copyto(self._gy_residual, self.y)
            self._gy_residual -= self.data
            self.h_codomain.gram(
                self._gy_residual,
                out=self._gy_residual,
            )
            norm_sq = (
                codomain.vdot(self.y, self._gy_residual)
                - codomain.vdot(self.data, self._gy_residual)
            ).real
            self.log.info("|residual| = {}".format(sqrt(max(norm_sq, 0.0))))


class LandweberOnlyDomain(RegSolver):
    r"""Landweber iteration without constructing codomain vectors.

    This variant applies the normal operator directly:

    .. math::
        x_{k+1}
        = x_k-\mu G_X^{-1}
          \left(T^*G_YTx_k-T^*G_Ydata\right).

    Therefore ``op.adjoint_eval`` must provide an optimized in-place
    implementation of :math:`T^*G_YT`. In particular, it must incorporate the
    codomain Gram matrix. ``backprop_data`` supplies
    :math:`T^*G_Ydata`.

    Neither initialization nor iteration creates an element of
    ``op.codomain``. Consequently, automatic norm estimation is deliberately
    not performed: provide either ``stepsize`` or ``op_norm``.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Setting with a linear forward operator.
    init : setting.op.domain
        Initial guess.
    backprop_data : setting.op.domain
        The vector :math:`T^*G_Ydata`.
    change_init : bool, optional
        If true, use ``init`` as the iteration vector; otherwise copy it.
    stepsize : float, optional
        Landweber step size. If omitted, ``1 / op_norm**2`` is used.
    op_norm : float, optional
        A supplied norm of ``T``. Required when ``stepsize`` is omitted.
    logging_level
        Logging level.
    """

    def __init__(
        self,
        setting,
        init,
        backprop_data,
        change_init=False,
        stepsize=None,
        op_norm=None,
        logging_level="INFO",
    ):
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(
                Errors.not_linear_op(
                    self.op,
                    add_info=(
                        "LandweberOnlyDomain requires a linear operator."
                    ),
                )
            )
        if init not in self.op.domain:
            raise ValueError(
                Errors.not_in_vecsp(
                    init,
                    self.op.domain,
                    vec_name="initial guess",
                    space_name="domain",
                )
            )
        if backprop_data not in self.op.domain:
            raise ValueError(
                Errors.not_in_vecsp(
                    backprop_data,
                    self.op.domain,
                    vec_name="backpropagated data",
                    space_name="domain",
                )
            )

        if stepsize is None:
            if op_norm is None:
                raise ValueError(
                    Errors.value_error(
                        "Provide stepsize or op_norm. Automatic norm "
                        "estimation would require codomain vectors."
                    )
                )
            if op_norm <= 0:
                raise ValueError(
                    Errors.value_error("op_norm must be positive.")
                )
            stepsize = 1 / op_norm**2
        if stepsize <= 0:
            raise ValueError(Errors.value_error("stepsize must be positive."))

        self.log.setLevel(logging_level)
        self.stepsize = stepsize
        self.backprop_data = backprop_data
        self.x = init if change_init else init.copy()
        self.y = None

        domain = self.op.domain
        self._normal_residual = domain.empty()
        self._update = domain.empty()

    def primal(self):
        """Return the current domain iterate and no codomain image."""
        return self.x, None

    def _next(self):
        domain = self.op.domain

        self.op.adjoint_eval(
            self.x,
            out=self._normal_residual,
        )
        self._normal_residual -= self.backprop_data
        self.h_domain.gram_inv(
            self._normal_residual,
            out=self._update,
        )

        if self.log.isEnabledFor(20):  # logging.INFO
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
