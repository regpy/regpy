from typing import Any

from regpy.util import Errors
from regpy.vecsps import Vector

from ..solver import RegSolver
from ..setting import Setting

__all__ = ["ForwardBackwardSplitting"]


class ForwardBackwardSplitting(RegSolver):
    r"""Minimize :math:`\mathcal{S}(F(x))+\alpha\mathcal{R}(x)` with
    forward-backward splitting.

    Setting ``codomain_free=True`` computes the data-term gradient
    from ``F'[x]^* G_Y F(x)`` and ``F'[x]^* G_Y data`` directly in the
    operator domain. This requires optimized implementations of
    ``Operator.linearize(..., return_adjoint_eval=True)`` and
    ``Derivative.adjoint_data`` that incorporate the codomain Hilbert metric.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Setting of the forward problem.
    init : setting.op.domain, optional
        Initial guess. If omitted, it is taken from ``setting``.
    data : setting.op.codomain or operator-specific data handle, optional
        Measured data. In domain-only mode, the object is passed directly to
        ``Derivative.adjoint_data`` and is not copied into solver-owned
        codomain storage.
    tau : float, optional
        Positive step size. If omitted, the derivative norm at the initial
        guess is used.
    proximal_pars : dict, optional
        Keyword arguments for the penalty proximal mapping.
    update_setting : bool, default: True
        Update the setting with explicitly supplied initial values in the
        setting.

    Notes
    -----
    If ``setting.codomain_free`` is True, the solver avoids vectors in
    ``setting.op.codomain``.
    """

    def __init__(
            self,
            setting: Setting,
            init: Vector | None = None,
            data: Vector | Any | None = None,
            tau: float | None = None,
            proximal_pars: dict | None = None,
            logging_level: int | str = "INFO",
            update_setting: bool = True):
        if not setting.is_tikhonov:
            raise ValueError(Errors.value_error(
                "ForwardBackwardSplitting requires the setting to contain a "
                "regularization parameter!"
            ))
        super().__init__(setting)
        if self.op.linear:
            self.log.warning(
                "Using non-linear ForwardBackwardSplitting with a linear "
                "Operator! Consider using the linear implementation."
            )

        domain, codomain = self.op.domain, self.op.codomain
        initial = setting.get_or_update_initial_guess(init, update_setting)
        self.x = domain.empty()
        domain.copyto(self.x, initial)

        self.codomain_free = setting.codomain_free
        self.proximal_pars = (
            {} if proximal_pars is None else dict(proximal_pars)
        )
        
        self.log.setLevel(logging_level)
        self._gradient = domain.empty()

        if self.codomain_free:
            self.data = setting.get_or_update_data(data, update_setting)
            if self.data is None:
                raise ValueError(Errors.value_error(
                    "ForwardBackwardSplitting without codomain vectors "
                    "requires data either in the setting or as an argument."
                ))

            self._adjoint_eval = domain.empty()
            self._adjoint_data = domain.empty()
            _, self.deriv = self.op.linearize(
                self.x, out=self._adjoint_eval,
                return_adjoint_eval=True
            )

            self.y = None
        else:
            self.data = setting.get_or_update_data(data, update_setting)
            self.y = codomain.empty()
            self._data_gradient = codomain.empty()
            _, self.deriv = self.op.linearize(self.x, out=self.y)

        self.tau = (
            1 / self.deriv.norm(
                setting.h_domain,
                None if self.codomain_free else setting.h_codomain,
                codomain_free=self.codomain_free
            )**2
            if tau is None else tau
        )
        if self.tau <= 0:
            raise ValueError(Errors.value_error(
                "tau, the step size, needs to be positive!"
            ))

    def _next(self):
        if self.codomain_free:
            # ``_adjoint_eval`` contains F'[x]^* G_Y F(x) from the most
            # recent linearization. Subtract F'[x]^* G_Y data in place.
            self.deriv.adjoint_data(
                self.data, out=self._adjoint_data
            )
            self._adjoint_eval -= self._adjoint_data
            self.h_domain.gram_inv(
                self._adjoint_eval, out=self._gradient
            )
        else:
            self.data_fid.subgradient(
                self.y, out=self._data_gradient
            )
            self.deriv.adjoint(
                self._data_gradient, out=self._gradient
            )
            self.h_domain.gram_inv(
                self._gradient, out=self._gradient
            )

        self._gradient *= self.tau
        self.x -= self._gradient
        self.penalty.proximal(
            self.x, self.regpar * self.tau, out=self.x,
            **self.proximal_pars
        )

        if self.codomain_free:
            _, self.deriv = self.op.linearize(
                self.x, out=self._adjoint_eval,
                return_adjoint_eval=True
            )
        else:
            _, self.deriv = self.op.linearize(
                self.x, out=self.y
            )
