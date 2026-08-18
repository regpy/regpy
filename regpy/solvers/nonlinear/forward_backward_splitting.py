import numpy as np

from regpy.util import Errors

from ..setting import RegSolver, Setting

__all__ = ["ForwardBackwardSplitting"]


class ForwardBackwardSplitting(RegSolver):
    r"""Minimize :math:`\mathcal{S}(F(x))+\alpha\mathcal{R}(x)` with
    forward-backward splitting.

    Setting ``without_codomain_vectors=True`` computes the data-term gradient
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
    without_codomain_vectors : bool, default: False
        Avoid solver-owned vectors in ``setting.op.codomain``.
    update_setting : bool, default: True
        Update the setting with explicitly supplied initial values in the
        ordinary mode.
    """

    def __init__(
            self, setting, init=None, data=None, tau=None,
            proximal_pars=None, logging_level="INFO",
            update_setting=True, without_codomain_vectors=False):
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

        self.without_codomain_vectors = bool(without_codomain_vectors)
        self.proximal_pars = (
            {} if proximal_pars is None else dict(proximal_pars)
        )
        
        self.log.setLevel(logging_level)
        self._gradient = domain.empty()

        if self.without_codomain_vectors:
            self.data = data if data is not None else setting.data
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

            # Keep the usual ``y`` attribute without allocating a codomain
            # vector. It is only a compatibility placeholder and must not be
            # interpreted as the current forward image.
            zero = np.zeros((), dtype=getattr(codomain, "dtype", float))
            self.y = np.broadcast_to(zero, codomain.shape)
        else:
            setting.get_or_update_data(data, update_setting)
            self.y = codomain.empty()
            self._data_gradient = codomain.empty()
            _, self.deriv = self.op.linearize(self.x, out=self.y)

        self.tau = (
            1 / self.deriv.norm(
                setting.h_domain,
                setting.h_codomain,
                without_codomain_vectors=self.without_codomain_vectors
            )**2
            if tau is None else tau
        )
        if self.tau <= 0:
            raise ValueError(Errors.value_error(
                "tau, the step size, needs to be positive!"
            ))

    def _next(self):
        if self.without_codomain_vectors:
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

        if self.without_codomain_vectors:
            _, self.deriv = self.op.linearize(
                self.x, out=self._adjoint_eval,
                return_adjoint_eval=True
            )
        else:
            _, self.deriv = self.op.linearize(
                self.x, out=self.y
            )
