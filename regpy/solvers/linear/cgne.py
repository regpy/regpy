from regpy.util import Errors

from ..general import RegSolver

__all__ = ["CGNE"]


def _axpy(space, target, scalar, direction, work):
    """Compute ``target += scalar * direction`` using a reusable vector."""
    space.copyto(work, direction)
    work *= scalar
    target += work


class CGNE(RegSolver):
    r"""
    The conjugate gradient method applied to the normal equation
    :math:`T^*T=T^*g` for solving linear inverse problems :math:`Tf=g`.
    Regularization is achieved by early stopping, typically using the
    discrepancy principle.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Regularization setting involving Hilbert space norms.
    data : array-like, optional
        Right-hand side ``g``. If omitted, the data is taken from ``setting``.
    x0 : array-like, optional
        Initial iterate. Defaults to zero.
    logging_level
        Controls the amount of output.
    """

    def __init__(self, setting, data=None, x0=None, logging_level="INFO"):
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(
                Errors.not_linear_op(
                    self.op,
                    add_info="CGNE requires the operator to be linear!",
                )
            )

        if data is None:
            if setting.data is None:
                raise ValueError(
                    Errors.value_error(
                        "Data has to be included in setting or given directly."
                    )
                )
            data = setting.data
        if data not in self.op.codomain:
            raise ValueError(
                Errors.not_in_vecsp(
                    data,
                    self.op.codomain,
                    vec_name="data",
                    space_name="codomain",
                )
            )
        if x0 is not None and x0 not in self.op.domain:
            raise ValueError(
                Errors.not_in_vecsp(
                    x0,
                    self.op.domain,
                    vec_name="first iteration",
                    space_name="domain",
                )
            )

        self.log.setLevel(logging_level)
        self.x0 = x0
        r"""The zero-th CG iterate. ``x0=None`` corresponds to zero."""

        domain = self.op.domain
        codomain = self.op.codomain

        self.x = domain.zeros() if x0 is None else x0.copy()
        """The current iterate."""

        if x0 is None:
            self.y = codomain.zeros()
        else:
            self.y = codomain.empty()
            self.op(self.x, out=self.y)
        """The image of the current iterate under the operator."""

        # Persistent work vectors used by initialization and every CG step.
        self._residual = codomain.empty()
        self._g_residual = codomain.empty()
        self._Tdir = codomain.empty()
        self._g_Tdir = codomain.empty()

        self._domain_work = domain.empty()
        self._res = domain.empty()
        self.g_res = domain.empty()
        self.dir = domain.empty()
        self.g_dir = domain.empty()

        codomain.copyto(self._residual, data)
        self._residual -= self.y
        self.h_codomain.gram(self._residual, out=self._g_residual)
        self.op.adjoint(self._g_residual, out=self.g_res)
        r"""The Gram matrix applied to the normal-equation residual.

        In each iteration,
        :math:`g_{res}=T^*G_Y(data-Tx)`.
        """

        self.h_domain.gram_inv(self.g_res, out=self._res)
        self.sq_norm_res = domain.vdot(self.g_res, self._res).real
        """The squared norm of the residual."""

        domain.copyto(self.dir, self._res)
        """The direction of descent."""
        domain.copyto(self.g_dir, self.g_res)
        """The domain Gram matrix applied to the search direction."""

    def _next(self):
        domain = self.op.domain
        codomain = self.op.codomain

        self.op(self.dir, out=self._Tdir)
        self.h_codomain.gram(self._Tdir, out=self._g_Tdir)

        denominator = codomain.vdot(self._g_Tdir, self._Tdir).real
        if denominator == 0:
            raise RuntimeError(
                "The CGNE step size is undefined because the search direction "
                "has zero image norm."
            )
        alpha = self.sq_norm_res / denominator

        _axpy(domain, self.x, alpha, self.dir, self._domain_work)
        _axpy(codomain, self.y, alpha, self._Tdir, self._residual)

        self.op.adjoint(self._g_Tdir, out=self._domain_work)
        _axpy(domain, self.g_res, -alpha, self._domain_work, self._res)

        self.h_domain.gram_inv(self.g_res, out=self._res)

        sq_norm_res_old = self.sq_norm_res
        self.sq_norm_res = domain.vdot(self.g_res, self._res).real
        beta = self.sq_norm_res / sq_norm_res_old

        self.dir *= beta
        self.dir += self._res
        self.g_dir *= beta
        self.g_dir += self.g_res
