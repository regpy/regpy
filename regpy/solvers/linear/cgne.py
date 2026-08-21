from regpy.util import Errors
from regpy.vecsps import Vector

from ..setting import (
    RegSolver,
    Setting,
    check_Hilbert_applicability_linear,
)

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
    data : Vector or Any or None [default: None]
        Right-hand side in ``setting.op.codomain``. In the default case,
        ``setting.data`` is used.
    init : Vector or None [default: None]
        Initial guess. In the default case, it is taken from the setting.
    change_init : bool [default: False]
        If true, ``init`` itself is used as iteration vector. Otherwise it is
        copied.
    update_setting : bool [default: True]
        If true, explicitly supplied data and initial guess are stored in the
        setting.
    logging_level : int or str [default: "INFO"]
        Logging level.
    """

    def __init__(
        self,
        setting: Setting,
        data: Vector | None = None,
        init: Vector | None = None,
        change_init: bool = False,
        update_setting: bool = True,
        logging_level: int | str = "INFO",
    ) -> None:
        out, _ = self.check_applicability(setting)
        if not out['applicable']:
            raise ValueError('CGNE not applicable to this setting. ' + out['info'])
        super().__init__(setting)
        self.log.setLevel(logging_level)
        data = setting.get_or_update_data(data, update_setting)
        initial = setting.get_or_update_initial_guess(init, update_setting, out=init)

        domain = self.op.domain
        codomain = self.op.codomain

        if change_init:
            self.x = initial
        else:
            self.x = domain.empty()
            domain.copyto(self.x, initial)
        """The current iterate."""

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

    @staticmethod
    def check_applicability(setting: Setting, op_norm: float | None = None) -> tuple[dict, dict]:
        """Check whether CGNE can be used for ``setting``."""
        out, par = check_Hilbert_applicability_linear(setting)
        if setting.without_codomain_vectors:
            out['info'] += (
                'CGNE has no implementation without codomain vectors. '
            )
        out['applicable'] = out['info'] == ''
        return out, par

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
