from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from regpy.vecsps.base import ReadOnlyVector, Scalar, VectorSpaceBase
from regpy.util import Errors
from regpy.vecsps import Vector
from math import sqrt

from ..solver import RegSolver
from ..setting import Setting, check_Hilbert_applicability_linear

__all__ = ["CGNE"]


def _axpy(space: VectorSpaceBase, target: Vector, scalar: Scalar, direction: ReadOnlyVector, work: Vector) -> None:
    r"""Compute ``target += scalar * direction`` using a reusable vector.
    
    Parameters
    ----------
    space: VectorSpaceBase
        Space.
    target: Vector
        Writable vector used by this operation.
    scalar: Scalar
        Scalar.
    direction: ReadOnlyVector
        Input vector; its coefficients are only read.
    work: Vector
        Writable vector used by this operation.
    """
    space.copyto(work, direction)
    work *= scalar
    target += work


class CGNE(RegSolver):
    r"""The conjugate gradient method applied to the normal equation
    :math:`T^*T=T^*g` for solving linear inverse problems :math:`Tf=g`.
    Regularization is achieved by early stopping, typically using the
    discrepancy principle.
    
    Parameters
    ----------
    setting: Setting
        Regularization setting involving Hilbert space norms.
    data: ReadOnlyVector | None [default: None]
        Right-hand side in ``setting.op.codomain``. In the default case,
        ``setting.data`` is used.
    init: Vector | None [default: None]
        Initial guess. If omitted, it is taken from the setting.
        Must be writable when ``change_init=True``; otherwise a copy is used.
    change_init: bool [default: False]
        If true, ``init`` itself is used as iteration vector. Otherwise it is
        copied.
    logging_level: int | str [default: 'INFO']
        Logging level.
    """

    def __init__(
        self,
        setting: Setting,
        data: ReadOnlyVector | None = None,
        init: Vector | None = None,
        change_init: bool = False,
        logging_level: int | str = "INFO",
    ) -> None:
        out, par = self.check_applicability(
            setting, user_data=data, user_init=init
        )
        if not out['applicable']:
            raise ValueError('CGNE not applicable to this setting. ' + out['info'])
        super().__init__(setting)
        self.log.setLevel(logging_level)
        data = par['data']
        initial = par['init']

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
    def check_applicability(
        setting: Setting,
        op_norm: float | None = None,
        user_data: ReadOnlyVector | None = None,
        user_init: ReadOnlyVector | None = None,
    ) -> tuple[dict, dict]:
        r"""Check whether CGNE can be used for ``setting``.
        
        Parameters
        ----------
        setting: Setting
            Problem setting containing the operator, penalty, and data fidelity.
        op_norm: float | None [default: None]
            Operator-norm estimate; None requests numerical estimation.
        user_data: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        user_init: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        tuple[dict, dict]
            Result described above.
        """
        out, par = check_Hilbert_applicability_linear(
            setting,
            op_norm=op_norm,
            user_data=user_data,
            user_init=user_init,
        )
        if setting.codomain_free:
            out['info'] += (
                'CGNE has no implementation without codomain vectors. '
            )
        out['applicable'] = out['info'] == ''
        return out, par

    def _next(self) -> None:
        domain = self.op.domain
        codomain = self.op.codomain

        self.op(self.dir, out=self._Tdir)
        self.h_codomain.gram(self._Tdir, out=self._g_Tdir)

        denominator = codomain.vdot(self._g_Tdir, self._Tdir).real
        if sqrt(denominator) <=1e-15:
            self.log.info(f'CGNE: stopped at iteration {self.iteration_step_nr} as residual is 0 to machine precision.')
            self.converge()
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
