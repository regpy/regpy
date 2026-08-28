from typing import Any

from regpy.vecsps import Vector
from regpy.util import Errors
from ..solver import RegSolver
from ..setting import Setting, check_Hilbert_applicability_linear

__all__ = ["Richardson", "Landweber"]


class Richardson(RegSolver):
    r"""Richardson iteration for a linear self-adjoint operator.

    This method solves

    .. math::
        A x = data

    by the iteration

    .. math::
        x_{k+1}=x_k-\mu G^{-1}\left(Ax_k-data\right),

    where :math:`A` must be linear and self-adjoint and :math:`G` is the Gram
    operator of the Hilbert space.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Setting whose forward operator is linear and self-adjoint.
    data : Vector or None [default: None]
        Right-hand side. If omitted, it is resolved from the setting.
    init : Vector or None [default: None]
        Initial guess. If omitted, it is resolved from the setting.
    change_init : bool [default: True]
        If true, ``init`` is used directly as iteration vector. Otherwise it
        is copied.
    stepsize : float or None [default: None]
        Step size :math:`\mu`. By default, ``1 / op_norm`` is used.
    op_norm : float or None [default: None]
        Norm of :math:`A`. If omitted and ``stepsize`` is not supplied, it is
        estimated numerically.
    """

    def __init__(
        self,
        setting: Setting,
        data: Vector | None = None,
        init: Vector | None = None,
        change_init: bool = True,
        stepsize: float | None = None,
        op_norm: float | None = None,
        logging_level: int | str = "INFO",
    ) -> None:
        out, par = Richardson.check_applicability(
            setting,
            op_norm=op_norm,
            user_data=data,
            user_init=init,
        )
        if not out['applicable']:
            raise ValueError(
                'Richardson not applicable to this setting. ' + out['info']
            )
        super().__init__(setting)
        self.log.setLevel(logging_level)

        self.data = par['data']
        if self.data is None:
            raise ValueError(Errors.value_error(
                "Richardson requires data either in the setting or as an "
                "argument."
            ))
        if self.data not in self.op.codomain:
            raise TypeError(Errors.not_in_vecsp(
                self.data,
                self.op.codomain,
                vec_name='data',
                space_name='codomain',
            ))

        init = par['init']
        if change_init:
            self.x = init
        else:
            self.x = self.op.domain.empty()
            self.op.domain.copyto(self.x, init)

        self.y = self.op.codomain.empty()
        self.op(self.x, out=self.y)
        self._update = self.op.domain.empty()

        if stepsize is None:
            norm = (
                setting.op.norm(
                    setting.h_domain,
                    None if setting.codomain_free else setting.h_codomain,
                    codomain_free=setting.codomain_free,
                )
                if op_norm is None else op_norm
            )
            if norm <= 0:
                raise ValueError(Errors.value_error(
                    "op_norm must be positive."
                ))
            self.stepsize = 1 / norm
        else:
            self.stepsize = stepsize
        if self.stepsize <= 0:
            raise ValueError(Errors.value_error("stepsize must be positive."))

    @staticmethod
    def check_applicability(
        setting: Setting,
        user_data: Vector | None = None,
        user_init: Vector | None = None,
        op_norm: float | None = None,
        **kwargs: Any,
    ) -> tuple[dict, dict]:
        r"""Check Hilbert applicability and self-adjointness of the operator."""
        out, par = check_Hilbert_applicability_linear(
            setting,
            user_data=user_data,
            user_init=user_init,
            op_norm=op_norm,
            **kwargs,
        )
        if not getattr(setting.op, 'self_adjoint', False):
            out['info'] += 'Operator is not self-adjoint. '
        data = par.get('data')
        if data is not None and data not in setting.op.codomain:
            out['info'] += 'Data do not belong to the operator codomain. '
        out['applicable'] = out['info'] == ''
        return out, par

    def _next(self) -> None:
        self.op(self.x, out=self._update)
        self._update -= self.data
        self.h_domain.gram_inv(self._update, out=self._update)
        self._update *= self.stepsize
        self.x -= self._update
        self.op(self.x, out=self.y)


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
        Setting of the inverse problem.
    data : Vector or Any or None [default: None]
        Right-hand side. In the default case, ``setting.data`` is used. For a
        codomain-free setting, it may be any object accepted by
        ``setting.op.adjoint_data``.
    init : Vector or None [default: None]
        Initial guess. In the default case, it is taken from the setting. 
    change_init : bool [default: True]
        If true, ``init`` (if provided) is used as iteration vector. Otherwise it is copied. 
    op_norm : float [default: None]
        The operator norm of the forward operator. If None, it is computed numerically unless stepsize is provided
    stepsize : float or None [default: None]
        Step size :math:`\mu`. In the default case ``1 / op_norm**2`` is used.

    Notes
    -----
    For a codomain-free setting, Landweber backpropagates the data once and
    delegates the iteration to :class:`Richardson` with
    ``setting.op.adjoint_eval`` as its self-adjoint forward operator.
    """

    def __init__(self,
        setting:Setting,
        data:Vector | Any | None  = None,
        init:Vector | None  = None,
        change_init:bool = True,
        stepsize:float | None = None,
        op_norm:float | None = None,
        logging_level:int | str = "INFO"
    ) -> None:
        out, par = Landweber.check_applicability(
            setting,
            op_norm=op_norm,
            user_data=data,
            user_init=init,
        )
        if not out['applicable']:
            raise ValueError('Landweber not applicable to this setting. '+out['info'])        
        super().__init__(setting)
        self.log.setLevel(logging_level)

        self.data = par['data']
        init = par['init']
        if setting.codomain_free:
            if self.data is None:
                raise ValueError(Errors.value_error(
                    "Codomain-free Landweber requires data either in the "
                    "setting or as an argument."
                ))
            self.backprop_data = self.op.domain.empty()
            self.op.adjoint_data(self.data, out=self.backprop_data)

            # ``data_fid`` is unused by a codomain-free setting. Reusing the
            # penalty avoids introducing an artificial codomain functional.
            self._richardson_setting = Setting(
                self.op.adjoint_eval,
                self.penalty,
                self.penalty,
                codomain_free=True,
                explicitDataWarning=False,
                logging_level=logging_level,
            )
            richardson_op_norm = None if op_norm is None else op_norm**2
            self._richardson = Richardson(
                self._richardson_setting,
                data=self.backprop_data,
                init=init,
                change_init=change_init,
                stepsize=stepsize,
                op_norm=richardson_op_norm,
                logging_level=logging_level,
            )
            self.x = self._richardson.x
            self.y = None
            self.stepsize = self._richardson.stepsize
            return

        if change_init:
            self.x = init  
        else:
            self.x = self.op.domain.empty()
            self.op.domain.copyto(self.x,init)

        self.y = self.op.codomain.empty()
        self.op(self.x, out=self.y)
        # After the Gram application this stores G_Y(Tx-data).
        self._update = self.op.domain.empty()

        if stepsize is None:
            norm = setting.op.norm(setting.h_domain,
                                   setting.h_codomain,
                                   codomain_free=False
                                   ) if op_norm is None else op_norm
            self.stepsize = 1 / norm**2
        else:
            self.stepsize = stepsize
        if self.stepsize <= 0:
            raise ValueError(Errors.value_error("stepsize must be positive."))

    check_applicability = staticmethod(check_Hilbert_applicability_linear)

    def _next(self):
        if self.setting.codomain_free:
            self._richardson.x = self.x
            self._richardson.stepsize = self.stepsize
            self._richardson._next()
            self.x = self._richardson.x
            self.y = None
            return

        self.y -= self.data
        self.h_codomain.gram(self.y, out=self.y)

        self.op.adjoint(self.y, out=self._update)

        self.h_domain.gram_inv(self._update, out=self._update)
        self._update *= self.stepsize
        self.x -= self._update

        self.op(self.x, out=self.y)
