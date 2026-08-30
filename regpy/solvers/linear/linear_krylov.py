from typing import Any

from regpy.vecsps import Vector
from regpy.util import Errors
from ..solver import RegSolver
from ..setting import Setting, check_Hilbert_applicability_linear

__all__ = ["Richardson", "NuMethodSelfAdjoint", "NuMethod", "Landweber"]


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
            norm = setting.op.norm(h_domain=setting.h_domain) if op_norm is None else op_norm
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

        
class NuMethodSelfAdjoint(RegSolver):
    r"""The :math:`\nu`-method for a linear self-adjoint operator.

    This class implements the actual :math:`\nu`-method recurrence. In the
    application to :math:`T x=y^\delta`, its operator is
    :math:`A=T^*T` and its right-hand side is :math:`T^*y^\delta`. Thus it
    solves

    .. math::
        A x = data

    for a linear self-adjoint operator :math:`A` by

    .. math::
        x_k = x_{k-1}
        + \theta_k (x_{k-1}-x_{k-2})
        - \omega_k\mu G^{-1}(A x_{k-1}-data),

    where :math:`G` is the Gram operator of the domain Hilbert space and

    .. math::
        \theta_k &={ (k-1)(2k-3)(2k+2\nu-1) \over
        (k+2\nu-1)(2k+4\nu-1)(2k+2\nu-3)},\\
        \omega_k &=4{(2k+2\nu-1)(k+\nu-1) \over
        (k+2\nu-1)(2k+4\nu-1)}.

    For a given initial guess :math:`x_0`, the first iterate is initialized
    separately by

    .. math::
        x_1={4\nu+2\over4\nu+1}\mu
        G^{-1}(data-Ax_0).

    The step size must satisfy :math:`\mu\lVert A\rVert\leq 1`. For the
    normal operator :math:`A=T^*T`, this is the usual condition
    :math:`\mu\lVert T\rVert^2\leq 1`.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Setting whose forward operator is linear and self-adjoint.
    data : Vector or None [default: None]
        Right-hand side. If omitted, it is resolved from the setting.
    init : Vector or None [default: None]
        Initial guess :math:`x_0`. If omitted, it is resolved from the
        setting.
    change_init : bool [default: True]
        If true, ``init`` is used directly as the initial iteration vector.
        Otherwise it is copied.
    nu : float [default: 1.0]
        Parameter :math:`\nu` determining the qualification of the method.
    stepsize : float or None [default: None]
        Step size :math:`\mu`. By default, ``1 / op_norm`` is used.
    op_norm : float or None [default: None]
        Norm of :math:`A`. If omitted, it is estimated numerically. If a step
        size is supplied, the norm is still needed to verify the step-size
        condition.

    """

    def __init__(
        self,
        setting: Setting,
        data: Vector | None = None,
        init: Vector | None = None,
        change_init: bool = True,
        nu: float = 1.0,
        stepsize: float | None = None,
        op_norm: float | None = None,
        logging_level: int | str = "INFO",
    ) -> None:
        out, par = NuMethodSelfAdjoint.check_applicability(
            setting,
            op_norm=op_norm,
            user_data=data,
            user_init=init,
        )
        if not out['applicable']:
            raise ValueError(
                'NuMethodSelfAdjoint not applicable to this setting. '
                + out['info']
            )
        super().__init__(setting)
        self.log.setLevel(logging_level)

        if nu <= 0:
            raise ValueError(Errors.value_error("nu must be positive."))
        self.nu = nu

        self.data = par['data']
        if self.data is None:
            raise ValueError(Errors.value_error(
                "NuMethodSelfAdjoint requires data either in the setting "
                "or as an argument."
            ))
        if self.data not in self.op.codomain:
            raise TypeError(Errors.not_in_vecsp(
                self.data,
                self.op.codomain,
                vec_name='data',
                space_name='codomain',
            ))

        norm = setting.op.norm(h_domain = setting.h_domain) if op_norm is None else op_norm
        if norm <= 0:
            raise ValueError(Errors.value_error("op_norm must be positive."))
        self.stepsize = 1 / norm if stepsize is None else stepsize
        if self.stepsize <= 0:
            raise ValueError(Errors.value_error("stepsize must be positive."))
        if self.stepsize * norm > 1 + 1e-12:
            raise ValueError(Errors.value_error(
                "stepsize must satisfy stepsize * op_norm <= 1."
            ))

        init = par['init']
        if change_init:
            self.x = init
        else:
            self.x = self.op.domain.empty()
            self.op.domain.copyto(self.x, init)

        self.y = self.op.codomain.empty()
        self.op(self.x, out=self.y)
        self._previous = self.op.domain.empty()
        self._next_x = self.op.domain.empty()
        self._momentum = self.op.domain.empty()
        self._update = self.op.domain.empty()

        self.theta = 0.0
        self.omega = (4 * self.nu + 2) / (4 * self.nu + 1)

    @staticmethod
    def check_applicability(
        setting: Setting,
        user_data: Vector | None = None,
        user_init: Vector | None = None,
        op_norm: float | None = None,
        **kwargs: Any,
    ) -> tuple[dict, dict]:
        r"""Check Hilbert applicability and self-adjointness."""
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

    def _coefficients(self, k: int) -> tuple[float, float]:
        if k == 1:
            return 0.0, (4 * self.nu + 2) / (4 * self.nu + 1)
        theta = (
            (k - 1) * (2 * k - 3) * (2 * k + 2 * self.nu - 1)
            / (
                (k + 2 * self.nu - 1)
                * (2 * k + 4 * self.nu - 1)
                * (2 * k + 2 * self.nu - 3)
            )
        )
        omega = (
            4 * (2 * k + 2 * self.nu - 1) * (k + self.nu - 1)
            / ((k + 2 * self.nu - 1) * (2 * k + 4 * self.nu - 1))
        )
        return theta, omega

    def _next(self) -> None:
        self.theta, self.omega = self._coefficients(
            self.iteration_step_nr
        )

        self.op(self.x, out=self._update)
        self._update -= self.data
        self.h_domain.gram_inv(self._update, out=self._update)
        self._update *= self.omega * self.stepsize

        if self.iteration_step_nr == 1:
            # x_1 = x_0+ omega_1 * mu * G^{-1}(data - A x_0).
            self.op.domain.copyto(self._next_x, self.x)
            self._next_x -= self._update
        else:
            self.op.domain.copyto(self._next_x, self.x)
            self.op.domain.copyto(self._momentum, self.x)
            self._momentum -= self._previous
            self._momentum *= self.theta
            self._next_x += self._momentum
            self._next_x -= self._update

        self.op.domain.copyto(self._previous, self.x)
        self.x, self._next_x = self._next_x, self.x
        self.op(self.x, out=self.y)


class NuMethod(RegSolver):
    r"""The :math:`\nu`-method for a bounded linear operator.

    For :math:`T:X\to Y`, this solver applies
    :class:`NuMethodSelfAdjoint` to the normal equation

    .. math::
        T^*T x = T^*y^\delta,

    where ``data`` represents :math:`y^\delta`. The delegated solver therefore
    receives ``setting.op.adjoint_eval`` as :math:`T^*T` and the result of
    ``setting.op.adjoint_data(data)`` as :math:`T^*y^\delta`.

    Thus, the self-adjoint implementation contains the recurrence, while this
    class only constructs the normal operator and backpropagates the data.
    For a given initial guess :math:`x_0`, the first iterate is

    .. math::
        x_1={4\nu+2\over4\nu+1}\mu G_X^{-1}
        T^*(y^\delta-Tx_0).

    The method uses a step size satisfying
    :math:`\mu\lVert T\rVert^2\leq 1`.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Non-Tikhonov Hilbert setting with a linear forward operator.
    data : Vector or Any or None [default: None]
        Right-hand side. For a codomain-free setting, this may be any object
        accepted by ``setting.op.adjoint_data``.
    init : Vector or None [default: None]
        Initial guess :math:`x_0`. If omitted, it is resolved from the
        setting.
    change_init : bool [default: True]
        If true, ``init`` is used directly as the initial iteration vector.
        Otherwise it is copied.
    nu : float [default: 1.0]
        Parameter :math:`\nu` determining the qualification of the method.
    stepsize : float or None [default: None]
        Step size :math:`\mu`. By default, the reciprocal norm of the normal
        operator is used.
    op_norm : float or None [default: None]
        Norm of :math:`T`. If supplied, its square is passed as the norm of
        :math:`T^*T`.

    """

    def __init__(
        self,
        setting: Setting,
        data: Vector | Any | None = None,
        init: Vector | None = None,
        change_init: bool = True,
        nu: float = 1.0,
        stepsize: float | None = None,
        op_norm: float | None = None,
        logging_level: int | str = "INFO",
    ) -> None:
        out, par = NuMethod.check_applicability(
            setting,
            op_norm=op_norm,
            user_data=data,
            user_init=init,
        )
        if not out['applicable']:
            raise ValueError(
                'NuMethod not applicable to this setting. ' + out['info']
            )
        super().__init__(setting)
        self.log.setLevel(logging_level)

        self.data = par['data']
        if self.data is None:
            raise ValueError(Errors.value_error(
                "NuMethod requires data either in the setting or as an "
                "argument."
            ))

        self.backprop_data = self.op.domain.empty()
        self.op.adjoint_data(self.data, out=self.backprop_data)

        # The normal equation is codomain-free: its right-hand side already
        # lives in the domain coordinates used by ``adjoint_eval``.
        self._normal_setting = Setting(
            self.op.adjoint_eval,
            self.penalty,
            self.penalty,
            codomain_free=True,
            explicitDataWarning=False,
            logging_level=logging_level,
        )
        normal_op_norm = None if op_norm is None else op_norm**2
        self._self_adjoint_method = NuMethodSelfAdjoint(
            self._normal_setting,
            data=self.backprop_data,
            init=par['init'],
            change_init=change_init,
            nu=nu,
            stepsize=stepsize,
            op_norm=normal_op_norm,
            logging_level=logging_level,
        )

        self.x = self._self_adjoint_method.x
        self.stepsize = self._self_adjoint_method.stepsize
        self.nu = self._self_adjoint_method.nu
        self.theta = self._self_adjoint_method.theta
        self.omega = self._self_adjoint_method.omega
        if setting.codomain_free:
            self.y = None
        else:
            self.y = self.op.codomain.empty()
            self.op(self.x, out=self.y)

    check_applicability = staticmethod(check_Hilbert_applicability_linear)

    def _next(self) -> None:
        self._self_adjoint_method.x = self.x
        self._self_adjoint_method.stepsize = self.stepsize
        self._self_adjoint_method.iteration_step_nr = self.iteration_step_nr
        self._self_adjoint_method._next()

        self.x = self._self_adjoint_method.x
        self.theta = self._self_adjoint_method.theta
        self.omega = self._self_adjoint_method.omega
        if self.setting.codomain_free:
            self.y = None
        else:
            self.op(self.x, out=self.y)
