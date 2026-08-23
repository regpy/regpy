from math import sqrt
from typing import Any

from regpy.vecsps import Vector
from regpy.util import Errors
from ..solver import RegSolver
from ..setting import Setting, check_Hilbert_applicability_linear

__all__ = ["Landweber"]

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
        if change_init:
            self.x = init  
        else:
            self.x = self.op.domain.empty()
            self.op.domain.copyto(self.x,init)

        if not setting.codomain_free:
            self.y = self.op.codomain.empty()
            self.op(self.x, out=self.y)
            # After the Gram application this stores G_Y(Tx-data).
        else:
            self.y = None
            self.backprop_data = self.op.domain.empty()
            self.op.adjoint_data(self.data,out=self.backprop_data)
        self._update = self.op.domain.empty()

        if stepsize is None:
            norm = setting.op.norm(setting.h_domain,
                                   None if setting.codomain_free else setting.h_codomain,
                                   codomain_free = setting.codomain_free
                                   ) if op_norm is None else op_norm
            self.stepsize = 1 / norm**2
        else:
            self.stepsize = stepsize
        if self.stepsize <= 0:
            raise ValueError(Errors.value_error("stepsize must be positive."))

    check_applicability = staticmethod(check_Hilbert_applicability_linear)

    def _next(self):
        if not self.setting.codomain_free:
            codomain = self.op.codomain

            self.y -= self.data
            self.h_codomain.gram(self.y, out=self.y)

            self.op.adjoint(self.y, out=self._update)

            self.h_domain.gram_inv(self._update, out=self._update)
            self._update *= self.stepsize
            self.x -= self._update

            self.op(self.x, out=self.y)
        else:
            self.op.adjoint_eval(self.x, out=self._update)
            self._update -= self.backprop_data
            self.h_domain.gram_inv(self._update, out=self._update)

            self._update *= self.stepsize
            self.x -= self._update
