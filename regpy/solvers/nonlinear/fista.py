from math import inf, sqrt

import numpy as np

from regpy.util import Errors
from ..general import RegSolver, Setting

__all__ = ["FISTA"]


class FISTA(RegSolver):
    r"""
    The generalized FISTA algorithm for minimization of Tikhonov functionals

    .. math::
        \mathcal{S}_{g^{\delta}}(F(f)) + \alpha \mathcal{R}(f).

    Gradient steps are performed on the first term, and proximal steps on the second term. 
    The step sizes for the gradient steps are determined using a backtracking method introduced in
    A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding algorithm for
    linear inverse problems. SIAM J. Imaging Sci., 2(1):183–202, 2009.
    
    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem. Includes the penalty and data fidelity functionals. 
    init : setting.op.domain [default: setting.op.domain.zeros()]
        The initial guess
    tau : float [default: 10**16]
        Initial step size of minimization procedure. Has to be sufficiently large.
    eta : float [default 0.8]
        Step size reduction constant.
    op_lower_bound : float [default: 0]
        lower bound of the operator: :math:`\|op(f)\|\geq op_lower_bound * \|f\|`\.
        Used to define convexity parameter of data functional.     
    proximal_pars : dict [default: {}]
        Parameter dictionary passed to the computation of the prox-operator for the penalty term. 
    logging_level: [default: logging.INFO]
        logging level
    data : setting.op.codomain or array-like or None [default: None]
        The data to be fitted. If None, the data is taken from the setting.
    without_codomain_vectors : bool [default: False]
        If True, the algorithm is run without codomain vectors. This can be used to save memory if the codomain is very large. 
    update_setting : bool [default: True]   
        If True, the setting is updated with the initial guess and data. This can be used to save memory if the initial guess and data are very large.
    """

    def __init__(
            self, setting: Setting, init=None, tau: float = 10**16,
            eta: float = 0.8, op_lower_bound: float = 0.,
            proximal_pars: dict | None = None, logging_level: str = "INFO",
            data=None, without_codomain_vectors: bool = False,
            update_setting: bool = True):
        if not setting.is_tikhonov:
            raise ValueError(Errors.value_error(
                "FISTA requires the setting to contain a regularization parameter!"
            ))
        super().__init__(setting)

        initial = setting.get_or_update_initial_guess(init, update_setting)
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, initial)
        self.data = (
            data if without_codomain_vectors
            else setting.get_or_update_data(data, update_setting)
        )

        if self.op.linear:
            self.log.warning(
                "Using non-linear FISTA with a linear Operator! Consider using "
                "the linear FISTA in solvers.linear."
            )
        self.log.setLevel(logging_level)
        self.regpar = self.setting.regpar
        self.mu_penalty = self.regpar * self.penalty.convexity_param
        self.mu_data_fidelity = (
            self.data_fid.convexity_param * op_lower_bound**2
        )
        self.mu = self.mu_data_fidelity + self.mu_penalty
        self.proximal_pars = (
            {} if proximal_pars is None else dict(proximal_pars)
        )
        self.without_codomain_vectors = without_codomain_vectors
        self.eta = eta
        if not 0 < eta < 1:
            raise ValueError(Errors.value_error(
                "The step size reduction constant must be between 0 and 1!"
            ))

        domain, codomain = self.op.domain, self.op.codomain
        self.x_old = domain.empty()
        domain.copyto(self.x_old, self.x)
        self._h = domain.empty()
        self._gradient = domain.empty()
        self._domain_work = domain.empty()

        if without_codomain_vectors:
            if self.data_fid.Lipschitz == inf:
                raise ValueError(Errors.value_error(
                    "FISTA without codomain vectors cannot use generic "
                    "backtracking."
                ))
            self._adjoint_eval_h = domain.empty()
            self._adjoint_data = domain.empty()
            _, deriv = self.op.linearize(
                self.x, out=self._adjoint_eval_h,
                return_adjoint_eval=True
            )
            zero = np.zeros((), dtype=getattr(codomain, "dtype", float))
            self.y = np.broadcast_to(zero, codomain.shape)
        else:
            self.y = codomain.empty()
            self._data_gradient = codomain.empty()
            if self.data_fid.Lipschitz != inf:
                _, deriv = self.op.linearize(self.x, out=self.y)
            else:
                self.op(self.x, out=self.y)

        if self.data_fid.Lipschitz != inf:
            self.tau = 1. / (
                deriv.norm(
                    setting.h_domain, setting.h_codomain,
                    without_codomain_vectors=without_codomain_vectors
                )**2 * self.data_fid.Lipschitz
            )
            self.backtracking = False
        else:
            self.tau = tau
            self.backtracking = True
        if self.tau <= 0:
            raise ValueError(Errors.value_error(
                "The step size must be positive!"
            ))

        self.t = self.t_old = 0.
        self.q = self.tau * self.mu / (1 + self.tau * self.mu_penalty)

    def _write_candidate(self):
        """Write ``prox(h - tau * gradient)`` into ``x_old``."""
        self.op.domain.copyto(self.x_old, self._h)
        np.multiply(self._gradient, self.tau, out=self._domain_work)
        self.x_old -= self._domain_work
        self.penalty.proximal(
            self.x_old, self.tau * self.regpar, out=self.x_old,
            **self.proximal_pars
        )

    def _next(self):
        if self.mu == 0:
            self.t = (1 + sqrt(1 + 4 * self.t_old**2)) / 2
            beta = (self.t_old - 1) / self.t
        else:
            self.q = self.tau * self.mu / (
                1 + self.tau * self.mu_penalty
            )
            self.t = (
                1 - self.q * self.t_old**2
                + sqrt((1 - self.q * self.t_old**2)**2 + 4 * self.t_old**2)
            ) / 2
            beta = (
                (self.t_old - 1) / self.t
                * (1 + self.tau * self.mu_penalty
                   - self.t * self.tau * self.mu)
                / (1 - self.tau * self.mu_data_fidelity)
            )

        # h = x + beta * (x - x_old)
        self.op.domain.copyto(self._h, self.x)
        self.op.domain.copyto(self._domain_work, self.x)
        self._domain_work -= self.x_old
        self._domain_work *= beta
        self._h += self._domain_work
        self.t_old = self.t

        if self.without_codomain_vectors:
            _, deriv = self.op.linearize(
                self._h, out=self._adjoint_eval_h,
                return_adjoint_eval=True
            )
            deriv.adjoint_data(self.data, out=self._adjoint_data)
            self._adjoint_eval_h -= self._adjoint_data
            self.h_domain.gram_inv(
                self._adjoint_eval_h, out=self._gradient
            )
        else:
            _, deriv = self.op.linearize(self._h, out=self.y)
            if self.backtracking:
                data_fid_of_h = self.data_fid(self.y)
            self.data_fid.subgradient(self.y, out=self._data_gradient)
            deriv.adjoint(self._data_gradient, out=self._gradient)
            self.h_domain.gram_inv(self._gradient, out=self._gradient)

        self._write_candidate()
        while self.backtracking:
            self.op(self.x_old, out=self.y)
            self.op.domain.copyto(self._domain_work, self.x_old)
            self._domain_work -= self._h
            upper_bound = (
                data_fid_of_h
                + self.setting.h_domain.inner(
                    self._domain_work, self._gradient
                )
                + self.setting.h_domain.inner(
                    self._domain_work, self._domain_work
                ) / (2 * self.tau)
            )
            if self.data_fid(self.y) <= upper_bound:
                break
            self.tau *= self.eta
            self._write_candidate()

        # ``x_old`` now contains the accepted next iterate.
        self.x, self.x_old = self.x_old, self.x
        if not self.without_codomain_vectors and not self.backtracking:
            self.op(self.x, out=self.y)
