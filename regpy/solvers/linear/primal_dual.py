import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util
from regpy.functionals import Functional

class PDHG(Solver):
    r"""The Primal-dual hybrid gradient (PDHG) or Chambolle-Pock Algorithm
    For $\theta=0$ this is the Arrow-Hurwicz-Uzawa algorithm.

    Solves the minimization problem: $\mathcal{S}(Tf)+\alpha*\mathcal{R}(f)$
    by solving the saddle-point problem: 
    $$
        \inf_f \sup_p [ \langle Tf,p\rangle+\alpha\mathcal{R}(f)-\mathcal{S}^\ast(p) ].
    $$
    Here $\mathcal{S}^\ast$ denotes the Fenchel conjugate functional.

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem. The operator needs to be linear.
    data_fidelity_conjugate : regpy.functionals.Functional
        The Fenchel conjugate of the data fidelity functional. Needs to have a prox-operator defined.
    init_domain : array_like
        The initial guess "f".
    init_codomain : array-like
        The initial guess "p". 
    tau : float , optional
        The parameter to compute the proximal operator of the penalty term. Must be positive. Stepsize of the primal step.
    sigma : float , optional
        The parameter to compute the proximal operator of the data-fidelity term. Must be positive. Stepsize of the dual step.
    regpar : float, optional
        The regularization parameter. Must be positive.
    theta : float, optional
        Relaxation parameter. For theta==0 PDHG is the Arrow-Hurwicz-Uzawa algorithm.
    proximal_pars_data_fidelity_conjugate : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the data fidelity functional.
    proximal_pars_penalty : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the penalty functional.
    """
    def __init__(self,  setting, data_fidelity_conjugate, penalty, init_domain, init_codomain, tau = 1, sigma = 1, regpar = 1, theta= 0, proximal_pars_data_fidelity_conjugate = None, proximal_pars_penalty = None):
        super().__init__()
        self.setting = setting
        """Regularization Setting. 
        """
        assert self.setting.op.linear
        self.data_fidelity_conjugate = data_fidelity_conjugate
        """Conjugate functional of data fidelity functional. 
        """
        assert isinstance(self.data_fidelity_conjugate, Functional)

        self.x = init_domain
        self.x_old = self.x
        self.y = self.setting.op(self.x)
        self.p = init_codomain

        self.tau = tau
        self.sigma = sigma
        self.regpar = regpar
        self.theta = theta
        self.proximal_pars_data_fidelity_conjugate = proximal_pars_data_fidelity_conjugate
        self.proximal_pars_penalty = proximal_pars_penalty

    def _next(self):
        primal_step = self.x - self.tau * self.setting.h_domain.gram_inv(self.setting.op.adjoint(self.setting.h_codomain.gram(self.p)))
        self.x = self.setting.penalty.proximal(primal_step, self.regpar * self.tau, self.proximal_pars_penalty)
        dual_step = self.p + self.sigma * self.setting.op( self.x+self.theta*(self.x-self.x_old) )
        self.p = self.data_fidelity_conjugate.proximal(dual_step, self.sigma, self.proximal_pars_data_fidelity_conjugate)
        self.x_old = self.x
        self.y = self.setting.op(self.x)


class DouglasRashford(Solver):
    r"""The Douglas-Rashford Splitting Algorithm

    Minimizes $\mathcal{S}(Tf)+\alpha*\mathcal{R}(f)$

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem, both penalty and data fidelity need prox-operators. The operator needs to be linear.
    init_h : array_like
        The initial guess "f". Must be in setting.op.domain.
    tau : float , optional
        The parameter to compute the proximal operator of the penalty term. Must be positive. (Default: 1)
    regpar : float, optional
        The regularization parameter. Must be positive. (Default: 1)
    proximal_pars_data_fidelity : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the data fidelity functional. (Default: None)
    proximal_pars_penalty : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the penalty functional. (Default: None))
    """
    def __init__(self,  setting, init_h, tau = 1, regpar = 1, proximal_pars_data_fidelity = None, proximal_pars_penalty = None):
        super().__init__()
        self.setting = setting
        """Regularization setting includes both penalty and data fidelity functionals.
        """
        assert init_h in self.setting.op.domain
        self.h = init_h

        self.tau = tau
        self.regpar = regpar
        self.proximal_pars_data_fidelity = proximal_pars_data_fidelity
        self.proximal_pars_penalty = proximal_pars_penalty

        self.x = self.setting.penalty.proximal(self.h, self.tau*self.regpar, self.proximal_pars_penalty)
        self.y = self.setting.op(self.x)

    def _next(self):
        self.h += self.setting.data_fidelity.proximal(2*self.x-self.h, self.tau, self.proximal_pars_data_fidelity) - self.x
        self.x = self.setting.penalty.proximal(self.h, self.tau*self.regpar, self.proximal_pars_penalty)
        self.y = self.setting.op(self.x)