import logging
import numpy as np

from regpy.solvers import RegSolver, TikhonovRegularizationSetting
from regpy import util

class PDHG(RegSolver):
    r"""The Primal-Dual Hybrid Gradient (PDHG) or Chambolle-Pock Algorithm
    For \(\theta=0)\ this is the Arrow-Hurwicz-Uzawa algorithm.

    Solves the minimization problem: \(\mathcal{S}_{g^{\delta}}(Tf)+\alpha*\mathcal{R}(f))\
    by solving the saddle-point problem: 
    \[
        \inf_f \sup_p [ \langle Tf,p\rangle+\alpha\mathcal{R}(f)-\mathcal{S}_{g^{\delta}}^\ast(p) ].
    \]
    Here \(\mathcal{S}_{g^{\delta}}^\ast)\ denotes the Fenchel conjugate functional.

    Parameters
    ----------
    setting : regpy.solvers.TikhonovRegularizationSetting
        The setting of the forward problem. The operator needs to be linear.
    init_domain : setting.op.domain
        The initial guess "f".
    init_codomain : setting.op.codomain
        The initial guess "p". 
    tau : float [default: 1]
        The parameter to compute the proximal operator of the penalty term. Must be positive. Stepsize of the primal step.
    sigma : float [default: 1]
        The parameter to compute the proximal operator of the data-fidelity term. Must be positive. Stepsize of the dual step.
    theta : float [default: 1]
        Relaxation parameter. For theta==0 PDHG is the Arrow-Hurwicz-Uzawa algorithm.
    proximal_pars_data_fidelity_conjugate : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the data fidelity functional.
    proximal_pars_penalty : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the penalty functional.
    """
    def __init__(self,  setting, init_domain, init_codomain_star, tau = 1, sigma = 1, 
                 theta= 0, proximal_pars_data_fidelity_conjugate = None, proximal_pars_penalty = None
                 ):
        assert isinstance(setting, TikhonovRegularizationSetting)
        super().__init__(setting)
        assert self.op.linear
        assert init_domain in self.op.domain
        assert init_codomain_star in self.op.codomain

        self.x = init_domain
        self.x_old = self.x
        self.y = self.op(self.x)
        self.pstar = init_codomain_star

        self.tau = tau
        self.sigma = sigma
        self.regpar = setting.regpar
        self.theta = theta
        self.proximal_pars_data_fidelity_conjugate = proximal_pars_data_fidelity_conjugate
        self.proximal_pars_penalty = proximal_pars_penalty

    def _next(self):
        primal_step = self.x - self.tau * self.h_domain.gram_inv(self.op.adjoint(self.h_codomain.gram(self.p)))
        self.x = self.penalty.proximal(primal_step, self.regpar * self.tau, self.proximal_pars_penalty)
        dual_step = self.pstar + self.sigma * self.h_codomain.gram(self.op( self.x+self.theta*(self.x-self.x_old) ))
        self.pstar = self.data_fid.Conj.proximal(dual_step, self.sigma, self.proximal_pars_data_fidelity_conjugate)
        self.x_old = self.x
        self.y = self.op(self.x)


class DouglasRachford(RegSolver):
    r"""The Douglas-Rashford Splitting Algorithm

    Minimizes \(\mathcal{S}(Tf)+\alpha*\mathcal{R}(f)\)

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem, both penalty and data fidelity need prox-operators. The operator needs to be linear.
        And the data_fid term contains the the operator for example `data_fid = HilbertNorm(h_space=L2) * (op - data)`, i.e. it 
        is mapping from the domain of the operator.
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
        super().__init__(setting)
        assert init_h in self.op.domain
        self.h = init_h

        self.tau = tau
        self.regpar = regpar
        self.proximal_pars_data_fidelity = proximal_pars_data_fidelity
        self.proximal_pars_penalty = proximal_pars_penalty

        self.x = self.penalty.proximal(self.h, self.tau*self.regpar, self.proximal_pars_penalty)
        self.y = self.op(self.x)

    def _next(self):
        self.h += self.data_fid.proximal(2*self.x-self.h, self.tau, self.proximal_pars_data_fidelity) - self.x
        self.x = self.penalty.proximal(self.h, self.tau*self.regpar, self.proximal_pars_penalty)
        self.y = self.op(self.x)