import numpy as np

from regpy.solvers import RegSolver, TikhonovRegularizationSetting
from regpy import util
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

class PDHG(RegSolver):
    r"""The Primal-Dual Hybrid Gradient (PDHG) or Chambolle-Pock Algorithm
    For \(\theta=0)\ this is the Arrow-Hurwicz-Uzawa algorithm.

    Solves the minimization problem: \(\frac{1}{\alpha}\mathcal{S}_{g^{\delta}}(Tf)+\mathcal{R}(f))\
    by solving the saddle-point problem: 
    
    .. math::
        \inf_f \sup_p [ - \langle Tf,p\rangle + \mathcal{R}(f)- \frac{1}{\alpha}\mathcal{S}_{g^{\delta}}^\ast(-\alpha p) ].

    Here \(\mathcal{S}_{g^{\delta}}^\ast)\ denotes the Fenchel conjugate functional.

    Note: Due to a different sign convention for the dual variables, some signs in the iteration formula differ from 
    the originial paper and most of the literature.

    Parameters
    ----------
    setting : regpy.solvers.TikhonovRegularizationSetting
        The setting of the forward problem. The operator needs to be linear.
    init_domain : setting.op.domain [default: None]
        The initial guess "f". If None, it will be either initialized by 0 or using the optimality conditions in case 
        init_codomain_star is given.
    init_codomain_star : setting.op.codomain [default: None]
        The initial guess "p". Initialization is analogous to init_domain. 
    tau : float [default: 0]
        The parameter to compute the proximal operator of the penalty term. Stepsize of the primal step.
        Must be non-negative. If 0, a positive value is selected automatically based on the operator norm and the value of sigma.
    sigma : float [default: 0]
        The parameter to compute the proximal operator of the data-fidelity term. Stepsize of the dual step.
        Must be non-negative. If 0, a positive value is selected automatically based on the operator norm and the value of tau
    theta : float [default: 1]
        Relaxation parameter. For theta==0 PDHG is the Arrow-Hurwicz-Uzawa algorithm.
    proximal_pars_data_fidelity_conjugate : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the data fidelity functional.
    proximal_pars_penalty : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the penalty functional.
    compute_y : boolean [True]
        If True, the images y_k=T(x_k) are computed in each iteration. As they are not needed in the algorithm, 
        so this may considerably increase computational costs. If False, None is returned for y_k. 
    """
    def __init__(self,  setting, init_domain=None, init_codomain_star=None, tau = 0, sigma = 0, 
                 theta= 1, proximal_pars_data_fidelity_conjugate = None, proximal_pars_penalty = None, 
                 compute_y = True, logging_level = logging.INFO
                 ):
        assert isinstance(setting, TikhonovRegularizationSetting)
        super().__init__(setting)
        assert self.op.linear
        assert init_domain is None or init_domain in self.op.domain
        assert init_codomain_star is None or init_codomain_star in self.op.codomain
        self.log.setLevel(logging_level)

        if init_domain is None:
            if init_codomain_star is None:
                self.x = setting.op.domain.zeros()
                self.pstar = setting.op.codomain.zeros()
            else:
                self.pstar = init_codomain_star
                self.x = setting.dualToPrimal(self.pstar)
        else:
            self.x = init_domain
            if init_codomain_star is None:
                self.pstar = setting.primalToDual(self.x)
            else:
                self.pstar = init_codomain_star

        self.x_old = self.x
        self.compute_y = compute_y
        self.y = self.op(self.x) if self.compute_y else None

        assert tau>=0 and sigma>=0
        L = self.setting.op_norm()   
        if tau==0 and sigma==0:
            self.tau = 1/L
            self.sigma = 1/L
        elif tau==0 and sigma>0:
            self.tau = 1./(L**2*self.sigma)
            self.sigma = sigma
        elif sigma==0 and tau>0:
            self.sigma = 1./(L**2*self.tau)
            self.tau = tau
        else:
            self.sigma = sigma
            self.tau = tau

        self.muR = setting.penalty.convexity_param
        self.muSstar = self.regpar/setting.data_fid.Lipschitz
        if self.muR>0:
            if self.muSstar>0:
                self.mu = 2*np.sqrt(self.muR * self.muSstar)/L
                self.tau = self.mu/(2.*self.muR)
                self.sigma = self.mu/(2.*self.muSstar)
                self.theta = 1./(1.+self.mu)
                self.log.info('Using accelerated version 2 with convexity parameters mu_R={:.3e}, mu_S*={:.3e} and ||T||={:.3e}.\n Expected linear convergence rate: {:.3e}'.format(self.muR,self.muSstar,L,(1.+self.theta)/(2.+self.mu)))
            else:
                self.theta = 0
                self.log.info('Using accelerated version 1 with convexity parameter mu_R={:.3e} and ||T|={:.3e}. Expected convergence rate O(1/n^2).'.format(self.muR,L))                
        else:
            self.theta = theta
            self.log.info('Using unaccelerated version')            
        self.proximal_pars_data_fidelity_conjugate = proximal_pars_data_fidelity_conjugate
        self.proximal_pars_penalty = proximal_pars_penalty
        self.gap = self.setting.dualityGap(primal=self.x)   

    def _next(self):
        primal_step = self.x + self.tau * self.h_domain.gram_inv(self.op.adjoint(self.pstar))
        self.x = self.penalty.proximal(primal_step, self.tau, self.proximal_pars_penalty)
        self.y = self.op(self.x) if self.compute_y else None

        dual_step = -self.pstar + self.sigma * self.h_codomain.gram(self.op( self.x+self.theta*(self.x-self.x_old) ))
        self.pstar = (-1./self.regpar)*self.data_fid.conj.proximal(self.regpar*dual_step, self.regpar*self.sigma, self.proximal_pars_data_fidelity_conjugate)
        self.x_old = self.x        
        if self.muR>0 and self.muSstar==0:
            self.theta = 1./np.sqrt(1+self.muR*self.tau)
            self.tau *= self.theta
            self.sigma /= self.theta
        self.gap = self.setting.dualityGap(primal=self.x,dual= self.pstar) 
 



class DouglasRachford(RegSolver):
    r"""The Douglas-Rashford Splitting Algorithm

    Minimizes :math:`\mathcal{S}(Tf)+\alpha*\mathcal{R}(f)`

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