import math as ma
import numpy as np

from regpy.util import Errors
from regpy.functionals import SquaredNorm

from ..setting import RegSolver, Setting

__all__ = ["PDHG","DouglasRachford"]

class PDHG(RegSolver):
    r"""The Primal-Dual Hybrid Gradient (PDHG) or Chambolle-Pock Algorithm
    For :math:`\theta=0` this is the Arrow-Hurwicz-Uzawa algorithm.

    Solves the minimization problem: :math:`\frac{1}{\alpha}\mathcal{S}_{g^{\delta}}(Tf)+\mathcal{R}(f)`
    by solving the saddle-point problem: 
    
    .. math::
        \inf_f \sup_p [ - \langle Tf,p\rangle + \mathcal{R}(f)- \frac{1}{\alpha}\mathcal{S}_{g^{\delta}}^\ast(-\alpha p) ].

    Here :math:`\mathcal{S}_{g^{\delta}}^\ast` denotes the Fenchel conjugate functional.

    Note: Due to a different sign convention for the dual variables, some signs in the iteration formula differ from 
    the original paper and most of the literature.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem. The operator needs to be linear and
        "penalty.proximal" and "data_fid.conj.proximal" need to be implemented.
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
    theta : float [default: -1]
        Relaxation parameter. For theta==0 PDHG is the Arrow-Hurwicz-Uzawa algorithm.
        If -1, a suitable value is selected automatically based on the convexity parameters of penalty and data fidelity term.
    op_norm : float or None [default: None]
        The operator norm of the forward operator. If None, it is computed numerically.
    proximal_pars_data_fidelity_conjugate : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the data fidelity functional.
    proximal_pars_penalty : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the penalty functional.
    compute_y : boolean [True]
        If True, the images y_k=T(x_k) are computed in each iteration. As they are not needed in the algorithm, 
        so this may considerably increase computational costs. If False, None is returned for y_k. 
    """
    def __init__(self,  setting, init_domain=None, init_codomain_star=None, 
                 tau = 0, sigma = 0, theta= -1, op_norm = None, 
                 proximal_pars_data_fidelity_conjugate = None, proximal_pars_penalty = None, 
                 compute_y = True, logging_level = "INFO"
                 ):
        if not setting.is_tikhonov:
            raise ValueError(Errors.value_error("PDHG requires the setting to contain a regularization parameter!"))
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="PDHG requires the operator to be linear!"))
        if init_codomain_star is not None and init_codomain_star not in self.op.codomain:
            raise ValueError(Errors.not_in_vecsp(init_codomain_star,self.op.codomain,vec_name="initial guess p",space_name="codomain"))
        if init_domain is not None and init_domain not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init_domain,self.op.domain,vec_name="initial guess f",space_name="domain"))

        self.log.setLevel(logging_level)

        if init_domain is None and init_codomain_star is None:
            self.x = self.op.domain.zeros()
            self.pstar = self.op.codomain.zeros()
        else:
            self.x = self.op.domain.empty()
            self.pstar = self.op.codomain.empty()
            if init_domain is None:
                self.op.codomain.copyto(self.pstar, init_codomain_star)
                self.op.domain.copyto(
                    self.x, setting.dual_to_primal(self.pstar)
                )
            else:
                self.op.domain.copyto(self.x, init_domain)
                if init_codomain_star is None:
                    self.op.codomain.copyto(
                        self.pstar, setting.primal_to_dual(self.x)
                    )
                else:
                    self.op.codomain.copyto(self.pstar, init_codomain_star)

        # Keep a second primal buffer. It is used as proximal output and then
        # swapped with ``x``, preserving the previous iterate for extrapolation.
        self.x_old = self.op.domain.empty()
        self.op.domain.copyto(self.x_old, self.x)

        out,par = PDHG.check_applicability(setting,op_norm=op_norm,tau=tau,sigma=sigma,theta=theta)
        if out['applicable']:
            self.log.info(out['info'])
        else:
            raise ValueError('PDHG not applicable to this setting. '+out['info'])
        self.tau, self.sigma, self.theta, self.muR, self.muSstar = par['tau'], par['sigma'], par['theta'], par['muR'], par['muSstar']
        self.compute_y = compute_y

        if tau<0 or sigma<0:
            raise ValueError(Errors.value_error("tau and sigma, the stepsize of the primal and dual step need to be non-negative!"))
        self.proximal_pars_data_fidelity_conjugate = (
            {} if proximal_pars_data_fidelity_conjugate is None
            else dict(proximal_pars_data_fidelity_conjugate)
        )
        self.proximal_pars_penalty = (
            {} if proximal_pars_penalty is None
            else dict(proximal_pars_penalty)
        )

        # Persistent buffers used by every PDHG iteration.
        self.Tpstar = self.op.domain.empty()
        self.op.adjoint(self.pstar, out=self.Tpstar)
        self._domain_work = self.op.domain.empty()
        self._primal_step = self.op.domain.empty()
        self._extrapolated = self.op.domain.empty()
        self._op_extrapolated = self.op.codomain.empty()
        self._codomain_gram = self.op.codomain.empty()
        self._dual_step = self.op.codomain.empty()

        if self.compute_y:
            self.y = self.op.codomain.empty()
            self.op(self.x, out=self.y)
            self._primal_y = None
        else:
            self.y = None
            self._primal_y = self.op.codomain.empty()

    @staticmethod
    def check_applicability(setting,op_norm=None,tau=0,sigma=0,theta=-1)->tuple[dict,dict]:
        out = {'info':''}; par = {}
        if not setting.is_tikhonov:
            out['info'] += 'Setting has no regularization parameter. '
        if not setting.is_convex:
            out['info'] += 'Setting is not convex. '
        if 'proximal' not in setting.penalty.methods:
            out['info'] += 'Missing prox of penalty. '
        if 'proximal' not in setting.data_fid.conj.methods:
            out['info'] += 'Missing prox of conjugate data functional.'
        out['applicable'] = out['info'] == ''
        if out['applicable']:
            if sigma>0 or tau>0 or theta>=0:
                if sigma<=0 or tau<=0 or theta<0:
                    raise ValueError(Errors.value_error("If one of tau,sigma,theta is user defined, all three have to be!"))
                out['info'] += 'Using user defined parameters. '
                out['rate']=np.nan
                par = {'tau':tau, 'sigma':sigma, 'theta':theta, 'muR':0,'muSstar':0}
            else:
                L = setting.op.norm(setting.h_domain,setting.h_codomain) if op_norm is None else op_norm  
                if tau==0 and sigma==0:
                    tau = 1/L
                    sigma = 1/L
                elif tau==0 and sigma>0:
                    tau = 1./(L**2*sigma)
                elif sigma==0 and tau>0:
                    sigma = 1./(L**2*tau)

                muR = setting.penalty.convexity_param
                muSstar = setting.regpar/setting.data_fid.Lipschitz
                if muR>0:
                    if muSstar>0:
                        mu = 2*ma.sqrt(muR * muSstar)/L
                        tau = mu/(2.*muR)
                        sigma = mu/(2.*muSstar)
                        theta = 1./(1.+mu)
                        out['rate']=(1.+theta)/(2.+mu)
                        out['info']='Using accelerated version 2 with convexity parameters mu_R={:.3e}, mu_S*={:.3e} and ||T||={:.3e}.\n Expected linear convergence rate: {:.3e}'.format(muR,muSstar,L,out['rate'])
                    else:
                        theta = 0.
                        out['info']='Using accelerated version 1 with convexity parameter mu_R={:.3e} and ||T|={:.3e}. Expected convergence rate O(1/n^2).'.format(muR,L)
                        out['rate']=-2
                else:
                    out['info']='Using unaccelerated version.'
                    out['rate']=np.nan
                    theta =1.
                par = {'tau':tau, 'sigma':sigma, 'theta':theta, 'muR':muR, 'muSstar':muSstar}
        return out, par

    def primal(self):
        if not self.compute_y:
            self.op(self.x, out=self._primal_y)
            return (self.x, self._primal_y)
        return (self.x, self.y)

    def dual(self):
        return (self.pstar,self.Tpstar)

    def _next(self):
        # primal_step = x + tau * G_X^{-1} T^* pstar
        self.h_domain.gram_inv(self.Tpstar, out=self._domain_work)
        self.op.domain.copyto(self._primal_step, self.x)
        self._domain_work *= self.tau
        self._primal_step += self._domain_work

        # Reuse the old primal buffer for the new iterate, then swap.
        self.penalty.proximal(
            self._primal_step, self.tau, out=self.x_old,
            **self.proximal_pars_penalty
        )
        self.x, self.x_old = self.x_old, self.x

        if self.compute_y:
            self.op(self.x, out=self.y)

        if self.theta == 0.:
            if self.compute_y:
                self.h_codomain.gram(self.y, out=self._codomain_gram)
            else:
                self.op(self.x, out=self._op_extrapolated)
                self.h_codomain.gram(
                    self._op_extrapolated, out=self._codomain_gram
                )
        else:
            # extrapolated = x + theta * (x - x_old)
            self.op.domain.copyto(self._extrapolated, self.x)
            self.op.domain.copyto(self._domain_work, self.x)
            self._domain_work -= self.x_old
            self._domain_work *= self.theta
            self._extrapolated += self._domain_work
            self.op(self._extrapolated, out=self._op_extrapolated)
            self.h_codomain.gram(
                self._op_extrapolated, out=self._codomain_gram
            )

        # dual_step = -pstar + sigma * G_Y T extrapolated
        self.op.codomain.copyto(self._dual_step, self.pstar)
        self._dual_step *= -1
        self._codomain_gram *= self.sigma
        self._dual_step += self._codomain_gram
        self._dual_step *= self.regpar

        self.data_fid.conj.proximal(
            self._dual_step, self.regpar*self.sigma, out=self.pstar,
            **self.proximal_pars_data_fidelity_conjugate
        )
        self.pstar *= -1./self.regpar

        if self.muR>0 and self.muSstar==0:
            self.theta = 1./ma.sqrt(1+self.muR*self.tau)
            self.tau *= self.theta
            self.sigma /= self.theta
        self.op.adjoint(self.pstar, out=self.Tpstar)


class DouglasRachford(RegSolver):
    r"""The Douglas-Rashford Splitting Algorithm

    Minimizes :math:`\mathcal{S}(Tf)+\alpha*\mathcal{R}(f)`

    Parameters
    ----------
    setting : regpy.solvers.Setting
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
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="DouglasRachford requires the operator to be linear!"))
        if init_h not in self.op.domain:
            raise ValueError(Errors.value_error('init_h must be in the domain of the operator!'))
        self.h = self.op.domain.empty()
        self.op.domain.copyto(self.h, init_h)
        if setting.is_tikhonov and setting.op.domain != setting.op.codomain:
            if setting.data is None:
                raise ValueError(Errors.value_error('If the regularization parameter is given, the data must be given!'))
            if not isinstance(self.data_fid,SquaredNorm):
                raise ValueError(Errors.value_error('For setting with not matching domains the data_fid must be a SquaredNorm functional!'))
            self.log.info('Using Tikhonov regularization setting. The data fidelity term is reshifted and composed with the operator.')
            self.data_fid_adjusted = setting.data_fid.shift(-setting.data) * (self.op - setting.data)
        elif not setting.is_tikhonov and setting.op.domain != setting.op.codomain:
            raise ValueError(Errors.value_error('If no regularization parameter is given, the operator must be mapping from a space to itself!'))
        else:
            self.data_fid_adjusted = self.data_fid

        self.tau = tau
        self.regpar = regpar
        if self.tau <= 0 or self.regpar <= 0:
            raise ValueError(Errors.value_error(
                "tau and regpar must be positive."
            ))
        self.proximal_pars_data_fidelity = (
            {} if proximal_pars_data_fidelity is None
            else dict(proximal_pars_data_fidelity)
        )
        self.proximal_pars_penalty = (
            {} if proximal_pars_penalty is None
            else dict(proximal_pars_penalty)
        )

        self.x = self.op.domain.empty()
        self.y = self.op.codomain.empty()
        self._reflection = self.op.domain.empty()
        self._data_prox = self.op.domain.empty()

        self.penalty.proximal(
            self.h, self.tau*self.regpar, out=self.x,
            **self.proximal_pars_penalty
        )
        self.op(self.x, out=self.y)

    def _next(self):
        # reflection = 2*x - h
        self.op.domain.copyto(self._reflection, self.x)
        self._reflection *= 2
        self._reflection -= self.h

        self.data_fid_adjusted.proximal(
            self._reflection, self.tau, out=self._data_prox,
            **self.proximal_pars_data_fidelity
        )
        self.h += self._data_prox
        self.h -= self.x

        self.penalty.proximal(
            self.h, self.tau*self.regpar, out=self.x,
            **self.proximal_pars_penalty
        )
        self.op(self.x, out=self.y)
