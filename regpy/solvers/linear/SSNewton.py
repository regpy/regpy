from math import sqrt,inf
import numpy as np

from regpy.util import Errors
from regpy.operators import CoordinateMask 
from regpy.hilbert import GramHilbertSpace
from regpy.functionals.base import Functional, SquaredNorm, Composed
from regpy.stoprules import CountIterations

from ..general import RegSolver, Setting
from .tikhonov import TikhonovCG
__all__ = ["SemismoothNewton_bilateral","SemismoothNewton_nonneg","SemismoothNewtonAlphaGrid"]


def quadTaylor(func:Functional,x):
    """returns the quadratic Taylor polynomial of a functional at a given point as a SquaredNorm functional.
    The Gram matrix of the Hilbert space underlying the new functional is set to the Hessian of the given functional.
    """
    if not isinstance(func,Functional):
        raise TypeError(Errors.type_error("func must be a Functional",func))
    if not x in func.domain:
        raise ValueError(Errors.not_in_vecsp("x must be in the domain of func",x))
    hess = func.hessian(x)
    grad = func.subgradient(x)
    fx = func(x)
    h_space = GramHilbertSpace(hess)
    b=hess.inverse(grad)-x
    c= fx-np.dot(0.5*x+b,hess(x))
    return SquaredNorm(h_space,a=1.,b=b,c=c)

class SemiSmoothNewton(RegSolver):
    r"""Semi-smooth Newton method for minimizing quadratic Tikhonov functionals

    .. math::
        S_{g^{\delta}}(Tf) + regpar * R(f) 

    subject to bilateral constraints :math:`psi_{minus} \leq f \leq psi_{plus}` 
    for a separable smooth penalty functional :math:`R` and a smooth, strongly convex data functional :math:`S_{g^{\delta}}` 
    and a linear operator :math:`T`.
    
    Parameters
    ----------
    setting: Setting
        The setting. 
    x0: array-like, default: None
        Zeroth iterate. If None, x0=setting.op.doamin.zeros(). x0 must be feasible.
    cg_pars: dictionary, default: None
        Parameters of CG method for minimizing Tikhonov functional on inactive set in each SS Newton step.
    logging_level: Loglevel
        default: logging:INFO
    cg_logging_level: Loglevel
        default: logging.INFO
    """

    def __init__(self, setting,
                 cg_pars = None, logging_level = "INFO", cg_logging_level = "INFO",x0=None,
                 **kwargs
                 ):
        super().__init__(setting)

        out, _ = SemiSmoothNewton.check_applicability(self.setting)
        if not out['applicable']:
            raise RuntimeError('SemismoothNewton_bilateral not applicable to this setting. '+out['info'])
        self.R = self.setting.penalty
        self.S = self.setting.data_fid
        self.psi_plus  = self.R.dom_u
        self.psi_minus = self.R.dom_l
        regpar= setting.regpar
        x0 = x0 if x0 is not None else self.op.domain.zeros()
        if not self.x in self.op.domain:
            raise ValueError(Errors.not_in_vecsp('x0 must be in op.domain',x0))
        if not np.all(x0<=self.psi_plus) or not np.all(x0>=self.psi_minus):
            raise ValueError(Errors.generic_message('x0 not feasible.'))
        y0 = self.op(x0)

        quad_setting = Setting(self.op, quadTaylor(self.R.hessian(x0)),quadTaylor(self.S(y0)),regpar=regpar)

        cg_pars = {'tol': 0.001/sqrt(self.regpar)}
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""

        self.log.setLevel(logging_level)
        self.cg_logging_level = cg_logging_level

        self.b=self.h_domain.gram_inv(self.op.adjoint(self.h_codomain.gram(self.data)))
        if self.xref is not None:
            self.b += self.regpar*self.xref
            
        """Prepare first iteration step"""
        self.lam_plus = self.op.domain.zeros()
        self.lam_minus = self.op.domain.zeros()

        tikhcg=TikhonovCG(quad_setting, x0 = x0, logging_level=self.cg_logging_level, **self.cg_pars)
        self.x, self.y = tikhcg.run()
        cg_its = tikhcg.iteration_step_nr
        
        self.active_plus = self.op.domain.IfPos(self.lam_plus +self.regpar*(self.x-self.psi_plus ))
        self.active_minus = self.op.domain.IfPos(self.lam_minus-self.regpar*(self.x-self.psi_minus))
        if not (self.active_plus).any() and not (self.active_minus).any():
            self.log.info('Stopped at 0th iterate.')
            self.converge()
        self.log.debug('it {}: CG its {}; changes active sets +{},-{}'.format(self.iteration_step_nr,cg_its,
                                                                            (1.*self.active_minus+self.active_plus).sum(),0 )
        )


    def _next(self):
        """compute active and inactive sets, need to be computed in each step again"""
        self.active_plus_old=self.active_plus
        self.active_minus_old=self.active_minus
        self.active  = self.active_plus | self.active_minus
        self.inactive= self.op.domain.logical_not(self.active)

        # On the active sets the solution takes the values of the constraints.
        self.x[self.active_plus]=self.psi_plus[self.active_plus]
        self.x[self.active_minus]=self.psi_minus[self.active_minus]

        # Lagrange parameters are 0 where the corresponding constraints are not active. 
        self.lam_plus[self.inactive]=0
        self.lam_plus[self.active_minus]=0
        self.lam_minus[self.inactive]=0
        self.lam_minus[self.active_plus]=0

        projection = CoordinateMask(self.h_domain.vecsp, self.inactive)
        cg_its = 0
        if self.active.all():
            self.log.info('all indices active!')
        else:
            quad_setting = Setting(self.op * projection, 
                                  Composed(quadTaylor(self.R.hessian(self.x)),projection),
                                  quadTaylor(self.S.hessian(self.y)),
                                  regpar=self.regpar)
            tikhcg = TikhonovCG(quad_setting, x0 = projection(self.x), logging_level=self.cg_logging_level,**self.cg_pars)
            f, _ = tikhcg.run()
            self.x[self.inactive] = f[self.inactive]
            cg_its = tikhcg.iteration_step_nr
        self.y = self.op(self.x)
        z = self.regpar*self.x + self.h_domain.gram_inv(self.op.adjoint(self.h_codomain.gram(self.y)))
        
        self.lam_plus[self.active_plus]  = self.b[self.active_plus] -z[self.active_plus]
        self.lam_minus[self.active_minus]=-self.b[self.active_minus]+z[self.active_minus]

        #Update active and inactive sets
        self.active_plus  = (self.lam_plus +self.regpar*(self.x-self.psi_plus )) >=0 
        self.active_minus = (self.lam_minus-self.regpar*(self.x-self.psi_minus)) >=0
        added_ind = (self.op.domain.logical_and(self.active_plus,  self.op.domain.logical_not(self.active_plus_old ))).sum() \
                  + (self.op.domain.logical_and(self.active_minus, self.op.domain.logical_not(self.active_minus_old))).sum() 
        removed_ind = (self.op.domain.logical_and(self.active_plus_old, self.op.domain.logical_not(self.active_plus))).sum() \
                + (self.op.domain.logical_and(self.active_minus_old, self.op.domain.logical_not(self.active_minus))).sum()
        self.log.info('it {}: CG its {}, changes active sets +{},-{}'.format(self.iteration_step_nr,
                                                                            cg_its,
                                                                            added_ind, removed_ind
                                                                            )
                        )
        if added_ind+removed_ind==0:
            self.converge()

    @staticmethod
    def check_applicability(setting,op_norm=None):
        out = {'info':''}
        if not 'hessian' in setting.data_fid.methods:
            out['info'] += 'Data functional not twice differentiable. '
        elif setting.data_fid.convexity_param==0.:
            out['info'] += 'Data functional not strongly convex. '
        if not setting.penalty.separable:
            out['info'] += 'Penalty term not separable. '
        if not 'hessian' in setting.penalty.methods:
            out['info'] += 'Penalty term not twice differentiable. '
        out['applicable'] = out['info']==''
        out['rate'] = np.nan
        return out, None
