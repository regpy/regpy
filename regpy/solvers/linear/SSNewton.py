from math import sqrt,inf
import numpy as np
import logging as log

from regpy.util import Errors
from regpy.operators import CoordinateMask
from regpy.operators.base import Composition
from regpy.hilbert import GramHilbertSpace
from regpy.functionals.base import Functional, SquaredNorm, Composed
from regpy.stoprules import CountIterations

from ..solver import RegSolver
from ..setting import Setting
from .tikhonov import TikhonovCG
__all__ = ["SemiSmoothNewton"]

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
    c= fx-np.vdot(0.5*x+b,hess(x))
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
    x0: Vector, default: None
        Zeroth iterate. If None, x0=setting.op.doamin.zeros(). x0 must be feasible.
    cg_pars: dictionary, default: None
        Parameters of CG method for minimizing Tikhonov functional on inactive set in each SS Newton step.
    logging_level: Loglevel
        default: logging:INFO
    cg_logging_level: Loglevel
        default: logging.INFO
    """

    def __init__(self, setting,
                 cg_pars = None, logging_level = "INFO", cg_logging_level = "WARNING",x0=None,
                 psi_plus = None, psi_minus=None,
                 **kwargs
                 ):
        super().__init__(setting)

        out, _ = SemiSmoothNewton.check_applicability(self.setting)
        if not out['applicable']:
            raise RuntimeError('SemismoothNewton_bilateral not applicable to this setting. '+out['info'])
        self.R = self.setting.penalty
        self.S = self.setting.data_fid
        if psi_plus is not None:
            self.psi_plus = psi_plus
        else:
            self.psi_plus  = np.broadcast_to(self.R.dom_u, setting.op.domain.shape) if isinstance(self.R.dom_u,(float,int)) else self.R.dom_u
        if psi_minus is not None:
            self.psi_minus = psi_minus
        else:
            self.psi_minus = np.broadcast_to(self.R.dom_l, setting.op.domain.shape) if isinstance(self.R.dom_l,(float,int)) else self.R.dom_l
        self.Rpp_plus, self.Rpp_minus = self._get_second_R_derivatives_at_bound_constraints()
        regpar= setting.regpar
        x0 = x0 if x0 is not None else np.clip(self.op.domain.zeros(),a_min=self.psi_minus,a_max= self.psi_plus)
        if not x0 in self.op.domain:
            raise ValueError(Errors.not_in_vecsp('x0 must be in op.domain',x0))
        if not np.all(x0<=self.psi_plus) or not np.all(x0>=self.psi_minus):
            raise ValueError(Errors.generic_message('x0 not feasible.'))
        y0 = self.op(x0)
        quad_setting = Setting(self.op, quadTaylor(self.R,x0),quadTaylor(self.S,y0),regpar=regpar)

        cg_pars = {'tol': 0.001/sqrt(self.regpar)}
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""

        self.log.setLevel(logging_level)
        self.cg_logging_level = cg_logging_level

        """Prepare first iteration step"""
        self.lam_plus = self.op.domain.zeros()
        self.lam_minus = self.op.domain.zeros()

        self.tikhcg=TikhonovCG(quad_setting, x0 = x0, logging_level=self.cg_logging_level, **self.cg_pars)
        self.x, self.y = self.tikhcg.run()

        self.active_plus = self.op.domain.IfPos(self.lam_plus +self.regpar*(self.x-self.psi_plus ))
        self.active_minus = self.op.domain.IfPos(self.lam_minus-self.regpar*(self.x-self.psi_minus))
        if not (self.active_plus).any() and not (self.active_minus).any():
            self.log.info('Stopped at 0th iterate.')
            self.converge()
        self.log.info('it {}: CG its {}; changes active sets +{},-{}'.format(self.iteration_step_nr,self.tikhcg.iteration_step_nr,
                                                                            (1.*self.active_minus+self.active_plus).sum(),0 )
        )


    def _next(self):
        """compute active and inactive sets"""
        self.active_plus_old=self.active_plus
        self.active_minus_old=self.active_minus
        self.active  = self.active_plus | self.active_minus
        self.inactive= self.op.domain.logical_not(self.active)

        """On the active sets the solution takes the values of the constraints."""
        self.x[self.active_plus]=self.psi_plus[self.active_plus]
        self.x[self.active_minus]=self.psi_minus[self.active_minus]

        """Lagrange parameters are 0 where the corresponding constraints are not active."""
        self.lam_plus[self.inactive]=0
        self.lam_plus[self.active_minus]=0
        self.lam_minus[self.inactive]=0
        self.lam_minus[self.active_plus]=0

        projection = CoordinateMask(self.h_domain.vecsp, self.inactive)
        if self.active.all():
            self.log.info('all indices active!')
        else:
            """Define quadratic Taylor approximation of the data fidelity functional at the current iterate"""
            S_taylor = quadTaylor(self.S,self.y)
            """... and a quadratic Taylor approximation of the full penalty functional"""
            full_R_taylor = quadTaylor(self.R,self.x)            
            """ ... as wel as its restriction to the inactive set.""" 
            hess_R_full =  full_R_taylor.gram
            constr_R_taylor = Composed(full_R_taylor,
                                            op = projection.adjoint,
                                            hessian = lambda x: Composition(projection.adjoint, hess_R_full, projection,
                                                                            inverse = projection.adjoint * hess_R_full.inverse * projection
                                                                            )
                                            )
            """Define a quadratic approximation of the setting at the current iterate restricted to the inactive set, 
               which respects the solution on the active set that we have already computed.
               Since we assume separability of the penalty term, the solution on the active set only affects the data term.
            """
            quad_setting = Setting(self.op * projection, 
                                  constr_R_taylor,
                                  S_taylor,
                                  regpar=self.regpar,
                                  data = self.op(projection(self.x)-self.x)
                                  )         
            tikhcg = TikhonovCG(quad_setting, 
                                x0 = projection(self.x), 
                                logging_level=self.cg_logging_level,
                                **self.cg_pars)
            aux, _ = tikhcg.run()
            self.x[self.inactive] = aux[self.inactive]
        self.y = self.op(self.x)

        S_taylor.data = 0.*self.y # Has been changed by data in quad_setting!
        residual = self.op.adjoint(S_taylor.subgradient(self.y))
        residual += self.regpar*full_R_taylor.subgradient(self.x)
        residual = self.setting.h_domain.gram_inv(residual)
        
        self.lam_plus[self.active_plus]  = -residual[self.active_plus]
        self.lam_minus[self.active_minus]= +residual[self.active_minus]

        """Update active and inactive sets"""
        self.active_plus  = (self.lam_plus +self.regpar*(self.x-self.psi_plus )) >=0 
        self.active_minus = (self.lam_minus-self.regpar*(self.x-self.psi_minus)) >=0
        added_ind = (self.op.domain.logical_and(self.active_plus,  self.op.domain.logical_not(self.active_plus_old ))).sum() \
                  + (self.op.domain.logical_and(self.active_minus, self.op.domain.logical_not(self.active_minus_old))).sum() 
        removed_ind = (self.op.domain.logical_and(self.active_plus_old, self.op.domain.logical_not(self.active_plus))).sum() \
                + (self.op.domain.logical_and(self.active_minus_old, self.op.domain.logical_not(self.active_minus))).sum()
        self.log.info('it {}: CG its {}, changes active sets +{},-{}'.format(self.iteration_step_nr,
                                                                            tikhcg.iteration_step_nr if not self.active.all() else 0,
                                                                            added_ind, removed_ind
                                                                            )
                        )
        if added_ind+removed_ind==0:
            self.converge()

    @staticmethod
    def check_applicability(setting,op_norm=None):
        out = {'info':''}
        if not setting.is_tikhonov:
            out['info'] += 'Setting has no regularization parameter.'
        if not setting.is_convex:
            out['info'] += 'Setting has not convex.'    
        if setting.codomain_free:
            out['info'] += 'SemiSmoothNewton requires codomain vectors. '
        else:
            if not 'hessian' in setting.data_fid.methods:
                out['info'] += 'Data functional not twice differentiable. '
            elif setting.data_fid.Lipschitz==np.inf:
                out['info'] += 'Data functional not Lipschitz continuous. '
        if not setting.penalty.is_separable:
            out['info'] += 'Penalty term not separable. '
        if not 'hessian' in setting.penalty.methods:
            out['info'] += 'Penalty term not twice differentiable. '
        out['applicable'] = out['info']==''
        out['rate'] = np.nan
        return out, None

    def _get_second_R_derivatives_at_bound_constraints(self):
        """
        Since the (separable!) penalty functional R may not be defined outside of the feasible set and SemiSmoothNewton may produce infeasible iterates 
        for which the gradient of the penalty must be evaluated, we extend the R by a quadratic function outside of the feasible interval in each variable.
        For the evaluation of the gradient of R, evaluations at the boundaries and knowledge of the second derivatives suffice. 

        Returns: 
        Rpp_plus, Rpp_minus:
            Arrays of the the of the grid containing the second derivatives of the component functions of R at the upper and lower bound.  
        """
        ub = self.psi_plus.copy()
        ub[ub==np.inf] = self.psi_minus[ub==np.inf]
        ub[ub==-np.inf] = 0.

        lb = self.psi_minus.copy()
        lb[lb==-np.inf] = self.psi_plus[lb==-np.inf]
        lb[lb==np.inf] = 0

        ones = self.h_domain.vecsp.ones()
        return self.R.hessian(ub)(ones), self.R.hessian(lb)(ones)
