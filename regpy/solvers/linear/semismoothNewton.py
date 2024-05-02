from regpy.solvers import RegSolver
import numpy as np
from regpy.operators import CoordinateMask 
from regpy.solvers import RegularizationSetting
from regpy.solvers.linear.tikhonov import TikhonovCG
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)


class SemismoothNewton_bilateral(RegSolver):
    r"""Semismooth Newton method for minimizing quadratic Tikhonov functionals
    \[
        \Vert T x - data\Vert^2 + regpar * \Vert x - xref\Vert^2
        subject to bilateral constraints psi_minus \leq x \leq psi_plus
    \]
    
    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem.
    data : array-like
        The measured data.
    regpar : float
        The regularization parameter. Must be positive.
    xref: array-like, default: None
        Reference value in the Tikhonov functional. The default is equivalent to xref = setting.op.domain.zeros().
    psi_plus: array-like, default: None
        The upper bound. In the default case it is +inf
    psi_minus: array-like, default: None
        The lower bound. In the default case it is -inf
    cg_pars: dictionary, default: None
        Parameters of CG method for minimizing Tikhnonov functional on inactive set in each SS Newton step.
    logging_level: default: logging:INFO

    cg_logging_level: default: logging.INFO

    """
    def __init__(self,setting, data, regpar, xref = None, psi_plus = None, psi_minus = None, cg_pars = None,
                 logging_level = logging.INFO, cg_logging_level = logging.INFO):
        assert isinstance(setting,RegularizationSetting)
        super().__init__(setting)
        assert self.op.domain.dtype == float
        self.data=data
        """The measured data"""
        self.xref = xref
        """The initial guess."""
        if xref is None:
            self.x=self.op.domain.zeros()
        else:
            self.x = np.copy(xref)
        self.regpar=regpar
        """The regularizaton parameter."""
        if cg_pars is None:
            cg_pars = {'tol': 0.001/np.sqrt(self.regpar)}
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        if psi_minus is None:
            self.psi_minus = -np.inf*np.ones_like(self.x)
        else:
            self.psi_minus=psi_minus
        """The lower bound."""
        if psi_plus is None:
            self.psi_plus = np.inf*np.ones_like(self.x)
        else:
            self.psi_plus=psi_plus
        """The upper bound."""
        assert np.all(self.psi_minus < self.psi_plus)
        self.log.setLevel(logging_level)
        self.cg_logging_level = cg_logging_level

        """Prepare first iteration step"""
        self.y = self.op(self.x)
        self.rhs=self.data-self.y
        self.b=self.h_domain.gram_inv(self.op.adjoint(self.h_codomain.gram(self.rhs)))
        if self.xref is not None:
            self.b += self.regpar*self.xref

        res = self.b - self.regpar*self.x - self.h_domain.gram_inv(self.op.adjoint(self.h_codomain.gram(self.y)))
        self.lam_plus=np.maximum(np.zeros_like(res), res)
        self.lam_minus=-np.minimum(np.zeros_like(res), res)

        self.active_plus = self.lam_plus +self.regpar*(self.x-self.psi_plus )>=0 
        self.active_minus= self.lam_minus-self.regpar*(self.x-self.psi_minus)>=0 

    def _next(self):

        """compute active and inactive sets, need to be computed in each step again"""
        self.active_plus_old=self.active_plus
        self.active_minus_old=self.active_minus
        self.active  = np.logical_or(self.active_plus, self.active_minus)
        self.inactive= np.logical_not(self.active)

        # On the active sets the solution takes the values of the constraints.
        self.x[self.active_plus]=self.psi_plus[self.active_plus]
        self.x[self.active_minus]=self.psi_minus[self.active_minus]

        # Lagrange parameters are 0 where the corresponing constraints are not active. 
        self.lam_plus[self.inactive]=0
        self.lam_plus[self.active_minus]=0
        self.lam_minus[self.inactive]=0
        self.lam_minus[self.active_plus]=0

        projection = CoordinateMask(self.h_domain.vecsp, self.inactive)
        if self.active.all():
            self.log.info('all indices active!')
        else:
            f, _ = TikhonovCG(
                setting=RegularizationSetting(self.op * projection, self.h_domain, self.h_codomain),
                data=self.rhs, 
                regpar=self.regpar,
                xref=self.xref,
                x0 = projection(self.x),
                logging_level=self.cg_logging_level,
                **self.cg_pars
            ).run()
            self.x[self.inactive] = f[self.inactive]
        self.y = self.op(self.x)
        z = self.regpar*self.x + self.h_domain.gram_inv(self.op.adjoint(self.h_codomain.gram(self.y)))
        
        self.lam_plus[self.active_plus]  = self.b[self.active_plus] -z[self.active_plus]
        self.lam_minus[self.active_minus]=-self.b[self.active_minus]+z[self.active_minus]

        #Update active and inactive sets
        self.active_plus  = self.lam_plus +self.regpar*(self.x-self.psi_plus) >0 
        self.active_minus = self.lam_minus-self.regpar*(self.x-self.psi_minus)>0
        added_ind = np.sum(np.logical_and(self.active_plus,  np.logical_not(self.active_plus_old ))) \
                  + np.sum(np.logical_and(self.active_minus, np.logical_not(self.active_minus_old))) 
        removed_ind = np.sum(np.logical_and(self.active_plus_old, np.logical_not(self.active_plus))) \
                + np.sum(np.logical_and(self.active_minus_old, np.logical_not(self.active_minus)))
        self.log.debug('{} indices added to active set, {} removed'.format(added_ind, removed_ind))
        if added_ind+removed_ind==0:
            self.converge()
