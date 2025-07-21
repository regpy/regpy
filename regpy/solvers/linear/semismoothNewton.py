from regpy.solvers import RegSolver
import numpy as np
from regpy.operators import CoordinateMask 
from regpy.hilbert import GramHilbertSpace
from regpy.solvers import RegularizationSetting, TikhonovRegularizationSetting
from regpy.solvers.linear.tikhonov import TikhonovCG, GeometricSequence
from regpy.functionals import Functional,QuadraticBilateralConstraints, HorizontalShiftDilation, Conj, Huber, LinearCombination
from regpy.stoprules import CountIterations
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

class SemismoothNewton_bilateral(RegSolver):
    r"""Semi-smooth Newton method for minimizing quadratic Tikhonov functionals

    .. math::
        \Vert T x - data\Vert^2 + regpar * \Vert x - xref\Vert^2 

    subject to bilateral constraints :math:`psi_{minus} \leq x \leq psi_{plus}`

    
    Parameters
    ----------
    *args : [regpy.solvers.RegularizationSetting,array-like,float] or [regpy.solver.TikhonovRegularizationSetting]
        Either 3 positional arguments [setting : `regpy.solvers.RegularizationSetting`, data : `array-like`,
        regpar : `float`] consisting og the regularization setting, data and a positive float for the 
        regularization parameter or 1 positional argument [setting : regpy.solver.TikhonovRegularizationSetting] which 
        already binds the former arguments together.
    xref: array-like, default: None
        Reference value in the Tikhonov functional. The default is equivalent to xref = setting.op.domain.zeros().
    x0: array-like, default: None
        Zeroth iterate. If None, x0=xref
    psi_plus: array-like, default: None
        The upper bound. In the default case it is +inf
    psi_minus: array-like, default: None
        The lower bound. In the default case it is -inf
    cg_pars: dictionary, default: None
        Parameters of CG method for minimizing Tikhonov functional on inactive set in each SS Newton step.
    logging_level: Loglevel
        default: logging:INFO
    cg_logging_level: Loglevel
        default: logging.INFO

    Notes
    -----
    In this case 
     * setting.penalty has to be an instance of one of the following classes: 
        * QuadraticBilateralConstraints, 
        * conj of Huber
        * conj of HorizontalShiftDilation of Huber
     * Then psi_plus, psi_minus, xref and regpar are extracted from setting.penalty.
     * setting.data_fid has to be a shifted quadratic functional, and data is extracted from the shift.
     * regpar is setting.regpar
    
    Keyword arguments x0, cg_pars, logging_level, and cg_logging_level are as for the case of 3 positional arguments. 

    """

    def __init__(self, *args,
                 cg_pars = None, logging_level = logging.INFO, cg_logging_level = logging.INFO,x0=None,
                 **kwargs
                 ):
        if len(args)==3:
            setting, data, regpar = args
            assert isinstance(setting,RegularizationSetting)
            if 'psi_plus' in kwargs:
                psi_plus = kwargs['psi_plus']
            else:
                psi_plus = None
            if 'psi_minus' in kwargs:
                psi_minus = kwargs['psi_minus']
            else:
                psi_minus = None
            if 'xref' in kwargs:
                xref = kwargs['xref']
            else:
                xref = None
            alpha_fac = 1.
        elif len(args)==1:
            Tsetting = args[0]
            assert isinstance(Tsetting,TikhonovRegularizationSetting)
            R = Tsetting.penalty
            gram = Tsetting.h_domain.gram
            psi_plus, psi_minus, xref, alpha_fac = getPenaltyParamsFromFunctional(R,gram)
            regpar= Tsetting.regpar
            gramY = Tsetting.h_codomain.gram
            data = -gramY.inverse(Tsetting.data_fid.subgradient(Tsetting.op.codomain.zeros()))
            setting = RegularizationSetting(Tsetting.op,
                                            GramHilbertSpace(R.hessian(0.5*(psi_plus+psi_minus))),
                                            GramHilbertSpace(Tsetting.data_fid.hessian(Tsetting.op.codomain.zeros()))
                                            )
        else:
            raise TypeError('SemismoothNewton_bilateral takes either 1 or 3 positional arguments ({} given)'.format(len(args)))
                                
        super().__init__(setting)
        assert self.op.domain.dtype == float
        self.data=data
        """The measured data"""
        self.regpar=regpar * alpha_fac
        """The regularizaton parameter."""
        self.xref = (1./alpha_fac)*xref if xref is not None else setting.op.domain.zeros()
        """The initial guess."""
        if x0 is None:
            if xref is None:
                self.x=self.op.domain.zeros()
            else:
                self.x = np.copy(self.xref)
        else:
            self.x = np.copy(x0)
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

        self.b=self.h_domain.gram_inv(self.op.adjoint(self.h_codomain.gram(self.data)))
        if self.xref is not None:
            self.b += self.regpar*self.xref
            
        """Prepare first iteration step"""
        self.lam_plus = np.zeros_like(self.b)
        self.lam_minus = np.zeros_like(self.b)
        tikhcg=TikhonovCG(
                setting=RegularizationSetting(self.op, self.h_domain, self.h_codomain),
                data=self.data, 
                regpar=self.regpar,
                xref=self.xref,
                x0 = self.x,
                logging_level=self.cg_logging_level,
                **self.cg_pars
            )
        self.x, self.y = tikhcg.run()
        cg_its = tikhcg.iteration_step_nr
        self.active_plus = (self.lam_plus +self.regpar*(self.x-self.psi_plus ))>=0 
        self.active_minus = (self.lam_minus-self.regpar*(self.x-self.psi_minus))>=0 
        if not np.any(self.active_plus) and not np.any(self.active_minus):
            self.log.info('Stopped at 0th iterate.')
            self.converge()
        self.log.debug('it {}: CG its {}; changes active sets +{},-{}'.format(self.iteration_step_nr,cg_its,
                                                                            np.sum(1.*self.active_minus+self.active_plus),0 )
        )


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
        cg_its = 0
        if self.active.all():
            self.log.info('all indices active!')
        else:
            tikhcg = TikhonovCG(
                setting=RegularizationSetting(self.op * projection, self.h_domain, self.h_codomain),
                data=self.data-self.op(self.x-projection(self.x)), 
                regpar=self.regpar,
                xref=projection(self.xref),
                x0 = projection(self.x),
                logging_level=self.cg_logging_level,
                **self.cg_pars
            )
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
        added_ind = np.sum(np.logical_and(self.active_plus,  np.logical_not(self.active_plus_old ))) \
                  + np.sum(np.logical_and(self.active_minus, np.logical_not(self.active_minus_old))) 
        removed_ind = np.sum(np.logical_and(self.active_plus_old, np.logical_not(self.active_plus))) \
                + np.sum(np.logical_and(self.active_minus_old, np.logical_not(self.active_minus)))
        self.log.info('it {}: CG its {}, changes active sets +{},-{}'.format(self.iteration_step_nr,
                                                                            cg_its,
                                                                            added_ind, removed_ind
                                                                            )
                        )
        if added_ind+removed_ind==0:
            self.converge()


def getPenaltyParamsFromFunctional(R,gram=None):
    r"""
    Extract the parameters :math:`u_b`, :math:`l_b`, :math:`x_0`, :math:`\alpah` from a functional 

    .. math::
        R(x) &= \frac{\alpha}{2} \|x-x_0\|^2 +c   if l_b\leq x\leq u_b\\
        R(x) &= \infty else

    Parameters
    ----------
    R: regpy.functional.Functional
       The functional to be analyzed.
    gram: regpy.operator.Operator [default: None]
       Gram matrix of the dual Hilbert space. Only used if R is a conjugate functional       
    """
    assert isinstance(R,Functional)
    if isinstance(R,QuadraticBilateralConstraints):
        return R.ub, R.lb, R.x0, 1.
    elif isinstance(R,HorizontalShiftDilation):
        ub,lb,x0,alpha = getPenaltyParamsFromFunctional(R.F,gram)
        if R.shift is None:
            shift = R.domain.zeros()
        else:
            shift = R.shift
        if R.dilation >0:
            return shift + (1./R.dilation)*ub, shift + (1./R.dilation)*lb, shift+(1./R.dilation)*x0, alpha*R.dilation**2
        else:
            return shift + (1./R.dilation)*lb, shift + (1./R.dilation)*ub, shift+(1./R.dilation)*x0, alpha*R.dilation**2
    elif isinstance(R,Conj):
        return getPenaltyParamsFromConjFunctional(R.func,gram.inverse)
    else:
        raise TypeError('Unknown or inappropriate type of functional')
    
def getPenaltyParamsFromConjFunctional(Rs,gram):
    r"""
    Extract the parameters :math:`u_b`, :math:`l_b`, :math:`x_0`, :math:`\alpah` from a functional 

    .. math::
        R^*(x) &= \frac{\alpha}{2} \|x-x_0\|^2 +c   if lb\leq x\leq ub \\
        R^*(x) &= \infty else


    Parameters
    ----------
    Rs: regpy.functional.Functional
       The functional to be analyzed.
    gram: regpy.operator.Operator [default: None]
       Gram matrix of the Hilbert space on which Rs is defined 
    """
    assert isinstance(Rs,Functional)
    if isinstance(Rs,Huber):
        return gram(Rs.sigma), gram(-Rs.sigma), gram.domain.zeros(), 1.
    elif isinstance(Rs,LinearCombination):
        assert len(Rs.coeffs)==1
        ub, lb, x0, alpha = getPenaltyParamsFromConjFunctional(Rs.funcs[0],gram)
        lam = Rs.coeffs[0]
        assert lam>0
        return lam*ub, lam*lb, x0 , alpha/lam
    elif isinstance(Rs,HorizontalShiftDilation):
        assert Rs.dilation == 1.
        ub, lb, x0, alpha = getPenaltyParamsFromConjFunctional(Rs.F,gram)
        return ub, lb, (x0 if Rs.shift is None else x0- (1./alpha)*Rs.shift), alpha
    else:
        raise TypeError('Unknown or inappropriate type of functional')


class SemismoothNewton_nonneg(RegSolver):
    r"""Semismooth Newton method for minimizing quadratic Tikhonov functionals
    
    .. math::
        \Vert T x - data\Vert^2 + regpar * \Vert x - xref\Vert^2 \\
        subject to x>=0


    Compared to SemismoothNewton_bilateral, less storage is needed, and an a-posteriori stopping rule 
    can be used. By a change of variables, arbitrary lower bounds x\geq \psi may be used.

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
    x0: array-like, default: None
        First iterate. If None, then x0=xref
    lambda0: array-like, default: None
        Initial guess for Lagrange parameter
    cg_pars: dictionary, default: None
        Parameters of CG method for minimizing Tikhnonov functional on inactive set in each SS Newton step.
    TOL: float, default: 0
        Tolerance for absolute error in standard l^2-norm for a-posteriori duality gap error estimate given by 
         \|x-xtrue\|_2^2 \leq \|[T^*p-xref]_+-x\|^2 - 2 <[T^*p-xref]_-,x> \leq TOL^2  where p =-(Tx-data)/regpar
    logging_level: default: logging:INFO

    cg_logging_level: default: logging.INFO

    """
    def __init__(self,setting, data, regpar, xref = None,  x0=None, lambda0=None, cg_pars = None, TOL = 0.,
                 logging_level = logging.INFO, cg_logging_level = logging.INFO):
        assert isinstance(setting,RegularizationSetting)
        super().__init__(setting)
        assert self.op.domain.dtype == float
        self.data=data
        """The measured data"""
        self.xref = xref
        """The reference value in the Tikhonov functional."""
        if x0 is None:
            if xref is None:
                self.x=self.op.domain.zeros()
            else:
                self.x = np.copy(xref)
        else:
            self.x = np.copy(x0)
            """The current iterate"""
        self.regpar=regpar
        """The regularizaton parameter."""
        if cg_pars is None:
            cg_pars = {'tol': 0.001/np.sqrt(self.regpar)}
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        self.TOL = TOL
        """Absolute tolerance."""
        self.log.setLevel(logging_level)
        self.cg_logging_level = cg_logging_level

        """Prepare first iteration step"""
        self.y = self.op(self.x)
        self.b=self.h_domain.gram_inv(self.op.adjoint(self.h_codomain.gram(self.data)))
        if self.xref is not None:
            self.b += self.regpar*self.xref

        self.lam = lambda0 if lambda0 is not None else np.zeros_like(self.b)
        tikhcg=TikhonovCG(
                setting=RegularizationSetting(self.op, self.h_domain, self.h_codomain),
                data=self.data, 
                regpar=self.regpar,
                xref=self.xref,
                x0 = self.xref,
                logging_level=self.cg_logging_level,
                **self.cg_pars
            )
        self.x, self.y = tikhcg.run()
        cg_its = tikhcg.iteration_step_nr
        self.active= (self.lam-self.regpar*self.x)>=0 
        if not np.any(self.active):
            self.log.info('Stopped at 0th iterate.')
            self.converge()
        self.log.debug('it {}: CG its {}; changes active set +{},-{}'.format(self.iteration_step_nr,cg_its,
                                                                            np.sum(1.*self.active),0 )
        )

    def _next(self):

        """compute active and inactive sets, need to be computed in each step again"""
        self.active_old=self.active
        self.inactive= np.logical_not(self.active)

        # On the active sets the solution takes the values of the constraints.
        self.x[self.active]=0

        # Lagrange parameters are 0 where the corresponing constraints are not active. 
        self.lam[self.inactive]=0

        projection = CoordinateMask(self.h_domain.vecsp, self.inactive)
        cg_its = 0
        if self.active.all():
            self.log.debug('all indices active!')
        else:
            tikhcg=TikhonovCG(
                setting=RegularizationSetting(self.op * projection, self.h_domain, self.h_codomain),
                data=self.data-self.op(self.x-projection(self.x)), 
                regpar=self.regpar,
                xref=self.xref,
                x0 = projection(self.x),
                logging_level=self.cg_logging_level,
                **self.cg_pars
            )
            f, _ = tikhcg.run()
            cg_its = tikhcg.iteration_step_nr
            self.x[self.inactive] = f[self.inactive]
        self.y = self.op(self.x)
        z =  self.h_domain.gram_inv(self.op.adjoint(self.h_codomain.gram(self.y)))-self.b
        aux = (-1/self.regpar)*z
        bound = np.linalg.norm(np.maximum(aux,0)-self.x)**2 - 2*np.vdot(np.maximum(-aux,0),self.x)
        if np.sqrt(bound)<=self.TOL:
            self.log.info('Stopped by a-posteriori error estimate.')
            self.converge()

        z += self.regpar*self.x
        self.lam[self.active]=z[self.active]

        #Update active and inactive sets
        self.active = (self.lam-self.regpar*self.x)>=0
        added_ind =  np.sum(np.logical_and(self.active, np.logical_not(self.active_old))) 
        removed_ind = np.sum(np.logical_and(self.active_old, np.logical_not(self.active)))
        self.log.debug('it {}: CG its {}; changes active set +{},-{}; error bound {:1.2e}/{:1.2e}'.format(self.iteration_step_nr,
                                                                            cg_its,
                                                                            added_ind, removed_ind,
                                                                            np.sqrt(bound),self.TOL
                                                                            )
                        )
        if added_ind+removed_ind==0:
            self.converge()

class SemismoothNewtonAlphaGrid(RegSolver):
    r"""Class runnning Tikhononv regularization with bound constraints on a grid of different regularization parameters.

    Parameters
    ----------
    setting:  regpy.solvers.RegularizationSetting
        The setting of the forward problem.
    data: array-like
        The right hand side.
    alphas: Either an iterable giving the grid of alphas or a tuple (alpha0,q)
        In the latter case the seuqence :math:`(alpha0*q^n)_{n=0,1,2,...}` is generated.
    xref: array-like, default None
        initial guess in Tikhonov functional. Default corresponds to zeros()
    max_Newton_iter: int, default: 50
        maximum number of Newton iterations
    tol_fac: float, default: 0.33
        absolute tolerance for termination of SSNewton by a-posteriori error estimation is tol_fac/sqrt(alpha)
    tol_fac_cg: float, default: 1e-6
        absolute tolerance for inner cg iteration is tol_fac_cg/sqrt(alpha)
    """
    def __init__(self,setting, data, alphas, xref=None,max_Newton_iter=50,
                 delta=None, tol_fac = 0.33, tol_fac_cg = 1e-6, logging_level= logging.INFO):
        super().__init__(setting)
        if isinstance(alphas,tuple) and len(alphas)==2:
            self._alphas = GeometricSequence(alphas[0],alphas[1])
        else:
            self._alphas = iter(alphas)
        self.data = data
        """Right hand side of the operator equation."""
        self.xref = xref
        """initial guess in Tikhonov functional."""
        if self.xref is not None:
            self.x = self.xref
            self.y = self.op(self.xref)
        else:
            self.x = self.op.domain.zeros()
            self.y = self.op.codomain.zeros()
        self.max_Newton_iter = max_Newton_iter
        """maximum number of CG iterations."""    
        self.tol_fac = tol_fac
        """absolute tolerance for termination of SSNewton by a-posteriori error estimation"""
        self.tol_fac_cg = tol_fac_cg
        """tolerance factor for inner cg iteration"""
        self.logging_level = logging_level
        """logging level for CG iteration."""

    def _next(self):
        try:
            if hasattr(self,'alpha'):
                self.alpha_old = self.alpha
            self.alpha = next(self._alphas)
        except StopIteration:
            return self.converge()
        setting = RegularizationSetting(op=self.op, penalty = self.h_domain, data_fid = self.h_codomain)
        inner_stoprule = CountIterations(max_iterations=self.max_Newton_iter)
        inner_stoprule.log = self.log.getChild('CountIterations')
        inner_stoprule.log.setLevel(logging.WARNING)
        if not hasattr(self,'alpha_old'):
            SSNewton = SemismoothNewton_nonneg(setting,self.data,self.alpha,xref=self.xref,
                                TOL = self.tol_fac / np.sqrt(self.alpha),
                                cg_pars = {'tol': self.tol_fac_cg / np.sqrt(self.alpha)},
                                logging_level=self.logging_level,
                                cg_logging_level = logging.WARNING
                               )    
        else:
            lambda0 = (self.alpha/self.alpha_old)*self.lam
            SSNewton = SemismoothNewton_nonneg(setting,self.data,self.alpha,xref=self.xref,x0=self.x,lambda0=lambda0,
                                TOL = self.tol_fac / np.sqrt(self.alpha),
                                cg_pars = {'tol': self.tol_fac_cg / np.sqrt(self.alpha)},
                                logging_level=self.logging_level,
                                cg_logging_level = logging.WARNING
                               )
        self.x, self.y = SSNewton.run(inner_stoprule)
        self.lam = SSNewton.lam
        self.log.info('alpha = {}, SS Newton its = {}'.format(self.alpha,inner_stoprule.iteration))