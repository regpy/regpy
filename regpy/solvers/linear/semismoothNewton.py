from math import sqrt,inf
import numpy as np

from regpy.util import Errors
from regpy.operators import CoordinateMask 
from regpy.hilbert import GramHilbertSpace
from regpy.functionals.base import Functional, HorizontalShiftDilation, Conj, LinearCombination
from regpy.functionals.numpy import QuadraticBilateralConstraints,QuadraticLowerBound, QuadraticNonneg, Huber,LppPower
from regpy.stoprules import CountIterations

from ..general import RegSolver, Setting
from .tikhonov import TikhonovCG,GeometricSequence

__all__ = ["SemismoothNewton_bilateral","SemismoothNewton_nonneg","SemismoothNewtonAlphaGrid"]

class SemismoothNewton_bilateral(RegSolver):
    r"""Semi-smooth Newton method for minimizing quadratic Tikhonov functionals

    .. math::
        \Vert T x - data\Vert^2 + regpar * \Vert x - xref\Vert^2 

    subject to bilateral constraints :math:`psi_{minus} \leq x \leq psi_{plus}`

    
    Parameters
    ----------
    *args : [regpy.solvers.Setting,array-like,float] or [regpy.solver.Setting]
        Either 3 positional arguments [setting : `regpy.solvers.Setting`, data : `array-like`,
        regpar : `float`] consisting og the regularization setting, data and a positive float for the 
        regularization parameter or 1 positional argument [setting : regpy.solver.Setting] which 
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
                 cg_pars = None, logging_level = "INFO", cg_logging_level = "INFO",x0=None,
                 **kwargs
                 ):
        if len(args)==3:
            setting, data, regpar = args
            if setting.is_tikhonov:
                raise ValueError(Errors.value_error("If constructing the SemismoothNewton_bilateral with three arguments the setting must not contain a regularization parameter"))
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
            if Tsetting.is_tikhonov:
                raise ValueError(Errors.value_error("If constructing the SemismoothNewton_bilateral with one argument the setting must contain a regularization parameter."))
            out, _ = SemismoothNewton_bilateral.check_applicability(Tsetting)
            if not out['applicable']:
                raise RuntimeError('SemismoothNewton_bilateral not applicable to this setting. '+out['info'])
            R = Tsetting.penalty
            gram = Tsetting.h_domain.gram
            psi_plus, psi_minus, xref, alpha_fac = getPenaltyParamsFromFunctional(R,gram)
            regpar= Tsetting.regpar
            gramY = Tsetting.h_codomain.gram
            data = -gramY.inverse(Tsetting.data_fid.subgradient(Tsetting.op.codomain.zeros()))
            setting = Setting(Tsetting.op,
                                            GramHilbertSpace(R.hessian(0.5*(psi_plus+psi_minus))),
                                            GramHilbertSpace(Tsetting.data_fid.hessian(Tsetting.op.codomain.zeros()))
                                            )
        else:
            raise TypeError('SemismoothNewton_bilateral takes either 1 or 3 positional arguments ({} given)'.format(len(args)))
                                
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="SemismoothNewton_bilateral in as a linear solver requires the operator to be linear!"))
        if self.op.domain.dtype != float:
            raise TypeError(Errors.type_error("SemismoothNewton_bilateral requires the domain to be real!"))
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
                self.x = self.xref.copy()
        else:
            self.x = x0.copy()
        if cg_pars is None:
            cg_pars = {'tol': 0.001/sqrt(self.regpar)}
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        if psi_minus is None:
            self.psi_minus = -inf*self.op.domain.ones()
        else:
            self.psi_minus=psi_minus
        """The lower bound."""
        if psi_plus is None:
            self.psi_plus = inf*self.op.domain.ones()
        else:
            self.psi_plus=psi_plus
        """The upper bound."""
        if (self.psi_minus >= self.psi_plus).any():
            raise ValueError(Errors.value_error("The upper bound is less or equal the lower bound in SemismoothNewton_bilateral. Given: "+"\n\t "+f"psi_minus = {self.psi_minus} "+"\t\n "+f"psi_plus = {self.psi_plus}"))

        self.log.setLevel(logging_level)
        self.cg_logging_level = cg_logging_level

        self.b=self.h_domain.gram_inv(self.op.adjoint(self.h_codomain.gram(self.data)))
        if self.xref is not None:
            self.b += self.regpar*self.xref
            
        """Prepare first iteration step"""
        self.lam_plus = self.op.domain.zeros()
        self.lam_minus = self.op.domain.zeros()

        tikhcg=TikhonovCG(
                setting=Setting(self.op, self.h_domain, self.h_codomain),
                data=self.data, 
                regpar=self.regpar,
                xref=self.xref,
                x0 = self.x,
                logging_level=self.cg_logging_level,
                **self.cg_pars
            )
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
            tikhcg = TikhonovCG(
                setting=Setting(self.op * projection, self.h_domain, self.h_codomain),
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
        if not isQuadratic(setting.data_fid):
            out['info'] += 'Data functional not quadratic. '
        if not isQuadratic(setting.penalty):
            out['info'] += 'Penalty term not quadratic. '
        out['applicable'] = out['info']==''
        out['rate'] = np.nan
        return out, None

def isQuadratic(func):
    r"""checks if a functional is quadratic."""
    print("")
    if isinstance(func,(QuadraticBilateralConstraints,QuadraticNonneg)):
        return True
    elif isinstance(func,LppPower):
        return func.p==2
    elif isinstance(func,HorizontalShiftDilation):
        return isQuadratic(func.func)
    elif isinstance(func,LinearCombination):
        if len(func.coeffs)!=1:
            return False
        else:
            return isQuadratic(func.funcs[0])
    elif isinstance(func,Conj):
        return isQuadraticConj(func.func)
    else:
        return False

def isQuadraticConj(func):
    if isinstance(func, Huber):
        return True
    elif isinstance(func,LppPower):
        return func.p==2
    elif isinstance(func,HorizontalShiftDilation):
        return isQuadraticConj(func.func)
    elif isinstance(func,LinearCombination):
        if len(func.coeffs)!=1:
            return False
        else:
            return isQuadraticConj(func.funcs[0])        
    else:
        return False

 

def getPenaltyParamsFromFunctional(R,gram=None):
    r"""
    Extract the parameters :math:`u_b`, :math:`l_b`, :math:`x_0`, :math:`\alpha` from a functional 

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
    if not isinstance(R,Functional):
        raise TypeError(Errors.not_instance(R,Functional,add_info="Construction the parameters of upper and lower bound, x_0 and alpha from the regularization functional is only defined for a Functional!"))
    if isinstance(R,QuadraticBilateralConstraints):
        return R.ub, R.lb, R.x0, 1.
    elif isinstance(R,HorizontalShiftDilation):
        ub,lb,x0,alpha = getPenaltyParamsFromFunctional(R.func,gram)
        if R.shift_val is None:
            shift = R.domain.zeros()
        else:
            shift = R.shift_val
        if R.dilation >0:
            return shift + (1./R.dilation)*ub, shift + (1./R.dilation)*lb, shift+(1./R.dilation)*x0, alpha*R.dilation**2
        else:
            return shift + (1./R.dilation)*lb, shift + (1./R.dilation)*ub, shift+(1./R.dilation)*x0, alpha*R.dilation**2
    elif isinstance(R,Conj):
        return getPenaltyParamsFromConjFunctional(R.func,gram.inverse)
    else:
        raise TypeError(Errors.type_error('Unknown or inappropriate type of functional. Cannot construct the parameters of upper and lower bound, x_0 and alpha from the regularization functional.'))
    
def getPenaltyParamsFromConjFunctional(Rs,gram):
    r"""
    Extract the parameters :math:`u_b`, :math:`l_b`, :math:`x_0`, :math:`\alpha` from a functional 

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
    if not isinstance(Rs,Functional):
        raise TypeError(Errors.not_instance(Rs,Functional,add_info="Construction the parameters of upper and lower bound, x_0 and alpha from the conjugate regularization functional is only defined for a Functional!"))
    if isinstance(Rs,Huber):
        return gram(Rs.sigma), gram(-Rs.sigma), gram.domain.zeros(), 1.
    elif isinstance(Rs,LinearCombination):
        if len(Rs.coeffs)!=1:
            raise ValueError(Errors.value_error("Construction the parameters of upper and lower bound, x_0 and alpha from the conjugate regularization functional given as a LinearCombination is only given for linear combinations of length one (Scalar multiplications)!"))
        ub, lb, x0, alpha = getPenaltyParamsFromConjFunctional(Rs.funcs[0],gram)
        lam = Rs.coeffs[0]
        if lam<=0:
            raise ValueError(Errors.value_error("Construction the parameters of upper and lower bound, x_0 and alpha from the conjugate regularization functional given as a LinearCombination is only given for linear combinations with positive scalar multiplication!"))
        return lam*ub, lam*lb, x0 , alpha/lam
    elif isinstance(Rs,HorizontalShiftDilation):
        if Rs.dilation != 1.:
            raise ValueError(Errors.value_error("Construction the parameters of upper and lower bound, x_0 and alpha from the conjugate regularization functional given as a HorizontalShiftDilation is only given for non dilation!!"))
        ub, lb, x0, alpha = getPenaltyParamsFromConjFunctional(Rs.func,gram)
        return ub, lb, (x0 if Rs.shift_val is None else x0- (1./alpha)*Rs.shift_val), alpha
    else:
        raise TypeError(Errors.type_error('Unknown or inappropriate type of functional. Cannot construct the parameters of upper and lower bound, x_0 and alpha from the conjugate regularization functional.'))


class SemismoothNewton_nonneg(RegSolver):
    r"""Semismooth Newton method for minimizing quadratic Tikhonov functionals
    
    .. math::
        \Vert T x - data\Vert^2 + regpar * \Vert x - xref\Vert^2 \\
        subject to x>=0


    Compared to SemismoothNewton_bilateral, less storage is needed, and an a-posteriori stopping rule 
    can be used. By a change of variables, arbitrary lower bounds x\geq \psi may be used.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem.
    data : array-like, default None
        The measured data. If it is None it is taken from the setting.
    regpar : float, default None
        The regularization parameter. Must be positive. If it is None it is taken from the setting.
    xref: array-like, default: None
        Reference value in the Tikhonov functional. The default is equivalent to xref = setting.op.domain.zeros().
    x0: array-like, default: None
        First iterate. If None, then x0=xref
    lambda0: array-like, default: None
        Initial guess for Lagrange parameter
    cg_pars: dictionary, default: None
        Parameters of CG method for minimizing Tikhonov functional on inactive set in each SS Newton step.
    TOL: float, default: 0
        Tolerance for absolute error in standard l^2-norm for a-posteriori duality gap error estimate given by 
         :math:`\|x-xtrue\|_2^2 \leq \|[T^*p-xref]_+-x\|^2 - 2 <[T^*p-xref]_-,x> \leq TOL^2`  where :math:`p =-(Tx-data)/regpar`
    logging_level: default: logging:INFO

    cg_logging_level: default: logging.INFO

    """
    def __init__(self,setting, data=None, regpar=None, xref = None,  x0=None, lambda0=None, cg_pars = None, TOL = 0.,
                 logging_level = "INFO", cg_logging_level = "INFO"):
        super().__init__(setting)
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        if(regpar is None):
            if(not setting.is_tikhonov):
                raise ValueError(Errors.value_error("Regularization parameter has to be included in setting or given directly."))
            regpar=setting.regpar
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="SemismoothNewton_nonneg in as a linear solver requires the operator to be linear!"))
        if self.op.domain.dtype != float:
            raise TypeError(Errors.type_error("SemismoothNewton_nonneg requires the domain to be real!"))
        if data not in self.op.codomain:
            raise ValueError(Errors.not_in_vecsp(data,self.op.codomain,vec_name="data",space_name="codomain"))
        if x0 is not None and x0 not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(x0,self.op.domain,vec_name="first iteration",space_name="domain"))
        if xref is not None and xref not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(xref,self.op.domain,vec_name="reference",space_name="domain"))
        out, _ = SemismoothNewton_nonneg.check_applicability(setting)
        if not out['applicable']:
            raise RuntimeError('SemismoothNewton_nonneg not applicable to this setting. '+out['info'])
        self.data=data
        """The measured data"""
        self.xref = xref
        """The reference value in the Tikhonov functional."""
        if x0 is None:
            if xref is None:
                self.x=self.op.domain.zeros()
            else:
                self.x = xref.copy()
        else:
            self.x = x0.copy()
            """The current iterate"""
        self.regpar=regpar
        """The regularizaton parameter."""
        if cg_pars is None:
            cg_pars = {'tol': 0.001/sqrt(self.regpar)}
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

        self.lam = lambda0 if lambda0 is not None else self.op.domain.zeros()
        tikhcg=TikhonovCG(
                setting=Setting(self.op, self.h_domain, self.h_codomain),
                data=self.data, 
                regpar=self.regpar,
                xref=self.xref,
                x0 = self.xref,
                logging_level=self.cg_logging_level,
                **self.cg_pars
            )
        self.x, self.y = tikhcg.run()
        cg_its = tikhcg.iteration_step_nr
        self.active= self.op.domain.IfPos(self.lam-self.regpar*self.x) 
        if not self.active.any():
            self.log.info('Stopped at 0th iterate.')
            self.converge()
        self.log.debug('it {}: CG its {}; changes active set +{},-{}'.format(self.iteration_step_nr,cg_its,
                                                                            self.active.sum(),0 )
        )

    @staticmethod
    def check_applicability(setting,op_norm=None):
        out = SemismoothNewton_bilateral(setting)
        if np.any(setting.penalty.dom_u<inf):
            out['info'] = '' if out['applicable'] else out['info']
            out['applicable'] = False
            out['info'] += 'SemismoothNewton_nonneg cannot handle upper bounds.'
        return out, None

    def _next(self):

        """compute active and inactive sets, need to be computed in each step again"""
        self.active_old=self.active
        self.inactive= self.op.domain.logical_not(self.active)

        # On the active sets the solution takes the values of the constraints.
        self.x[self.active]=0

        # Lagrange parameters are 0 where the corresponding constraints are not active. 
        self.lam[self.inactive]=0

        projection = CoordinateMask(self.h_domain.vecsp, self.inactive)
        cg_its = 0
        if self.active.all():
            self.log.debug('all indices active!')
        else:
            tikhcg=TikhonovCG(
                setting=Setting(self.op * projection, self.h_domain, self.h_codomain),
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
        aux_pos = self.op.domain.IfPos(aux)
        bound = self.op.domain.norm(aux[aux_pos]-self.x[aux_pos])**2 - 2*self.op.domain.vdot(-aux[~aux_pos],self.x[~aux_pos])
        if sqrt(bound)<=self.TOL:
            self.log.info('Stopped by a-posteriori error estimate.')
            self.converge()

        z += self.regpar*self.x
        self.lam[self.active]=z[self.active]

        #Update active and inactive sets
        self.active = (self.lam-self.regpar*self.x)>=0
        added_ind = (self.op.domain.logical_and(self.active, self.op.domain.logical_not(self.active_old))).sum() 
        removed_ind = (self.op.domain.logical_and(self.active_old, self.op.domain.logical_not(self.active))).sum()
        self.log.debug('it {}: CG its {}; changes active set +{},-{}; error bound {:1.2e}/{:1.2e}'.format(self.iteration_step_nr,
                                                                            cg_its,
                                                                            added_ind, removed_ind,
                                                                            sqrt(bound),self.TOL
                                                                            )
                        )
        if added_ind+removed_ind==0:
            self.converge()

class SemismoothNewtonAlphaGrid(RegSolver):
    r"""Class running Tikhonov regularization with bound constraints on a grid of different regularization parameters.

    Parameters
    ----------
    setting:  regpy.solvers.Setting
        The setting of the forward problem.
    alphas: Either an iterable giving the grid of alphas or a tuple (alpha0,q)
        In the latter case the sequence :math:`(alpha0*q^n)_{n=0,1,2,...}` is generated.
    data: array-like, default None
        The right hand side. If it is None the data is taken from setting.
    xref: array-like, default None
        initial guess in Tikhonov functional. Default corresponds to zeros()
    max_Newton_iter: int, default: 50
        maximum number of Newton iterations
    tol_fac: float, default: 0.33
        absolute tolerance for termination of SSNewton by a-posteriori error estimation is tol_fac/sqrt(alpha)
    tol_fac_cg: float, default: 1e-6
        absolute tolerance for inner cg iteration is tol_fac_cg/sqrt(alpha)
    """
    def __init__(self,setting,alphas, data=None, xref=None,max_Newton_iter=50,
                 delta=None, tol_fac = 0.33, tol_fac_cg = 1e-6, logging_level= "INFO"):
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="SemismoothNewtonAlphaGrid in as a linear solver requires the operator to be linear!"))
        if self.op.domain.dtype != float:
            raise TypeError(Errors.type_error("SemismoothNewtonAlphaGrid requires the domain to be real!"))
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        if data not in self.op.codomain:
            raise ValueError(Errors.not_in_vecsp(data,self.op.codomain,vec_name="data",space_name="codomain"))
        if xref is not None and xref not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(xref,self.op.domain,vec_name="reference",space_name="domain"))
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
        setting = Setting(op=self.op, penalty = self.h_domain, data_fid = self.h_codomain)
        inner_stoprule = CountIterations(max_iterations=self.max_Newton_iter)
        inner_stoprule.log = self.log.getChild('CountIterations')
        inner_stoprule.log.setLevel("WARNING")
        if not hasattr(self,'alpha_old'):
            SSNewton = SemismoothNewton_nonneg(setting,self.data,self.alpha,xref=self.xref,
                                TOL = self.tol_fac / sqrt(self.alpha),
                                cg_pars = {'tol': self.tol_fac_cg / sqrt(self.alpha)},
                                logging_level=self.logging_level,
                                cg_logging_level = "WARNING"
                               )    
        else:
            lambda0 = (self.alpha/self.alpha_old)*self.lam
            SSNewton = SemismoothNewton_nonneg(setting,self.data,self.alpha,xref=self.xref,x0=self.x,lambda0=lambda0,
                                TOL = self.tol_fac / sqrt(self.alpha),
                                cg_pars = {'tol': self.tol_fac_cg / sqrt(self.alpha)},
                                logging_level=self.logging_level,
                                cg_logging_level = "WARNING"
                               )
        self.x, self.y = SSNewton.run(inner_stoprule)
        self.lam = SSNewton.lam
        self.log.info('alpha = {}, SS Newton its = {}'.format(self.alpha,inner_stoprule.iteration))