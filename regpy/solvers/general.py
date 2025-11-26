import math as ma
from scipy.sparse.linalg import eigsh


from regpy.util import ClassLogger, Errors
from regpy.util.operator_tests import test_derivative
from regpy.operators import Operator
from regpy.functionals.base import  as_functional, Composed
from regpy.functionals import SquaredNorm, QuadraticLowerBound, QuadraticNonneg, QuadraticBilateralConstraints
from regpy.stoprules import NoneRule,DualityGapStopping,CombineRules,CountIterations
from numpy import inf
import logging

class Solver:
    r"""Abstract base class for solvers. Solvers do not implement loops themselves, but are driven by
    repeatedly calling the `next` method. They expose the current iterate stored in and value as attributes
    `x` and `y`, and can be iterated over, yielding the `(x, y)` tuple on every iteration (which
    may or may not be the same arrays as before, modified in-place).

    There are some convenience methods to run the solver with a `regpy.stoprules.StopRule`.

    Subclasses should override the method `_next(self)` to perform a single iteration where the values of 
    the attributes `x` and `y` are updated. The main difference to `next` is that `_next` does not have a
    return value. If the solver converged, `converge` should be called, afterwards `_next` will never be
    called again. Most solvers will probably never converge on their own, but rely on the caller or a
    `regpy.stoprules.StopRule` for termination.

    Parameters
    ----------
    x : numpy.ndarray
        Initial argument for iteration. Defaults to None.
    y : numpy.ndarray
        Initial value at current iterate. Defaults to None.
    """

    log = ClassLogger()

    def __init__(self,x=None,y=None):
        self.x = x
        """The current iterate."""
        self.y = y
        """The value at the current iterate. May be needed by stopping rules, but callers should
        handle the case when it is not available."""
        self.__converged = False
        self.iteration_step_nr = 0
        """Current number of iterations performed."""

    def converge(self):
        """Mark the solver as converged. This is intended to be used by child classes
        implementing the `_next` method.
        """
        self.__converged = True

    def next(self):
        r"""Perform a single iteration.

        Returns
        -------
        boolean
            False if the solver already converged and no step was performed.
            True otherwise.
        """
        if self.__converged:
            return False
        self.iteration_step_nr += 1    
        self._next()
        return True

    def _next(self):
        r"""Perform a single iteration. This is an abstract method called from the public method
        `next`. Child classes should override it.

        The main difference to `next` is that `_next` does not have a return value. If the solver
        converged, `converge` should be called.
        """
        raise NotImplementedError

    def __iter__(self):
        r"""Return an iterator on the iterates of the solver.

        Yields
        ------
        tuple of arrays
            The (x, y) pair of the current iteration.
        """
        while self.next():
            yield self.x, self.y

    def while_(self, stoprule=NoneRule()):
        r"""Generator that runs the solver with the given stopping rule. This is a convenience method
        that implements a simple generator loop running the solver until it either converges or the
        stopping rule triggers.

        Parameters
        ----------
        stoprule : regpy.stoprules.StopRule, optional
            The stopping rule to be used. If omitted, stopping will only be
            based on the return value of `next`.

        Yields
        ------
        tuple of arrays
            The (x, y) pair of the current iteration, or the solution chosen by
            the stopping rule.
        """
        self.check_for_duality_stoprule(stoprule)
        if hasattr(self,"compute_dual") and self.compute_dual and hasattr(self,"_compute_dual"):
            self._compute_dual()
        while not stoprule.stop(self.x,self.y,getattr(self,"dual",None)) and self.next(): 
            yield self.x, self.y
        self.log.info('Solver converged after {} iteration.'.format(self.iteration_step_nr))
 


    def until(self, stoprule=NoneRule()):
        r"""Generator that runs the solver with the given stopping rule. This is a convenience method
        that implements a simple generator loop running the solver until it either converges or the
        stopping rule triggers.

        Parameters
        ----------
        stoprule : regpy.stoprules.StopRule, optional
            The stopping rule to be used. If omitted, stopping will only be
            based on the return value of `next`.

        Yields
        ------
        tuple of arrays
            The (x, y) pair of the current iteration, or the solution chosen by
            the stopping rule.
        """
        self.next()
        yield self.x, self.y
        self.check_for_duality_stoprule(stoprule)
        if hasattr(self,"compute_dual") and self.compute_dual and hasattr(self,"_compute_dual"):
            self._compute_dual()
        while not stoprule.stop(self.x,self.y,getattr(self,"dual",None)) and self.next(): 
            yield self.x, self.y

        self.log.info('Solver converged after {} iteration.'.format(self.iteration_step_nr))

    def run(self, stoprule=NoneRule()):
        r"""Run the solver with the given stopping rule. This method simply runs the generator
        `regpy.solvers.Solver.while_` and returns the final `(x, y)` pair.
        """
        for x, y in self.while_(stoprule):
            pass
        if not 'x' in locals() or not 'y' in locals(): 
            # This happens if the stopping criterion is satisfied for the initial guess.
            x = self.x
            y = self.y
        return x, y
    
    def check_for_duality_stoprule(self,stoprule) -> None:
        if not hasattr(self,"compute_dual") or not self.compute_dual:
            if isinstance(stoprule,DualityGapStopping):
                self.compute_dual = True
            elif isinstance(stoprule,CombineRules):
                for rule in stoprule.rules:
                    self.check_for_duality_stoprule(rule)


class RegSolver(Solver):
    r"""Abstract base class for solvers working with a regularization setting. Solvers do not 
    implement loops themselves, but are driven by repeatedly calling the `next` method. They 
    expose the current iterate stored in and value as attributes `x` and `y`, and can be iterated 
    over, yielding the `(x, y)` tuple on every iteration (which may or may not be the same 
    arrays as before, modified in-place).

    There are some convenience methods to run the solver with a `regpy.stoprules.StopRule`.

    Subclasses should override the method `_next(self)` to perform a single iteration where the values of 
    the attributes `x` and `y` are updated. The main difference to `next` is that `_next` does not have a
    return value. If the solver converged, `converge` should be called, afterwards `_next` will never be
    called again. Most solvers will probably never converge on their own, but rely on the caller or a
    `regpy.stoprules.StopRule` for termination.

    Parameters
    ----------
    setting: RegularizationSetting
        RegularizationSetting used for solver
    x : numpy.ndarray
        Initial argument for iteration. Defaults to None.
    y : numpy.ndarray
        Initial value at current iterate. Defaults to None.
    """

    def __init__(self,setting,x=None,y=None):
        if not isinstance(setting,RegularizationSetting):
            raise TypeError(Errors.not_instance(setting,RegularizationSetting))
        self.op=setting.op
        """The operator."""
        self.penalty = setting.penalty
        """The penalty functional."""
        self.data_fid = setting.data_fid
        """The data misfit functional."""
        self.h_domain = setting.h_domain
        """The Hilbert space associated to penalty functional"""
        self.h_codomain =  setting.h_codomain
        """The Hilbert space associated to data fidelity functional"""
        self.setting = setting
        """The regularization setting"""
        if isinstance(setting,TikhonovRegularizationSetting):
            self.regpar = setting.regpar
            """The regularization parameter"""
        super().__init__(x,y)

    def runWithDP(self,data,delta=0, tau=2.1, max_its = 1000):
        r"""
        Run solver with Morozov's discrepancy principle as stopping rule.

        Parameters
        ----------
        data: array-like
            The right-hand side
        delta: float, default:0
            noise level
        tau: float, default: 2.1
            parameter in discrepancy principle
        max_its: int, default: 1000
            maximal number of iterations
        """
        from regpy.stoprules import CountIterations, Discrepancy
        stoprule =  (CountIterations(max_iterations=max_its)
                        + Discrepancy(self.h_codomain.norm, data,
                        noiselevel=delta, tau=tau)
                    )
        reco, reco_data = self.run(stoprule)
        if not isinstance(stoprule.active_rule, Discrepancy):
            self.log.warning('Discrepancy principle not satisfied after maximum number of iterations.')
        return reco, reco_data


class RegularizationSetting:
    r"""A Regularization *setting* for an inverse problem, used by solvers. A
    setting consists of

    - a forward operator,
    - a penalty functional with an associated Hilbert space structure to measure the error, and
    - a data fidelity functional with an associated Hilbert space structure to measure the data misfit.

    This class is mostly a container that keeps all of this data in one place and makes sure that
    the the used penalty and data fidelity have matching domains `regpy.hilbert.HilbertSpace.vecsp`\s 
    with the operator's domain and codomain.

    It also handles the case when the specified data fidelity or penalty is a Hilbert space which constructs 
    the associated squared Hilbert norm functionals. It also handles cases when `regpy.hilbert.AbstractSpace` 
    or `AbstractFunctional`\s (or actually any callable) instead of a `regpy.functionals.Functional`, calling 
    it on the operator's domain or codomain to construct the concrete `Functional`'s instances.

    Parameters
    ----------
    op : regpy.operators.Operator
        The forward operator.
    penalty : regpy.functionals.Functional or regpy.hilbert.HilbertSpace or callable
        The penalty functional.
    data_fid : regpy.functionals.Functional or regpy.hilbert.HilbertSpace or callable
        The data misfit functional.
    """

    log = ClassLogger()

    def __init__(self, op, penalty, data_fid):
        if not isinstance(op,Operator):
            raise TypeError(Errors.not_instance(op,Operator,add_info="Regularization Setting requires op to be a RegPy operator."))
        self.op = op
        """The operator."""
        self.penalty = as_functional(penalty, op.domain)
        """The penalty functional."""
        self.data_fid = as_functional(data_fid, op.codomain)
        """The data fidelity functional."""
        self.h_domain = self.penalty.h_domain
        """The Hilbert space associated to penalty functional"""
        self.h_codomain =  self.data_fid.h_domain if not isinstance(self.data_fid,Composed) else self.data_fid.func.h_domain
        """The Hilbert space associated to data fidelity functional"""

    def check_adjoint(self,test_real_adjoint=False,tolerance=1e-10):
        r"""Convenience method to run `regpy.util.operator_tests`. Which test if the provided adjoint in the operator 
        is the true matrix adjoint. That is 

        .. code-block:: python
    
           (vec_typ.vdot(y, self.op(x)) - vec_typ.vdot(self.op.adjoint(y), x)).real < tolerance

        If the operator is non-linear this will be done for the derivative.

        Parameters
        ----------
        tolerance : float
            Tolerance of the two computed inner products.

        Returns
        -------
        bool
            Tests either the operator or the derivative with `regpy.util.operator_tests.test_adjoint` and returns that value. 
        """
        from regpy.util.operator_tests import test_adjoint
        if self.op.linear:
            return test_adjoint(self.op,tolerance=tolerance)
        else:
            _, deriv = self.op.linearize(self.op.domain.randn())
            return test_adjoint(deriv, tolerance=tolerance)

    def check_deriv(self,steps=None):
        r"""Convenience method to run `regpy.util.operator_tests.test_derivative`. Which test if the 
        provided derivative in the operator ,if it is a non-linear operator. It computes for 
        the provided `steps` as :math:`t`

        .. math::
            ||\frac{F(x+tv)-F(x)}{t}-F'(x)v|| 

        wrt the :math:`L^2`-norm and returns true if it is a decreasing sequence.

        Parameters
        ----------
        steps : list, optional
            A decreasing sequence used as steps. Defaults to (Default: [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]).

        Returns
        -------
        Boolean
            True if the operator is linear or affine linear or if test_derivative returns True.
        """
        from regpy.util.operator_tests import test_derivative, test_affine_linearity
        if self.op.linear or test_affine_linearity(self.op):
            return True
        return test_derivative(self.op,steps=steps)
    
    def h_adjoint(self,y=None):
        r"""Returns the adjoint with respect ro the Hilbert spaces by implementing :math:`G_X^{-1} \circ F \circ G_Y`.

        If the operator is non-linear this provided the adjoint to the derivative at `y`.

        Parameters
        ----------
        y : op.codomain
            Element of the domain at which to evaluate the adjoint of the derivative. 

        Returns
        -------
        regpy.operators.Operator
            Adjoint wrt chosen Hilbert spaces. 
        regpy.operators.Operator
            The operator who's adjoint is computed. Only needed for non-linear case as this return the 
            derivative at the point.
        """
        if self.op.linear:
            return self.h_domain.gram_inv * self.op.adjoint * self.h_codomain.gram, self.op
        else:
            _ , deriv = self.op.linearize(y)
            return self.h_domain.gram_inv * deriv.adjoint * self.h_codomain.gram, deriv
        
    def is_hilbert_setting(self):
        r"""Assert if the setting is a Hilbert space setting. 

        Returns
        -------
        Boolean
            True if both `penalty` and `data_fid` are `SquaredNorm` functionals. 
        """
        return isinstance(self.penalty,SquaredNorm) and isinstance(self.data_fid,SquaredNorm)
        

class TikhonovRegularizationSetting(RegularizationSetting):
    r"""Tikhonov regularization setting for minimizing a Tikhonov functional 

    .. math::
        \frac{1}{\alpha}\mathcal{S}_{g^{\delta}}(Tf) + \mathcal{R}(f) = \min!

    In contrast to RegularizationSetting, the regularization parameter is fixed, 
    the data fidelity functional :math:`\mathcal{S}=self.data_fid` incorporates the data :math:`g^{\delta}` of the inverse problem, 
    and the penalty term :math:`\mathcal{R}` incorporates a potential initial guess.

    Parameters
    ----------
    op : regpy.operators.Operator
        The forward operator.
    penalty : regpy.functionals.Functional
        The penalty functional :math:`\mathcal{R}`.
    data_fid : regpy.functionals.Functional
        The data misfit functional :math:`\mathcal{S}_{g^{\delta}}`.
    regpar: float [default: 1]
        regularization parameter
    penalty_shift: op.domain [default: None]
        If not None, the penalty functional is replaced by penalty(. - penalty_shift).
    data_fid_shift: op.co_domain [default: None]
        If not None, the data fidelity functional is replaced by data_fid(. - data_fid_shift).
    primal_setting: None or TikhonovRegularizationSetting [default:None]
        Indicates whether or not a setting serves as primal setting. For a primal setting, primal_setting is None, for a dual setting it is the primal setting. 
        This affects the duality relations and the duality gap. 
    logging_level: int [default: INFO]
        logging level
    """

    def __init__(self, op, penalty, data_fid,regpar=1.,penalty_shift= None, data_fid_shift= None, 
                 primal_setting=None,logging_level = "INFO",gap_threshold = 1e5):
        super().__init__(op,penalty=penalty, data_fid= data_fid)

        if not penalty_shift is None:
            self.penalty_shift = penalty_shift
            self.penalty = self.penalty.shift(penalty_shift)
        else:
            self.penalty_shift = None
        
        if not data_fid_shift is None:
            self.data_fid_shift = data_fid_shift
            self.data_fid = self.data_fid.shift(data_fid_shift)
        else:
            self.data_fid_shift = None

        if not isinstance(regpar,(float,int)):
            raise TypeError(Errors.type_error("The regularization parameter need to be a scalar"))
        if regpar <= 0:
            raise ValueError(Errors.value_error("The regularization parameter need to be a positive scalar"))
        self.regpar = float(regpar)
        self.log.setLevel(logging_level)
        self.gap_threshold = gap_threshold
        """The regularization parameter"""
        if primal_setting is not None and not isinstance(primal_setting,TikhonovRegularizationSetting):
            raise TypeError(Errors.type_error(f"The primal_setting needs to be either None or of {type(self)}!"))
        self.primal_setting = primal_setting

        self.determine_methods()
    
    def dualSetting(self):
        r"""Yields the setting of the dual optimization problem

        .. math::
           \mathcal{R}^*(\T^*p) + \frac{1}{\alpha}\mathcal{S}^*(- \alpha p) = \min!

        """
        if not self.op.linear:
            raise RuntimeError(Errors.not_linear_op(self.op,add_info="To properly construct a dual setting the operator needs to be linear!"))
        return TikhonovRegularizationSetting(
            self.op.adjoint,
            self.data_fid.conj.dilation(-self.regpar),
            self.penalty.conj,
            regpar= 1/self.regpar,
            primal_setting = self,
            logging_level=self.log.level
        )

    def dualToPrimal(self,pstar,argumentIsOperatorImage = False, own= False):
        r""" Returns an element of :math:`\partial \mathcal{R}^*(T^*p)` 
        If :math:`p` is a solution to the dual problem and :math:`\partial\mathcal{R}^*` is a singleton, this yields a solution to the primal problem. 
        If :math:`\xi=T^*p` is already known, the option `argumentIsOperatorImage=True' can be used to pass :math:`\xi` as argument and avoid an operator evaluation.
                
        Parameters
        ----------
        pstar: self.op.codomain (or self.op.domain if argumentIsOperatorImage=True)
            argument to be transformed
        argumentIsOperatorImage: boolean [default: False]
            See above.
        own: bool [default: False]
            Only relevant for dual settings. If False, the duality relations of the primal setting are used. 
            If true, the duality relations of the dual setting are used. 
        """
        if self.primal_setting is None or own == True:
            if argumentIsOperatorImage:
                return self.penalty.conj.subgradient(pstar)
            else:
                if not self.op.linear:
                    raise RuntimeError(Errors.not_linear_op(self.op,add_info="To construct a primal solution from the dual in case using the adjoint only allowed for linear operators!"))
                return self.penalty.conj.subgradient(self.op.adjoint(pstar))
        else:
            return self.primal_setting.primalToDual(-self.regpar*pstar, argumentIsOperatorImage= argumentIsOperatorImage)
            """Note that the dual variables of the dual problem differ by a factor -alpha_d from the primal variables of the primal problem.
            Here alpha_d=1/alpha_p is the regularization parameter of the dual problem, and alpha_p the regularization parameter of the primal problem.
            """
        
    def primalToDual(self,x,argumentIsOperatorImage = False, own=False):
        r"""
        Returns an element of :math:`(-1/\alpha) \partial \mathcal{S}(Tx)` 
        If :math:`x` is a solution to the primal problem and :math:`\partial \mathcal{S}` is a singleton, this 
        yields a solution to the dual problem. If :math:`\y=Tx` is already known, 
        the option `argumentIsOperatorImage=True' can be used to pass :math:`\y` as argument and avoid an operator evaluation.
    
        Parameters
        ----------------------------
        x: self.op.domain (or self.op.codomain if argumentIsOperatorImage=True)
            argument to be transformed
        argumentIsOperatorImage: boolean [default: False]
            See above.
        own: bool [default: False]
            Only relevant for dual settings. If False, the duality relations of the primal setting are used. 
            If true, the duality relations of the dual setting are used. 
        """
        if self.primal_setting is None or own==True:
            if argumentIsOperatorImage:
                return (-1./self.regpar) * self.data_fid.subgradient(x)
            else:
                return (-1./self.regpar) * self.data_fid.subgradient(self.op(x))
        else:
            return self.primal_setting.dualToPrimal(x, argumentIsOperatorImage=argumentIsOperatorImage)

    def dualityGap(self, primal=None, dual=None):
        r"""Computes the value of the duality gap 
        
        .. math::
            \frac{1}{\alpha}\mathcal{S}_{g^{\delta}}(Tf) + \mathcal{R}(f) - \frac{1}{\alpha} }\mathcal{S}_{g^{\delta}}^*(-\alpha p) - \mathcal{R}^*(T^*p)

        Parameters
        ----------
        primal: setting.op.domain [default: None]
            primal variable f
        dual: setting.op.codomain [default: None]
            dual variable p        
        """
        if not self.op.linear:
            raise RuntimeError(Errors.not_linear_op(self.op,add_info="The duality gap can only be computed for settings with linear operators!"))
        if primal is None and dual is None:
            raise ValueError(Errors.value_error("Either a primal or dual vector need to be given to compute the duality gap!"))
        if primal is None:
            f = self.dualToPrimal(dual)
        else:
            f = primal
        if dual is None:
            p = self.primalToDual(primal)
        else:
            p = dual
        alpha = self.regpar

        dat = 1./alpha * self.data_fid(self.op(f))
        pen = self.penalty(f)
        ddat = self.penalty.conj(self.op.adjoint(p))
        dpen = 1./alpha * self.data_fid.conj(-alpha*p)
        ares = ma.fabs(dat)+ma.fabs(pen)+ma.fabs(ddat)+ma.fabs(dpen) 
        if not ma.isfinite(ares):
            self.log.warning('duality gap infinite: R(..)={:.3e}, S(..)={:.3e}, S*(..)={:.3e}, R*(..)={:.3e}'.format(pen,dat,dpen,ddat))
            return ma.inf
        res = dat+pen+ddat+dpen
        if ares/res>1e10:
            self.log.warning('estimated loss of rel. accuracy in duality gap by cancellation: {:.3e}'.format(ares/res))
        elif ares/res>self.gap_threshold:
            self.log.debug('estimated loss of rel. accuracy in duality gap by cancellation: {:.3e}'.format(ares/res))
        return res
    
    def isSaddlePoint(self,x,p,tol):
        r"""Checks if \((x,p) )\ is a saddle point of \(<Tx,p> + \mathcal{R}(f)-\frac{1}{\alpha}\mathcal{S}^*(\alpha p) )\
        or equivalently (in case of strong duality)
        - if x is a solution to the primal problem and p a solution of the dual problem (up to a given tolerance)
        - if 
        .. math::
        Tx \in \partial \mathcal{S}^*(\alpha p), \qquad -T^*p \in \partial \mathcal{R}(f).


        Parameters
        ---------------------------
        x: self.op.domain
        Candidate solution of primal problem.
        p: self.op.codomain
        Candidate solution of dual problem.
        tol: float [default: 1e-10]
        Tolerance value
        """
        if not self.op.linear:
            raise RuntimeError(Errors.not_linear_op(self.op,add_info="To determine if on a saddle point the setting need to be with linear operators!"))
        return self.data_fid.conj.is_subgradient(self.op(x),self.regpar*p,tol=tol) and \
               self.penalty.is_subgradient(-self.op.adjoint(p),x,tol=tol) 

    def determine_methods(self):
        from regpy.solvers.linear import ForwardBackwardSplitting,FISTA,PDHG,ADMM,AMA,semismoothNewton,TikhonovCG
        self._methods = {
            'FB': {'class':ForwardBackwardSplitting, 'primal': True},
            'dual_FB': {'class':ForwardBackwardSplitting, 'primal': False},
            'FISTA': {'class':FISTA, 'primal': True},
            'dual_FISTA': {'class':FISTA, 'primal': False},
            'PDHG': {'class':PDHG, 'primal': True},
            'dual_PDHG': {'class':PDHG, 'primal': False},
            'ADMM': {'class':ADMM, 'primal': True},
            'AMA': {'class':AMA, 'primal': True},
            'SSNewton': {'class':semismoothNewton, 'primal': True},
            'dual_SSNewton': {'class':semismoothNewton, 'primal': False}, 
            'CG': {'class':TikhonovCG, 'primal':True},
            'dual_CG': {'class':TikhonovCG, 'primal':False}
        }
        self._methods['FB']['applicable'] =  ('proximal' in self.penalty.methods and self.data_fid.Lipschitz<inf)
        self._methods['FISTA']['applicable'] = ('proximal' in self.penalty.methods and self.data_fid.Lipschitz<inf)
        
        self._methods['dual_FB']['applicable'] = ('proximal' in self.data_fid.conj.methods and 'subgradient' in self.penalty.conj.methods
                                    and self.penalty.conj.Lipschitz<inf)
        self._methods['dual_FISTA']['applicable'] = ('proximal' in self.data_fid.conj.methods and 'subgradient' in self.penalty.conj.methods 
                                       and self.penalty.conj.Lipschitz<inf)

        self._methods['PDHG']['applicable'] = ('proximal' in self.penalty.methods and 'proximal' in self.data_fid.conj.methods)
        self._methods['dual_PDHG']['applicable'] = ('proximal' in self.penalty.conj.methods and 'subgradient' in self.penalty.conj.methods
                                      and 'proximal' in self.data_fid.methods)

        self._methods['ADMM']['applicable'] = ('proximal' in self.penalty.methods and 'proximal' in self.data_fid.methods)
        self._methods['AMA']['applicable'] = ('subgradient' in self.penalty.conj.methods and 'proximal' in self.data_fid.methods)

        self._methods['SSNewton']['applicable'] = (isinstance(self.penalty,(QuadraticNonneg, QuadraticBilateralConstraints)) 
                                     and isinstance(self.data_fid,SquaredNorm))
        self._methods['dual_SSNewton']['applicable'] = (isinstance(self.data_fid.conj,(QuadraticNonneg, QuadraticBilateralConstraints)) 
                                     and isinstance(self.penalty,SquaredNorm))

        self._methods['CG']['applicable'] = isinstance(self.penalty,SquaredNorm) and isinstance(self.data_fid,SquaredNorm)
        self._methods['dual_CG']['applicable'] = isinstance(self.penalty,SquaredNorm) and isinstance(self.data_fid,SquaredNorm)

    def run(self,method = 'FISTA',**kwargs):
        if not method in self._methods:
            raise ValueError('Unknown method')
        themethod= self._methods[method]
        if themethod['applicable'] == False:
            raise RuntimeError(f'{method} is not applicable in this setting.')

        thesetting = self if themethod['primal'] else self.dualSetting()
        if 'stoprule' not in themethod or themethod['stoprule'] is None:
            themethod['stoprule'] = DualityGapStopping(thesetting,threshold = 1.,logging_level=logging.INFO) + CountIterations(max_iterations=1000)

        
        solver = themethod['class'](thesetting,**kwargs)
        x,y = solver.run(themethod['stoprule'])
        
        if themethod['primal']==False:
            x_star,y_star = x,y
            x = self.dualToPrimal(y_star,argumentIsOperatorImage=True)
            y = self.op(x)
        return x,y

        