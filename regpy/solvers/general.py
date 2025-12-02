import math as ma
import numpy as np
from scipy.sparse.linalg import eigsh


from regpy.util import ClassLogger, Errors
from regpy.util.operator_tests import test_derivative
from regpy.operators import Operator
from regpy.functionals.base import  as_functional, Composed
from regpy.functionals import SquaredNorm, QuadraticLowerBound, QuadraticNonneg, QuadraticBilateralConstraints
from regpy.stoprules import StopRule,NoneRule,DualityGapStopping,CombineRules,CountIterations
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
    setting: Setting
        Setting used for solver
    x : numpy.ndarray
        Initial argument for iteration. Defaults to None.
    y : numpy.ndarray
        Initial value at current iterate. Defaults to None.
    """

    def __init__(self,setting,x=None,y=None):
        if not isinstance(setting,Setting):
            raise TypeError(Errors.not_instance(setting,Setting))
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
        if setting.is_tikhonov:
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

        

class Setting:
    r"""A *setting* for an inverse problem, used by solvers. A
    setting always consists at least of

    - a forward operator,
    - a penalty functional with an associated Hilbert space structure to measure the error, and
    - a data fidelity functional with an associated Hilbert space structure to measure the data misfit.

    If a regularization parameter is given this is the setting for the minimization problem 

    .. math::
        \frac{1}{\alpha}\mathcal{S}_{g^{\delta}}(Tf) + \mathcal{R}(f) = \min!

    If the operator is linear and both functionals are convex this is, more generally, the setting of Rockafellar-Fenchel duality, 
    which involves a rich and algorithmically useful mathematical structure. In this case, the dual setting 
    and primal-dual optimality conditions are provided. 
    This class is mostly a container that keeps all of this data in one place and makes sure that all initializations are 
    done correctly.

    It also handles the case when the specified data fidelity or penalty is a Hilbert space which constructs 
    the associated squared Hilbert norm functionals. It also handles cases when `regpy.hilbert.AbstractSpace` 
    or `AbstractFunctional`\s (or actually any callable) instead of a `regpy.functionals.Functional`, calling 
    it on the operator's domain or codomain to construct the concrete `Functional`'s instances.
    """
    log = ClassLogger()

    def __init__(self, op, penalty, data_fid,regpar=None,penalty_shift= None, data_fid_shift= None,
                 logging_level = "INFO",primal_setting=None,gap_threshold = 1e5):
        if not isinstance(op,Operator):
            raise TypeError(Errors.not_instance(op,Operator,add_info="Setting requires op to be a RegPy operator."))
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
        self.regpar=regpar#The flags are set by setting the regularization parameter
        """The Regularization parameter"""
        self.log.setLevel(logging_level)
        self.gap_threshold = gap_threshold
        if primal_setting is not None and not (primal_setting.is_convex and primal_setting.is_tikhonov):
            raise ValueError(Errors.value_error("The primal_setting needs to be convex and contain a regularization parameter!"))
        self.primal_setting = primal_setting
        if primal_setting is None and self.is_convex and self.is_tikhonov:
            self._methods = Setting._generate_full_solver_dictionary()


    def _set_flags(self):
        self.is_tikhonov=(self.regpar is not None)
        """True if a regularization parameter is set"""
        self.is_convex=self.op.linear and self.penalty.convex and self.data_fid.convex
        """True if the operator is linear"""
        self.is_hilbert=(isinstance(self.penalty,SquaredNorm) and isinstance(self.data_fid,SquaredNorm))
        """Ture if penalty and data fidelity are both squared norms"""

    @property
    def regpar(self):
        return self._regpar

    @regpar.setter
    def regpar(self,new_regpar):
        if(new_regpar is not None):
            if not isinstance(new_regpar,(float,int)):
                raise TypeError(Errors.type_error("The regularization parameter need to be a scalar"))
            if new_regpar <= 0:
                raise ValueError(Errors.value_error("The regularization parameter need to be a positive scalar"))
            new_regpar = float(new_regpar)
        self._regpar=new_regpar
        self._set_flags()

    ######General convenience methods
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

    ######Methods exploiting duality
    def get_dual_setting(self):
        r"""Yields the setting of the dual optimization problem

        .. math::
           \mathcal{R}^*(\T^*p) + \frac{1}{\alpha}\mathcal{S}^*(- \alpha p) = \min!

        """
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for the computation of a dual setting."))
        if(not self.is_convex):
            raise RuntimeError(Errors.generic_message("The setting has to be convex for the computation of a dual setting."))

        return Setting(
            self.op.adjoint,
            self.data_fid.conj.dilation(-self.regpar),
            self.penalty.conj,
            regpar= 1/self.regpar,
            primal_setting = self,
            logging_level=self.log.level
        )

    def dual_to_primal(self,pstar,argumentIsOperatorImage = False, own= False):
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
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for the computation of a dual primal mapping."))
        if(not self.is_convex):
            raise RuntimeError(Errors.generic_message("The setting has to be convex for the computation of a dual primal mapping."))
        if self.primal_setting is None or own == True:
            if argumentIsOperatorImage:
                return self.penalty.conj.subgradient(pstar)
            else:
                return self.penalty.conj.subgradient(self.op.adjoint(pstar))
        else:
            return self.primal_setting.primal_to_dual(-self.regpar*pstar, argumentIsOperatorImage= argumentIsOperatorImage)
            """Note that the dual variables of the dual problem differ by a factor -alpha_d from the primal variables of the primal problem.
            Here alpha_d=1/alpha_p is the regularization parameter of the dual problem, and alpha_p the regularization parameter of the primal problem.
            """
        
    def primal_to_dual(self,x,argumentIsOperatorImage = False, own=False):
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
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for the computation of a primal dual mapping."))
        if(not self.is_convex):
            raise RuntimeError(Errors.generic_message("The setting has to be convex for the computation of a primal dual mapping."))
        if self.primal_setting is None or own==True:
            if argumentIsOperatorImage:
                return (-1./self.regpar) * self.data_fid.subgradient(x)
            else:
                return (-1./self.regpar) * self.data_fid.subgradient(self.op(x))
        else:
            return self.primal_setting.dual_to_primal(x, argumentIsOperatorImage=argumentIsOperatorImage)

    def duality_gap(self, primal=None, dual=None):
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
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for the computation of the duality gap."))
        if not self.is_convex:
            raise RuntimeError(Errors.not_linear_op(self.op,add_info="The duality gap can only be computed for convex settings with linear operators!"))
        if primal is None and dual is None:
            raise ValueError(Errors.value_error("Either a primal or dual vector need to be given to compute the duality gap!"))
        if primal is None:
            f = self.dual_to_primal(dual)
        else:
            f = primal
        if dual is None:
            p = self.primal_to_dual(primal)
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
    
    def is_saddle_point(self,x,p,tol):
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
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for this check."))
        if not self.is_convex:
            raise RuntimeError(Errors.not_linear_op(self.op,add_info="This check requires a convex setting with a linear operator!"))
        return self.data_fid.conj.is_subgradient(self.op(x),self.regpar*p,tol=tol) and \
               self.penalty.is_subgradient(-self.op.adjoint(p),x,tol=tol) 



    
    ######Methods checking applicability
    @staticmethod
    def _generate_full_solver_dictionary():
        '''This so far contains only linear solvers'''
        from regpy.solvers.linear import ForwardBackwardSplitting,FISTA,PDHG,ADMM,AMA,SemismoothNewton_bilateral,TikhonovCG
        method_dict={
                'FB': {'class':ForwardBackwardSplitting, 'primal': True, 'full':'Forward Backward Splitting applied to primal problem'},
                'dual_FB': {'class':ForwardBackwardSplitting, 'primal': False, 'full': 'Forward Backward Splitting applied to primal problem'},
                'FISTA': {'class':FISTA, 'primal': True, 'full': 'Fast Iterative Thresholding applied to primal problem'}, 
                'dual_FISTA': {'class':FISTA, 'primal': False, 'full': 'Fast Iterative Thresholding applied to dual problem'},
                'PDHG': {'class':PDHG, 'primal': True, 'full': 'Primal-Dual Hybrid Gradient Method applied to primal problem'},
                'dual_PDHG': {'class':PDHG, 'primal': False, 'full': 'Primal-Dual Hybrid Gradient Method applied to dual problem'},
                'ADMM': {'class':ADMM, 'primal': True, 'full': 'Alternating Direction Method of Mulpliers' },
                'AMA': {'class':AMA, 'primal': True, 'full': 'Alternating Minimization Algorithm'},   
                'SSNewton': {'class':SemismoothNewton_bilateral, 'primal': True, 'full': 'Semismooth Newton method'},
                'dual_SSNewton': {'class':SemismoothNewton_bilateral, 'primal': False, 'full': 'Semismooth Newton method applied to dual problem'}
            }
        return method_dict


    
    def evaluate_methods(self,method_names = None):
        """Evaluates which methods are applicable to the current Setting. 
        This is achieved by calling method.check_applicability(self), which also provide information on guaranteed rates.

        Parameters:
        method_names: List of strings or None [default:None]
            List of names of methods to be evaluated. If None, all methods are evaluated.   
        """
        if not (self.primal_setting is None and self.is_convex and self.is_tikhonov):
            raise NotImplementedError(Errors.generic_message("Applicable methods so far can only be computed for convex settings with regularization parameter."))
        if method_names is None:
            method_names = self._methods.keys()
        else:
            for method_name in method_names:
                if not method_name in self._methods:
                    raise ValueError(f'Unknown method name {method_name}. Known methods are {self._methods.keys()}.')
        if len(method_names)>0:
            op_norm = self.op.norm()
        for method_name in method_names:
            method = self._methods[method_name]
            out,_ = method['class'].check_applicability(self if method['primal'] else self.get_dual_setting(),op_norm=op_norm)
            if not method['primal'] and not 'subgradient' in self.penalty.conj.methods:
                method['info'] = ('' if out['applicable'] else out['info']) + 'Missing subgradient of conjugate penalty.'
                method['applicable'] = False
            else:
                method['applicable'] = out['applicable']
                method['info'] = out['info']
                if out['applicable']:
                    method['rate'] = out['rate']

    def applicable_methods(self):
        """Yields subdictionary of the methods that can be applied to the given Tikhonov functional.
        """
        if not (self.primal_setting is None and self.is_convex and self.is_tikhonov):
            raise NotImplementedError(Errors.generic_message("Applicable methods so far can only be computed for primal convex settings with regularization parameter."))
        if any('applicable' not in self._methods[name] for name in self._methods.keys()):
            self.evaluate_methods()
        return {name:method for name, method in self._methods.items() if method['applicable']}
        
    def display_all_methods(self,full_names=True):
        """
        Displays all the methods for minimizing Tikhonov functionals together with information 
        on their applicability to the given Tikhonov functional. 
        """
        self.evaluate_methods()
        print('Applicable methods:\n')
        for name,method in self.applicable_methods().items():
            print(name, (' ('+method['full']+'): ' if full_names else ''),
                  method['info'],'linear rate: {:.3e}'.format(method['rate']))
        print('\n Non-applicable methods:\n')
        for name,method in self._methods.items(): 
            if method['applicable']==False:
                print(name, (' ('+method['full']+'): ' if full_names else ''),
                      method['info'])

    def select_best_method(self):
        """Returns the name of the applicable method with the best convergence rate predicted by theory 
        and the convexity and Lipschitz parameters of the data and penalty functional.
        (Since comparisons of first and second order methods are difficult, we only choose among first 
        order methods, and to achieve this, we set convergence rates of second order method >1.)
        """
        d = self.applicable_methods()
        best_method_name = min(d, key=lambda name: np.abs(d[name]['rate']))
        if isinstance(d[best_method_name]['rate'],int):
            best_method_name = min(d, key=lambda name: np.abs(d[name]['rate']))
        self.log.info('Choose '+best_method_name+' as best method.')
        return best_method_name

    def set_stopping_rule(self,method_name,rule):
        """Sets a StopRule for an optimization method.
        Parameters:
        method_name: string 
            key of the method
        rule: StopRule
            the stopping rule
        """
        if not isinstance(rule,StopRule):
            raise TypeError(f"rule must be of class StopRule. Got{rule}.")
        if method_name not in self._methods.keys():
            raise ValueError(f"{method_name} is unknown method key.")
        self._methods[method_name]['stoprule'] = rule

    def get_stopping_rule(self,method_name):
        """Retrieves a stopping rule that has run an optimization method 
        (e.g. to view statistics or (intermediate) solutions)
        Parameters:
        method_name: string
            Key of the method
        Returns:
        StopRule
        """
        if not method_name in self._methods.keys():
            raise ValueError(f"{method_name} is unknown method key.")
        if 'stoprule' not in self._methods[method_name]:
            raise RuntimeError(f'Method {method_name} has not StopRule.')
        else:
            return self._methods[method_name]['stoprule']   

    def run(self,method_name = None,**kwargs):
        """Runs a given method to minimize the Tikhonov functional.
        
        Parameters:
        method_name: string or None [default: None] 
            Key of the method to be run in the methods dictionary self._methods (can be displayed by display_all_methods())
            If None the "best" method is selected by select_best_method().
        **kwargs: dict
            Arguments to be passed to the method.

        Returns:
            x,y: x is the minimizer of the Tikhonov functional and y its value under the operator.         
        """
        if method_name is None:
            method_name = self.select_best_method()
        if not method_name in self._methods:
            raise ValueError('Unknown method name')
        themethod= self._methods[method_name]
        if not 'applicable' in themethod:
            self.evaluate_methods(themethod) 
        if themethod['applicable'] == False:
            raise RuntimeError(f'{method_name} is not applicable in this setting.')

        thesetting = self if themethod['primal'] else self.get_dual_setting()
        if 'stoprule' not in themethod or themethod['stoprule'] is None:
            self.set_stopping_rule(method_name, DualityGapStopping(thesetting,threshold = 0.1,logging_level=logging.INFO) + CountIterations(max_iterations=1000))

        
        solver = themethod['class'](thesetting,**kwargs)
        x,y = solver.run(themethod['stoprule'])
        
        if themethod['primal']==False:
            x_star,y_star = x,y
            x = self.dual_to_primal(y_star,argumentIsOperatorImage=True)
            y = self.op(x)
        return x,y