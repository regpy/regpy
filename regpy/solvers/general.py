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

    def is_converged(self):
        return self.__converged

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
        stoprule._complete_init_with_solver(self)
        while not stoprule.stop() and self.next(): 
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
        stoprule._complete_init_with_solver(self)
        self.next()
        yield self.x, self.y
        while not stoprule.stop() and self.next(): 
            yield self.x, self.y

        self.log.info('Solver converged after {} iteration.'.format(self.iteration_step_nr))

    def run(self, stoprule=NoneRule()):
        r"""Run the solver with the given stopping rule. This method simply runs the generator
        `regpy.solvers.Solver.while_` and returns the final `(x, y)` pair.
        """
        for x, y in self.while_(stoprule):
            pass
        #f not 'x' in locals(): 
        #    # This happens if the stopping criterion is satisfied for the initial guess.
        #    x = self.x
        #    y = self.y
        if stoprule.best_iterate() is None:
            self.log.info(f"Could not find a bet iterate with stop rule {stoprule}!")
            return self.x, self.y
        return stoprule.best_iterate()
    


    


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
    
    def compute_dual(self):
        """computes dual and primal components. This is a generic implementation that works for settings that are tikhonov.
        This should be reimplemented if the solver can compute the variables more effectively.
        """
        if not self.setting.is_tikhonov:
            raise RuntimeError(Errors.generic_message("It is not possible to compute the dual in the implementation of this setting"))
        self.primal,self.dual = self.setting._complete_primal_dual_tuples((self.x,self.y),self.dual)

        

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

    Parameters
    ----------
    op : regpy.operators.Operator
        The forward operator.
    penalty : regpy.functionals.Functional or regpy.hilbert.HilbertSpace or callable
        The penalty functional.
    data_fid : regpy.functionals.Functional or regpy.hilbert.HilbertSpace or callable
        The data misfit functional.
    regpar: float [default: None]
        regularization parameter
    penalty_shift: op.domain [default: None]
        If not None, the penalty functional is replaced by penalty(. - penalty_shift).
    data: op.co_domain [default: None]
        If not None, the data in the data fidelity functional is replaced by data.
    primal_setting: None or TikhonovRegularizationSetting [default:None]
        Indicates whether or not a setting serves as primal setting. For a primal setting, primal_setting is None, for a dual setting it is the primal setting. 
        This affects the duality relations and the duality gap. 
    gap_threshold: float [default: 1e5]
    logging_level: int [default: logging.INFO]
        logging level
    """

    
    
    log = ClassLogger()

    def __init__(self, op, penalty, data_fid,regpar=None,penalty_shift= None, data= None,primal_setting=None,gap_threshold = 1e5,
                 logging_level = "INFO"):
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
        self.regpar=regpar#The flags are set by setting the regularization parameter
        """The Regularization parameter"""
        if(not self.data_fid.is_data_func and data is None and primal_setting is None):
            self.log.warning("Setting does not contain any explicit data.")
            self._data=None
        if(self.data_fid.is_data_func):
            self._data=self.data_fid.data#just update internal data, update of data functional not necessary
        if(data is not None):
            self.data = data #data and data fidelity functional are updated
        
        self.log.setLevel(logging_level)
        self.gap_threshold = gap_threshold
        if primal_setting is not None and not (primal_setting.is_convex and primal_setting.is_tikhonov):
            raise ValueError(Errors.value_error("The primal_setting needs to be convex and contain a regularization parameter!"))
        self.primal_setting = primal_setting
        if primal_setting is None and self.is_convex and self.is_tikhonov:
            self._methods = Setting._generate_full_solver_dictionary()

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self,new_data):
        self.change_data(new_data=new_data)

    def change_data(self,new_data):
        if(new_data is None):
            raise ValueError(Errors.value_error(f"Overwriting data with {None} is not allowed."))
        if(self.data_fid.is_data_func):
            self.log.warning("Existing data in data fidelity functional is overwritten.")
        self.data_fid=self.data_fid.as_data_func(new_data)
        self._data=new_data
        self._set_flags()


    def _set_flags(self):
        self.is_tikhonov=(self.regpar is not None)
        """True if a regularization parameter is set"""
        self.is_convex=self.op.linear and self.penalty.convex and self.data_fid.convex
        """True if the operator is linear"""
        self.is_hilbert=(isinstance(self.penalty,SquaredNorm) and isinstance(self.data_fid,SquaredNorm))
        """True if penalty and data fidelity are both squared norms"""

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
           \mathcal{R}^\ast(T^\ast p) + \frac{1}{\alpha}\mathcal{S}^\ast(- \alpha p) = \min!

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

    def dual_to_primal(self,dual, own= False):
        r""" Returns an element of :math:`\partial \mathcal{R}^*(T^*p)` 
        If :math:`p` is a solution to the dual problem and :math:`\partial\mathcal{R}^*` is a singleton, this yields a solution to the primal problem. 
                
        Parameters
        ----------
        dual: tuple of self.op.adjoint.domain and self.op.adjoint.codomain
            tuple of dual variable p and T*p. Either p or T*p must not be None. If T*p is None, it will be 
            computed. Otherwise, p will not be used.
        own: bool [default: False]
            Only relevant for dual settings. If False, the duality relations of the primal setting are used. 
            If true, the duality relations of the dual setting are used. 
        """
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for the computation of a dual primal mapping."))
        if(not self.is_convex):
            raise RuntimeError(Errors.generic_message("The setting has to be convex for the computation of a dual primal mapping."))
        if not isinstance(dual,tuple) or len(dual)!=2:
            raise TypeError(Errors.type_error("dual must be a tuple of (p,T*p)"))
        if dual[0] is None and dual[1] is None:
            raise ValueError(Errors.value_error("Either p or T*p must be given in dual tuple!"))
        if dual[1] is None:
            dual[1] = self.op.adjoint(dual[0])
        if not dual[1] in self.op.adjoint.codomain:
            raise TypeError(Errors.type_error("T*p not in codomain of adjoint operator!"))
        if self.primal_setting is None or own == True:
            return self.penalty.conj.subgradient(dual[1])
        else:
            return self.primal_setting.primal_to_dual((None,-self.regpar*dual[1]))
            """Note that the dual variables of the dual problem differ by a factor -alpha_d from the primal variables of the primal problem.
            Here alpha_d=1/alpha_p is the regularization parameter of the dual problem, and alpha_p the regularization parameter of the primal problem.
            """
        
    def primal_to_dual(self,primal, own=False):
        r"""
        Returns an element of :math:`(-1/\alpha) \partial \mathcal{S}(Tf)` 
        If :math:`f` is a solution to the primal problem and :math:`\partial \mathcal{S}` is a singleton, this 
        yields a solution to the dual problem. 
    
        Parameters
        ----------------------------
        primal: tuple of self.op.domain and self.op.codomain
            tuple of primal variable f and Tf. Either f or Tf must not be None. If Tf is None, it will be 
            computed. Otherwise, x will not be used.
        own: bool [default: False]
            Only relevant for dual settings. If False, the duality relations of the primal setting are used. 
            If true, the duality relations of the dual setting are used. 
        """
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for the computation of a primal dual mapping."))
        if(not self.is_convex):
            raise RuntimeError(Errors.generic_message("The setting has to be convex for the computation of a primal dual mapping."))
        if not isinstance(primal,tuple) or len(primal)!=2:
            raise TypeError(Errors.type_error("primal must be a tuple of (f,Tf)"))
        if primal[0] is None and primal[1] is None:
            raise ValueError(Errors.value_error("Either f or Tf must be given in primal tuple!"))
        if primal[1] is None:
            primal[1]
        if not primal[1] in self.op.codomain:
            raise TypeError(Errors.type_error("Tf not in codomain of operator!"))
        if self.primal_setting is None or own==True:
            return (-1./self.regpar) * self.data_fid.subgradient(primal[1])
        else:
            return self.primal_setting.dual_to_primal(primal)

    def _complete_primal_dual_tuples(self,primal=None,dual=None):
        r"""Completes either the primal or dual tuple by computing the missing operator application.
        If one of the tuples is None, it is computed using the primal_to_dual or dual_to_primal methods.

        Parameters
        ----------
        primal: tuple of setting.op.domain and setting.op.codomain [default: None]
            tuple of primal variable f and Tf. If Tf is None, it will be computed.
        dual: tuple of setting.op.adjoint.domain and setting.op.adjoint.codomain [default: None]
            tuple of dual variable p and T*p. If T*p is None, it will be computed. 

        Returns
        -------
        tuple of tuples
            Completed primal and dual tuples.
        """
        if primal is None and dual is None:
            raise ValueError(Errors.value_error("Either a primal or dual tuple need to be given to complete both!"))
        if primal is None:
            if dual[1] is None:
                p = dual[0]
                Tsp = self.op.adjoint(p)
                dual = (p,Tsp)
            f = self.dual_to_primal(dual)
            Tf = self.op(f)
            primal = (f,Tf)
        else:
            f= primal[0]
            if not f in self.op.domain:
                raise TypeError(Errors.type_error("f not in domain of operator!"))
            Tf = self.op(f) if primal[1] is None else primal[1]
            primal = (f,Tf)
        if dual is None:
            p = self.primal_to_dual(primal)
            Tsp = self.op.adjoint(p)
            dual = (p,Tsp)
        else:
            p = dual[0]
            if not p in self.op.adjoint.domain:
                raise TypeError(Errors.type_error("p not in domain of adjoint operator!"))
            Tsp = self.op.adjoint(p) if dual[1]is None else dual[1]
            dual = (p,Tsp)
        return primal,dual

    def duality_gap(self, primal=None, dual=None):
        r"""Computes the value of the duality gap 
        
        .. math::
            \frac{1}{\alpha}\mathcal{S}_{g^{\delta}}(Tf) + \mathcal{R}(f) - \frac{1}{\alpha} }\mathcal{S}_{g^{\delta}}^\ast(-\alpha p) - \mathcal{R}^\ast(T^\ast p)

        Parameters
        ----------
        primal: tuple of setting.op.domain and setting.op.codomain [default: None]
            tuple of primal variable :math:`f` and :math:`Tf`. If :math:`Tf` is None, it will be computed.
        dual: tuple of setting.op.adjoint.domain and setting.op.adjoint.codomain [default: None]
            tuple of dual variable :math:`p` and :math:`T*p`. If :math:`T*p` is None, it will be computed.        
        """
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for the computation of the duality gap."))
        if not self.is_convex:
            raise RuntimeError(Errors.not_linear_op(self.op,add_info="The duality gap can only be computed for convex settings with linear operators!"))
        (f,Tf),(p,Tsp) = self._complete_primal_dual_tuples(primal,dual)
        alpha = self.regpar

        dat = 1./alpha * self.data_fid(Tf)
        pen = self.penalty(f)
        ddat = self.penalty.conj(Tsp)
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
    
    def violation_optimality_cond(self,primal=None,dual=None):
        r"""Returns the degree to which a pair :math:`(f,p)` of a primal point :math:`f` and a dual point :math:`p\ 
        violates the optimality conditions for being a saddle point of 
        :math:`-<Tf,p> + \mathcal{R}(f)-\frac{1}{\alpha}\mathcal{S}^*(-\alpha p) `
        These optimality conditions are:

        .. math::
            Tf \in \partial \mathcal{S}^\ast(-\alpha p), \qquad T^\ast p \in \partial \mathcal{R}(f).

        This violation is measured by the distances of the left-hand sides to the respective 
        subdifferentials on the right-hand sides, and the function returns a tuple of these two distances.

        Parameters
        ---------------------------
        primal: tuple of setting.op.domain and setting.op.codomain [default: None]
            tuple of primal variable :math:`f` and :math:`Tf`. If :math:`Tf` is None, it will be computed.
        dual: tuple of setting.op.adjoint.domain and setting.op.adjoint.codomain [default: None]
            tuple of dual variable :math:`p` and :math:`T*p`. If :math:`T*p` is None, it will be computed. 
        If one of the tuples is None, it is computed using the primal_to_dual or dual_to_primal methods.

        Returns
        -------
        tuple of floats
            Distances to the subdifferentials in the two optimality conditions.
        """
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for this check."))
        if not "dist_subdiff" in self.penalty.methods or not "dist_subdiff" in self.data_fid.conj.methods:
            raise RuntimeError(Errors.generic_message("Need dist_subdiff method of both penalty and conjugate data fidelity functional."))
        if not self.is_convex:
            raise RuntimeError(Errors.not_linear_op(self.op,add_info="This check requires a convex setting with a linear operator!"))

        (f,Tf),(p,Tsp) = self._complete_primal_dual_tuples(primal,dual)

        alpha = self.regpar
        return (1./alpha)*self.data_fid.conj.dist_subdiff(Tf,(-alpha)*p), \
               self.penalty.dist_subdiff(Tsp,f) 



    
    ######Methods checking applicability
    @staticmethod
    def _generate_full_solver_dictionary():
        '''This so far contains only linear solvers'''
        from regpy.solvers.linear import ForwardBackwardSplitting,FISTA,PDHG,ADMM,SemismoothNewton_bilateral,TikhonovCG
        method_dict={
                'FB': {'class':ForwardBackwardSplitting, 'primal': True, 'full':'Forward Backward Splitting applied to primal problem'},
                'dual_FB': {'class':ForwardBackwardSplitting, 'primal': False, 'full': 'Forward Backward Splitting applied to primal problem'},
                'FISTA': {'class':FISTA, 'primal': True, 'full': 'Fast Iterative Thresholding applied to primal problem'}, 
                'dual_FISTA': {'class':FISTA, 'primal': False, 'full': 'Fast Iterative Thresholding applied to dual problem'},
                'PDHG': {'class':PDHG, 'primal': True, 'full': 'Primal-Dual Hybrid Gradient Method applied to primal problem'},
                'dual_PDHG': {'class':PDHG, 'primal': False, 'full': 'Primal-Dual Hybrid Gradient Method applied to dual problem'},
                'ADMM': {'class':ADMM, 'primal': True, 'full': 'Alternating Direction Method of Multipliers' },
                'SSNewton': {'class':SemismoothNewton_bilateral, 'primal': True, 'full': 'Semismooth Newton method'},
                'dual_SSNewton': {'class':SemismoothNewton_bilateral, 'primal': False, 'full': 'Semismooth Newton method applied to dual problem'}
            }
        return method_dict
    
    def evaluate_methods(self,method_names = None):
        """Evaluates which methods are applicable to the current setting. 
        This is achieved by calling method.check_applicability(self), which also provide information on guaranteed rates.

        Parameters
        ----------
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
        
        Parameters
        ----------
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
        
        Parameters
        ----------
        method_name: string
            Key of the method
        
        Returns
        -------
        StopRule
        """
        if not method_name in self._methods.keys():
            raise ValueError(f"{method_name} is unknown method key.")
        if 'stoprule' not in self._methods[method_name]:
            raise RuntimeError(f'Method {method_name} has no StopRule.')
        else:
            return self._methods[method_name]['stoprule']   

    def run(self,method_name = None,**kwargs):
        """Runs a given method for the setting. If no method name is given, the "best" method is selected by select_best_method() if possible.
        
        Parameters
        ----------
        method_name: string or None [default: None] 
            Key of the method to be run in the methods dictionary self._methods (can be displayed by display_all_methods())
            If None the "best" method is selected by select_best_method().
        **kwargs: dict
            Arguments to be passed to the method.

        Returns
        -------
            x,y: x is the minimizer of the appproximate solution and y its value under the operator.         
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
            self.set_stopping_rule(method_name, DualityGapStopping(tol = 0.1,logging_level=logging.INFO)
                                   +CountIterations(1000,logging_level=logging.INFO))

        
        solver = themethod['class'](thesetting,**kwargs)
        x,y = solver.run(themethod['stoprule'])
        
        if themethod['primal']==False:
            x_star,y_star = x,y
            x = self.dual_to_primal((x_star,y_star))
            y = self.op(x)
        return x,y