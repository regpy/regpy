import math as ma
import numpy as np
from typing import Callable

from regpy.util import ClassLogger, Errors
from regpy.util.operator_tests import test_derivative
from regpy.operators import Operator
from regpy.hilbert import HilbertSpace
from regpy.functionals.base import  as_functional, Composed
from regpy.functionals import Functional,SquaredNorm, QuadraticLowerBound, QuadraticNonneg, QuadraticBilateralConstraints
from regpy.stoprules import StopRule,NoneRule,DualityGapStopping,CombineRules,CountIterations
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
        r"""Run the solver with the given stopping rule. 
        """
        for x, y in self.while_(stoprule):
            if not 'x' in locals(): 
                # This happens if the stopping criterion is satisfied for the initial guess.
                x = self.x
                y = self.y
        if stoprule.best_iterate() is None:
            self.log.info(f"Could not find a best iterate with stop rule {stoprule}!")
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
    r"""A *setting* for an inverse problem, used by solvers. A setting always consists at least of

    - a forward operator :math:`F`,
    - a penalty functional :math:`\mathcal{R}` with an associated Hilbert space structure to measure the error, and
    - a data fidelity functional :math:`\mathcal{S}_{g^{\delta}}` with an associated Hilbert space structure to measure the data misfit.

    If a regularization parameter :math:`alpha` is given, this is the setting for the minimization problem 

    .. math::
        \frac{1}{\alpha}\mathcal{S}_{g^{\delta}}(F(f)) + \mathcal{R}(f) = \min!

    If the operator is linear and both functionals are convex, this is the setting of Rockafellar-Fenchel duality 
    --- a rich and algorithmically useful mathematical structure. In this case, the dual setting 
    and primal-dual optimality conditions are provided. 
   
    This class is mostly a container that keeps all of this data in one place and makes sure that all initializations are 
    done correctly.

    The meaning of a "solver" is quite different in the case where a regularization parameter is given and in the case where it is not given. 
    In the first case, a solver is a minimization algorithm. In the seond case, a solver can be a wrapper applying solvers of the first kind
    for different regularization parameter, driven by a parameter selection rule as `StopRule`.
    Other solvers of the second kind are iterative mimimization method based 
    on the penalty and data fidelity term such as a Newton-type (which also uase solvers of the first kind as inner iterations) or radient methods. Since iterative methods exhibit semiconvergent behavior in the presence of ill-posedness, early stopping by an appropriate `StopRule` is 
    again essential.  

    Parameters
    ----------
    op : regpy.operators.Operator
        The forward operator :math:`F`.
    penalty : regpy.functionals.Functional or regpy.hilbert.HilbertSpace or callable
        The penalty functional  :math:`\mathcal{R}`.
        If a Hilbert space is given, the squared Hilbert norm (class `regyp.functionals.SquaredNorm`) is used as penalty functional. 
        If an `AbstractFunctional`, or more generally any callable is given instead of a `regpy.functionals.Functional`, it is called on the operator's domain to construct a concrete `Functional` instance.
    data_fid : regpy.functionals.Functional or regpy.hilbert.HilbertSpace or callable
        The data misfit functional :math:`\mathcal{S}_{g^{\delta}}`.
        The cases of Hilbert space and callable instances are treated in analogy to penalty. 
    regpar: float or None, optional
        regularization parameter  :math:`alpha`.
    penalty_shift: op.domain or None, optional
        If not None, the penalty functional  :math:`\mathcal{R}` is replaced by  :math:`\mathcal{R}(. - penalty_shift)`. Defaults to None
    data: array-like or None, optionals
        If not None, the data :math:`g^{\delta}` in the data fidelity functional is replaced by data (which often but not necessarily 
        belong to op.codomain). Defaults to None
    exact_data: op.codomain | None, optional 
        A setting may also be initialized with exact data. Once the setting is instantiated, convenience methods 
        such add add_Gaussian_noise() or generate_Poisson_data() can be used to generate synthetic noisy data. 
        Before this happens, exact_data will not have any effect.
    logging_level: int [default: logging.INFO]
        logging level
    """

    
    
    log = ClassLogger()

    def __init__(self, 
                 op: Operator, 
                 penalty: Functional|HilbertSpace|Callable, 
                 data_fid:Functional|HilbertSpace|Callable,
                 regpar:float|None=None,
                 penalty_shift= None, 
                 data= None, 
                 exact_data = None, 
                 logging_level = "INFO",
                 _primal_setting =None # intended only for internal use in get_dual_setting()
                 ):
        if not isinstance(op,Operator):
            raise TypeError(Errors.not_instance(op,Operator,add_info="Setting requires op to be a RegPy operator."))
        self._op = op
        self._data_fid = as_functional(data_fid, op.codomain)
        if not penalty_shift is None:
            self._penalty = as_functional(penalty, op.domain).shift(penalty_shift)
        else:
            self._penalty = as_functional(penalty, op.domain)
        self.regpar=regpar#The flags are set by setting the regularization parameter
        if(not self.data_fid.is_data_func and data is None and _primal_setting is None):
            if exact_data is None:
                self.log.warning("Setting does not contain any explicit data.")
            self._data=None
        if(self.data_fid.is_data_func):
            self._data=self.data_fid.data#just update internal data, update of data functional not necessary
        if(data is not None):
            self.data = data #data and data fidelity functional are updated
        if exact_data is not None:
            if not exact_data in op.codomain:
                raise ValueError(Errors.value_error("exact_data must be in codomain of operator."))
            self._exact_data = exact_data

        self.log.setLevel(logging_level)
        if _primal_setting is not None and not (_primal_setting.is_convex and _primal_setting.is_tikhonov):
            raise ValueError(Errors.value_error("The primal_setting needs to be convex and contain a regularization parameter!"))
        self._primal_setting = _primal_setting
        if _primal_setting is None and self.is_convex and self.is_tikhonov:
            self._methods = Setting._generate_full_solver_dictionary()

    @property
    def op(self):
        r"""The operator."""
        return self._op
    
    @property
    def penalty(self):
        r"""The penalty functional."""        
        return self._penalty

    @property
    def data_fid(self):
        r"""The data fidelity functional."""
        return self._data_fid 
    @property
    def h_domain(self):
        r"""The Hilbert space associated to penalty functional"""
        return self.penalty.h_domain

    @property   
    def h_codomain(self):
        r"""The Hilbert space associated to data fidelity functional"""
        return self.data_fid.h_domain if not isinstance(self.data_fid,Composed) else self.data_fid.func.h_domain


    @property
    def data(self):
        return self._data if hasattr(self,'_data') else None
    
    @data.setter
    def data(self,new_data):
        self.change_data(new_data=new_data)

    def change_data(self,new_data):
        if(new_data is None):
            raise ValueError(Errors.value_error(f"Overwriting data with {None} is not allowed."))
        if not hasattr(self,"_data") or not new_data is self._data:
            if(self.data_fid.is_data_func):
                self.log.warning("Existing data in data fidelity functional is overwritten.")
            self._data_fid=self.data_fid.as_data_func(new_data)
            self._data=new_data
            self._set_flags()

    def _set_flags(self):
        self.is_tikhonov=(self.regpar is not None)
        """True if a regularization parameter is set"""
        self.is_convex=self.op.linear and self.penalty.is_convex and self.data_fid.is_convex
        """True if the operator is linear"""
        self.is_hilbert=(isinstance(self.penalty,SquaredNorm) and isinstance(self.data_fid,SquaredNorm))
        """True if penalty and data fidelity are both squared norms"""

    @property
    def regpar(self):
        r"""The regularization parameter"""
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
    def add_Gaussian_noise(self,relative_noise_level:float=None, absolute_noise_level:float=None,white_noise=True,seed=None):
        r""" generates Gaussian noise using self.codomain.randn, adds it to exact_data (which must have been provided at initialization of the setting), and sets data to the sum.
        
        Parameters:
        -----------
        relative_noise_level: float | None: optional
        absolute_noise_level: float | None: optional
            Exactly one of these two parameters must be given as a positive float. 
        white_noise: bool: optional
            Defaults to True.
            If true, the noise is generated as white noise with respect to the codomain Hilbert space structure using a Cholesky-type factorization of the gram matrix (which needs to be implemented for the codomain space!); otherwise it is generated by self.codomain.randn.
            If true, relative_noise_level and absolute_noise_level refer to the Hilbert space norm of the covariance operator, and the noise process is asymptotically discretization invariant; otherwise they refer to the norm in the codomain vector space.
        seed: int | None: optional
            seed for random number generator for reproducibility
        """
        if relative_noise_level is None and absolute_noise_level is None:
            raise ValueError(Errors.value_error("Either relative or absolute noise level must be given."))
        if relative_noise_level is not None and absolute_noise_level is not None:
            raise ValueError(Errors.value_error("Your cannot provide both relative or absolute noise level!"))
        if relative_noise_level is not None and not isinstance(relative_noise_level,float) and not relative_noise_level>0:
            raise ValueError(Errors.value_error("ralative_noise_level must be a positive float.")) 
        if absolute_noise_level is not None and not isinstance(absolute_noise_level,float) and not absolute_noise_level>0:
            raise ValueError(Errors.value_error("absolute_noise_level must be a positive float.")) 
        if not isinstance(white_noise,bool):
            raise TypeError(Errors.type_error("white_noise must be a boolean."))
        if not hasattr(self,'_exact_data'):
            raise RuntimeError(Errors.runtime_error('No exact data has been provided at initialization of the setting.'))
        if self.data is not None:
            self.log.warning("Overwriting given data!")
        if white_noise:
            try:
                chol = self.h_codomain.cholesky
            except NotImplementedError:
                raise NotImplementedError("Cholesky factorization is not implemented for the codomain Hilbert space.")
            noise = chol.domain.randn(seed=seed)
            if relative_noise_level is not None:
                noise *= relative_noise_level/self.h_codomain.norm(self._exact_data)
            else:
                noise *= absolute_noise_level
            noise = chol.inverse.adjoint(noise) 
        else:
            noise = self.op.codomain.randn(seed=seed)
            if relative_noise_level is not None:
                noise *= relative_noise_level*self.h_codomain.norm(self._exact_data)/self.h_codomain.norm(noise)
            else:
                noise *= absolute_noise_level/self.h_codomain.norm(noise)
        self.data = self._exact_data + noise 

    def generate_Poisson_data(self,expected_nr_counts=None,seed=None):
        r""" generates Poisson data using self.codomain.poisson (which must have been provided at initialization of the setting), and sets data to this.
        
        Parameters:
        -----------
        expected_nr_counts: int | None: optional
            Expected total number of counts for the Poisson distribution.
            The Poisson data are rescaled such that the expectation of the data is the exact data. This means that the resulting data are not integer-valued. 
            If None, no scaling is applied, and data are integer-valued.
        seed: int | None: optional
            random seed for reproducibility
        """
        if not hasattr(self,'_exact_data'):
            raise RuntimeError(Errors.runtime_error('No exact data has been provided at initialization of the setting.'))
        if not np.issubdtype(self.op.codomain.dtype, np.floating) and np.all(self._exact_data >=0):
            raise RuntimeError(Errors.runtime_error('Poisson data can only be generated for non-negative real-valued data spaces.'))
        if self.data is not None:
            self.log.warning("Overwriting given data!")
        if expected_nr_counts is not None:
            scale = expected_nr_counts / self.h_codomain.sum(self._exact_data)
        else:
            scale = 1.0      
        poisson_data = self.op.codomain.poisson(scale*self._exact_data,seed=seed)/scale
        self.data = poisson_data    

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

    def get_or_update_data(self, new_data=None,update:bool=True):
        r""" Updates the data stored in the setting or gets these data of they are not provided.

        Parameters
        ----------
        update: bool
            If True, the initial guess stored in the setting is updated to new_init.
        new_init : op.domain
            New initial guess in the domain of the operator.
        """
        if new_data is not None:
            if update:
                if self.data is not new_data:
                    self.log.warning("Overwriting existing data in setting!")
                    self.data = new_data
            return new_data
        else:
            if self.data is None:
                self.log.warning("No data has been provided either explicitly or as an argument of the method or in the setting!")
            return self.data

    def get_or_update_initial_guess(self, new_init=None,update:bool=True):
        r""" Updates the initial guess stored in the setting or gets the initial guess if it is not provided.

        Parameters
        ----------
        update: bool
            If True, the initial guess stored in the setting is updated to new_init.
        new_init : op.domain
            New initial guess in the domain of the operator.
        """
        if new_init is not None:
            if not new_init in self.op.domain:
                raise TypeError(Errors.not_in_vecsp(new_init,self.op.domain,vec_name="initial guess",space_name="domain"))
            if update:
                if hasattr(self,'init') and not self.init is new_init:
                    self.log.warning("Overwriting existing initial guess in setting!")
                self.init = new_init.copy()
            return new_init
        else:
            if hasattr(self,'init'):
                toret = self.init.copy()
            else:
                toret = self.op.domain.zeros()
                if update:
                    self.init = toret.copy()
            return toret

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
            _primal_setting = self,
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
        if self._primal_setting is None or own == True:
            return self.penalty.conj.subgradient(dual[1])
        else:
            return self._primal_setting.primal_to_dual((None,-self.regpar*dual[1]))
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
        if self._primal_setting is None or own==True:
            return (-1./self.regpar) * self.data_fid.subgradient(primal[1])
        else:
            return self._primal_setting.dual_to_primal(primal)

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
        elif ares/res>1e5:
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
                'TikhCG': {'class':TikhonovCG, 'primal': True, 'full':'conjugate gradient method applied to normal equation'},
                'dual_TikhCG': {'class':TikhonovCG, 'primal': False, 'full':'conjugate gradient method applied to dual normal equation'},
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
        if not (self._primal_setting is None and self.is_convex and self.is_tikhonov):
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
        if not (self._primal_setting is None and self.is_convex and self.is_tikhonov):
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

    def set_stopping_rule(self,
                          rule:StopRule,
                          method_names:list[str]|None=None):
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
        if method_names is None:
            method_names = list(self.applicable_methods())
        if not isinstance(method_names,list) and all([isinstance(method_name,str) for method_name in method_names]):
            raise ValueError(Errors.value_error('method_names must be list of strings',method_names))
        for method_name in method_names:
            if method_name not in self._methods.keys():
                raise ValueError(f"{method_name} is unknown method key.")
            self._methods[method_name]['stoprule'] = rule.copy_and_reset()

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
            self.set_stopping_rule(DualityGapStopping(tol = 0.1,logging_level=logging.INFO)
                                   +CountIterations(1000,logging_level=logging.INFO),
                                   method_names=[method_name])

        
        solver = themethod['class'](thesetting,**kwargs)
        x,y = solver.run(themethod['stoprule'])
        
        if themethod['primal']==False:
            x_star,y_star = x,y
            x = self.dual_to_primal((x_star,y_star))
            y = self.op(x)
        return x,y