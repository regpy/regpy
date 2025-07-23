"""Solvers for inverse problems.
"""
import numpy as np
from scipy.sparse.linalg import eigsh

import logging
from regpy.util import classlogger
from regpy.stoprules import NoneRule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

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

    log = classlogger

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

        while not stoprule.stop(self.x,self.y) and self.next(): 
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
        while not stoprule.stop(self.x,self.y) and self.next(): 
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
        assert isinstance(setting,RegularizationSetting)
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
        if isinstance(setting,TikhonovRegularizationSetting):
            self.setting = setting
            """The regularization setting"""
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
    def __init__(self, op, penalty, data_fid):
        from regpy.functionals import  as_functional, Composed
        from regpy.operators import Operator
        assert isinstance(op,Operator)
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
    
            np.real(np.vdot(y, self.op(x)) - np.vdot(self.op.adjoint(y), x)) < tolerance

        If the operator is non-linear this will be done for the derivative.

        Parameters
        ----------
        tolerance : float
            Tolerance of the two computed inner products.

        Assertion
        ---------
        Assertion is thrown by the `regpy.util.operator_tests.test_adjoint` when it does not fit. 
        """
        from regpy.util.operator_tests import test_adjoint
        if self.op.linear:
            test_adjoint(self.op,tolerance=tolerance)
        else:
            _, deriv = self.op.linearize(self.op.domain.randn())
            test_adjoint(deriv, tolerance=tolerance)

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
            True if the sequence provided by `regpy.util.operator_tests.test_derivative` is decreasing.
        """
        from regpy.util.operator_tests import test_derivative
        if steps is None:
            steps = [10**k for k in range(-1, -8, -1)]
        if self.op.linear:
            return True
        seq = test_derivative(self.op,steps=steps,ret_sequence=True)
        return all(seq_i > seq_j for seq_i, seq_j in zip(seq, seq[1:]))
    
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
            _ , deriv = self.op.linearize(x)
            return self.h_domain.gram_inv * deriv.adjoint * self.h_codomain.gram, deriv
        
    def op_norm(self,op = None, method = "lanczos"):
        r"""Approximate the operator norm of :math:`T` for a linear operator :math:`T` with respect to the vector norms of h_domain and h_codomain. 
        This is achieved by computing the largest eigenvalue of :math:`T^*T` using eigsh from scipy. 
        # To-do: Test making this a memoized property (should only be recomputed if non-linear, should be possible for user to input if analytically known).    
        #@memoized_property
 
        Parameters
        ----------
        op: linear `regpy.operators.Operator` from self.domain to self.codomain [default=None]
            Typically the derivative of the operator at some point. In the default case, self.op is used if self.op is linear. 
        method: string [default: "lanczos"]
            Method by which an approximation of the operator norm is computed. Alternative: "power_method"
        Returns
        -------
        scalar
            Approximation of the norm of T^*T. 

        Raises
        ------
        NotImplementedError
            If the setting is not a Hilbert space setting, meaning `penalty` and `data_fid`
            are not `HilbertNormGeneric` instances this is not implemented.
        """

        if op is None:
            if self.op.linear:
                T= self.op
            else:
                raise NotImplementedError
        else:
            T=op
            assert T.domain == self.op.domain
            assert T.codomain == self.op.codomain

        if method == "power_method":
            return power_method(self, op = T)
        elif method == "lanczos":
            from regpy.operators import SciPyLinearOperator
            return np.sqrt(eigsh(SciPyLinearOperator(T.adjoint * self.h_codomain.gram * T), 1, M=SciPyLinearOperator(self.h_domain.gram),tol=0.01)[0][0])
        else:
            raise NotImplementedError

    def is_hilbert_setting(self):
        r"""Assert if the setting is a Hilbert space setting. 

        Returns
        -------
        Boolean
            True if both `penalty` and `data_fid` are `HIlbertNormGeneric` functionals. 
        """
        from regpy.functionals import  HilbertNormGeneric
        return isinstance(self.penalty,HilbertNormGeneric) and isinstance(self.data_fid,HilbertNormGeneric)
        

class TikhonovRegularizationSetting(RegularizationSetting):
    r"""Tikhonov regularization setting for minimizing a Tikhonov functional 

    .. math::
        \frac{1}{\alpha}\mathcal{S}_{g^{\delta}}(Tf) + \mathcal{R}(f) = \min!

    In contrast to RegularizationSetting, the regularization parameter is fixed, 
    the data fidelity functional \(\mathcal{S}=self.data_fid)\ incorporates the data \(g^{\delta})\ of the inverse problem, 
    and the penalty term \(\mathcal{R})\ incorporates a potential initial guess.

    Parameters
    -------------------
    op : regpy.operators.Operator
        The forward operator.
    penalty : regpy.functionals.Functional
        The penalty functional :math:`\mathcal{R}`.
    data_fid : regpy.functionals.Functional
        The data misfit functional \(\mathcal{S}_{g^{\delta}})\.
    regpar: float [default: 1]
        regularization parameter
    penalty_shift: op.domain [default: None]
        If not None, the penalty functional is replaced by penalty(. - penalty_shift).
    data_fid_shift: op.co_domain [default: None]
        If not None, the data fidelity functional is replaced by data_fid(. - data_fid_shift).
    primal_setting: None or TikhonovRegularizationSetting [default:None]
        Indicates whether or not a setting serves as primal setting. For a primal setting, primal_setting is None, for a dual setting it is the primal setting. 
        This affects the duality relations and the duality gap. 
    logging_level: int [default: logging.INFO]
        logging level
    """

    log = classlogger

    def __init__(self, op, penalty, data_fid,regpar=1.,penalty_shift= None, data_fid_shift= None, 
                 primal_setting=None,logging_level = logging.INFO,gap_threshold = 1e5):
        super().__init__(op,penalty=penalty, data_fid= data_fid)

        if not penalty_shift is None:
            self.penalty = self.penalty.shift(penalty_shift)

        if not data_fid_shift is None:
            self.data_fid = self.data_fid.shift(data_fid_shift)

        assert isinstance(regpar,(float,int)) and regpar>=0
        self.regpar = float(regpar)
        self.log.setLevel(logging_level)
        self.gap_threshold = gap_threshold
        """The regularization parameter"""
        assert primal_setting is None or isinstance(primal_setting,TikhonovRegularizationSetting)
        self.primal_setting = primal_setting
    
    def dualSetting(self):
        r"""Yields the setting of the dual optimization problem

        .. math::
           \mathcal{R}^*(\T^*p) + \frac{1}{\alpha}\mathcal{S}^*(- \alpha p) = \min!

        """
        assert self.op.linear
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
                assert self.op.linear
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
        assert self.op.linear
        assert not (primal is None and dual is None)
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
        ares = np.abs(dat)+np.abs(pen)+np.abs(ddat)+np.abs(dpen) 
        if not np.isfinite(ares):
            self.log.warning('duality gap infinite: R(..)={:.3e}, S(..)={:.3e}, S*(..)={:.3e}, R*(..)={:.3e},'.format(pen,dat,dpen,ddat))
            return np.inf
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
        assert self.op.linear
        return self.data_fid.conj.is_subgradient(self.op(x),self.regpar*p,tol=tol) and \
               self.penalty.is_subgradient(-self.op.adjoint(p),x,tol=tol) 


def power_method(setting,op=None,max_iter=int(1e2),stopping_rule=1e-12):
    r"""Approximation of operator norm by the power method.

    Parameters
    ----------
    setting : RegularizationSetting
        Provides op and Gram. 
    op : Operator [default: None]
        Optionally overrides choice of operator (e.g. for linearization)
    """
    assert isinstance(setting,RegularizationSetting)
    if op is None:
        op = setting.op
    assert op.linear
    
    x = setting.op.domain.rand()
    relative_residual = np.inf
    for _ in range(max_iter):
        if relative_residual < stopping_rule:
            break
        ystar = (op.adjoint * setting.h_codomain.gram * op)(x)
        y = setting.h_domain.gram_inv(ystar)
        lmb = np.sqrt(np.vdot(y, ystar).real)
        relative_residual = setting.h_domain.norm(y - lmb * x)
        x = y/lmb
    return np.sqrt(lmb)
