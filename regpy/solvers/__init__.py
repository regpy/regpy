"""Solvers for inverse problems.
"""

from regpy.util import classlogger
from regpy.stoprules import NoneRule
from regpy.functionals import  as_functional, Composed


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

    def converge(self):
        """Mark the solver as converged. This is intended to be used by child classes
        implementing the `_next` method.
        """
        self.__converged = True

    def next(self):
        """Perform a single iteration.

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
        """Perform a single iteration. This is an abstract method called from the public method
        `next`. Child classes should override it.

        The main difference to `next` is that `_next` does not have a return value. If the solver
        converged, `converge` should be called.
        """
        raise NotImplementedError

    def __iter__(self):
        """Return an iterator on the iterates of the solver.

        Yields
        ------
        tuple of arrays
            The (x, y) pair of the current iteration.
        """
        while self.next():
            yield self.x, self.y

    def while_(self, stoprule=NoneRule()):
        """Generator that runs the solver with the given stopping rule. This is a convenience method
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
        """Generator that runs the solver with the given stopping rule. This is a convenience method
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
        """Run the solver with the given stopping rule. This method simply runs the generator
        `regpy.solvers.Solver.while_` and returns the final `(x, y)` pair.
        """
        for x, y in self.while_(stoprule):
            pass
        if not 'x' in locals() or not 'y' in locals(): 
            # This happens if the stopping criterion is satisfied for the initial guess.
            x = self.x
            y = self.y
        return x, y


class RegularizationSetting:
    """A Regularization *setting* for an inverse problem, used by solvers. A
    setting consists of

    - a forward operator,
    - a penalty functional with an associated Hilbert space structure to measure the error, and
    - a data fidelity functional with an associated Hilbert space structure to measure the data misfit.

    This class is mostly a container that keeps all of this data in one place and makes sure that
    the the used penalty and data fidelity have matching domains `regpy.hilbert.HilbertSpace.vecsp`s 
    with the operator's domain and codomain.

    It also handles the case when the specified data fidelity or penalty is a Hilbert space which constructs 
    the associated squared Hilbert norm functionals. It also handles cases when `regpy.hilbert.AbstractSpace` 
    or `AbstractFunctional`s (or actually any callable) instead of a `regpy.functionals.Functional`, calling 
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
        self.op = op
        """The operator."""
        self.penalty = as_functional(penalty, op.domain)
        """The penalty functional."""
        self.data_fid = as_functional(data_fid, op.codomain)
        """The data misfit functional."""
        self.h_domain = self.penalty.h_domain
        """The Hilbert space associated to penalty functional"""
        self.h_codomain =  self.data_fid.h_domain if not isinstance(self.data_fid,Composed) else self.data_fid.func.h_domain
        """The Hilbert space associated to data fidelity functional"""