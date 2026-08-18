from regpy.vecsps import Vector
from regpy.stoprules import StopRule, NoneRule
from regpy.util import ClassLogger

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
    x : Vector | None [default: None]
        Initial argument for iteration. Defaults to None.
    y : Vector | None [default: None]
        Initial value at current iterate. Defaults to None.
    """

    log = ClassLogger()

    def __init__(self, 
                 x:Vector | None = None,
                 y:Vector | None = None):
        self.x = x
        """The current iterate."""
        self.y = y
        """The value at the current iterate. May be needed by stopping rules, but callers should
        handle the case when it is not available."""
        self.__converged = False
        self.iteration_step_nr = 0
        """Current number of iterations performed."""

    def converge(self) -> None:
        """Mark the solver as converged. This is intended to be used by child classes
        implementing the `_next` method.
        """
        self.__converged = True

    def is_converged(self) -> bool:
        return self.__converged

    def next(self) -> bool:
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

    def while_(self, stoprule:StopRule = NoneRule()):
        r"""Generator that runs the solver with the given stoprule. This is a convenience method
        that implements a simple generator loop running the solver until it either converges or the
        stoprule triggers.

        Parameters
        ----------
        stoprule : regpy.stoprules.StopRule, optional
            The stoprule to be used. If omitted, stopping will only be
            based on the return value of `next`.

        Yields
        ------
        tuple of arrays
            The (x, y) pair of the current iteration, or the solution chosen by
            the stoprule.
        """
        stoprule._complete_init_with_solver(self)
        while not stoprule.stop() and self.next(): 
            yield self.x, self.y
        self.log.info('Solver converged after {} iteration.'.format(self.iteration_step_nr))
 


    def until(self, stoprule:StopRule = NoneRule()):
        r"""Generator that runs the solver with the given stoprule. This is a convenience method
        that implements a simple generator loop running the solver until it either converges or the
        stoprule triggers.

        Parameters
        ----------
        stoprule : regpy.stoprules.StopRule, optional
            The stoprule to be used. If omitted, stopping will only be
            based on the return value of `next`.

        Yields
        ------
        tuple of Vectors
            The (x, y) pair of the current iteration, or the solution chosen by
            the stoprule.
        """
        stoprule._complete_init_with_solver(self)
        self.next()
        yield self.x, self.y
        while not stoprule.stop() and self.next(): 
            yield self.x, self.y

        self.log.info('Solver converged after {} iteration.'.format(self.iteration_step_nr))

    def run(self, stoprule:StopRule = NoneRule()):
        r"""Run the solver with the given stoprule. 
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
    