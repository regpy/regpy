"""Solver classes."""

import logging


class Inner_Solver(object):
    """Abstract base class for inner solvers (for example SQP method).


    """

    def __init__(self,solver = landweber,log=logging.getLogger()):
        self.solver = solver
        self.op = solver.op
        self.
        
        
        
        
        self.log = log
        self.x = None
        self.y = None
        self.solver_param= solver.param
    def next(self):
        """Perform a single iteration.

        This is an abstract method. Child classes should override it.

        Returns
        -------
        bool
            `True` if caller should continue iterations, `False` if the method
            converged. Most solvers will always return `True` and delegate the
            stopping decision to a :class:`StopRule <itreg.stoprules.StopRule>`.

        """
        raise NotImplementedError()

    def __iter__(self):
        """Return and iterator on the iterates of the solver.

        Yields
        ------
        tuple of array
            The (x, y) pair of the current iteration. Callers should not expect
            arrays from previous iterations to be valid, as the solver might
            modify them in-place.

        """
        while self.next():
            yield (self.x, self.y)

    def run(self, stoprule=None):
        """Run the solver with the given stopping rule.

        This is convenience method that implements a simple loop running the
        solver until it either converges or the stopping rule triggers.

        Parameters
        ----------
        stoprule : :class:`StopRule <itreg.stoprules.StopRule>`, optional
            The stopping rule to be used. If omitted, stopping will only be
            based on the return value of :meth:`next`.

        """
        for x, y in self:
            if stoprule is not None and stoprule.stop(x, y):
                self.log.info('Stopping rule triggered.')
                return stoprule.x
        self.log.info('Solver converged.')
        return x


#from .landweber import Landweber  # NOQA
#from .newton_cg import Newton_CG
#from .irgnm_cg import IRGNM_CG
#
#__all__ = [
#    'Landweber',
#    'Newton_CG',
#    'IRGNM_CG',
#    'Solver'
#]
