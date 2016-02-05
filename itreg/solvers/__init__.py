"""Solver classes."""

import logging


class Solver(object):
    """Abstract base class for solvers.

    Parameters
    ----------
    log : :class:`logging.Logger`, optional
        The logger to be used. Defaults to the root logger.

    Attributes
    ----------
    x : array
        The current iterate.
    y : array, optional
        The value at the current iterate. May be needed by stopping rules, but
        callers should handle the case when it is not available.
    log : :class:`logging.Logger`
        The logger in use.

    """

    def __init__(self, log=logging.getLogger()):
        self.log = log

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

    def run(self, stoprule=None):
        """Run the solver with the given stopping rule.

        This is convenience method that implements a simple loop running the
        solver using its :meth:`next` method until it either converges or the
        stopping rule triggers

        Parameters
        ----------
        stoprule : :class:`StopRule <itreg.stoprules.StopRule>`, optional
            The stopping rule to be used. If omitted, stopping will only be
            based on the return value of :meth:`next`.

        """
        while True:
            if stoprule is not None:
                if hasattr(self, 'y'):
                    stop = stoprule.stop(self.x, self.y)
                else:
                    stop = stoprule.stop(self.x)
                if stop:
                    self.log.info('Stopping rule triggered.')
                    return stoprule.select()
            if not self.next():
                self.log.info('Solver converged.')
                return self.x


from .landweber import Landweber  # NOQA

__all__ = [
    'Landweber',
    'Solver'
]
