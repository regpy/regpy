"""Implementations of various stopping rules."""

import logging


class StopRule(object):
    """Abstract base class for stopping rules.

    Parameters
    ----------
    log : :class:`logging.Logger`, optional
        The logger to be used. Defaults to the root logger.

    Attributes
    ----------
    log : :class:`logging.Logger`
        The logger in use.
    needs_y : bool
        Should be set to `True` by child classes if the stopping rule needs the
        operator value at the current point. This can be used by callers to pass
        in the value if it is available, avoiding recomputation. Defaults to
        `False`.

    """

    def __init__(self, log=logging.getLogger()):
        self.log = log
        self.needs_y = False

    def stop(self, x, y=None):
        """Check whether to stop iterations.

        This is an abstract method. Child classes should override it.

        Parameters
        ----------
        x : array
            The current iterate.
        y : array, optional
            The operator value at the current iterate. Can be omitted if
            unavailable.

        Returns
        -------
        bool
            `True` if iterations should be stopped. The solution can be obtained
            from the :meth:`select` method in that case.

        """
        raise NotImplementedError()

    def select(self):
        """Select the final solution after iterations have been stopped.

        Stopping rules may decide to yield a result different from the last
        iterate. Therefore, after :meth:`stop` has triggered, this method needs
        to be called to obtain the solution.

        Returns
        -------
        array
            The final solution. The default implementation returns the attribute
            :attr:`x`, which child classes should set appropriately.

        """
        return self.x


from .combine_rules import CombineRules  # NOQA
from .count_iterations import CountIterations  # NOQA
from .discrepancy import Discrepancy  # NOQA

__all__ = [
    'CombineRules',
    'CountIterations',
    'Discrepancy'
]
