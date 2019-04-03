from itreg.util import classlogger


class MissingValueError(Exception):
    pass


class StopRule:
    """Abstract base class for stopping rules.

    Attributes
    ----------
    x, y : arrays or `None`
        The chosen solution. Stopping rules may decide to yield a result
        different from the last iterate. Therefore, after :meth:`stop` has
        triggered, it should store the solution in this attribute. Before
        triggering, these attributes contain the iterates passed to
        :meth:`stop`.
    triggered: bool
        Whether the stopping rule decided to stop.
    """

    log = classlogger

    def __init__(self):
        self.x = None
        self.y = None
        self.triggered = False

    def stop(self, x, y=None):
        """Check whether to stop iterations.

        Parameters
        ----------
        x : array
            The current iterate.
        y : array, optional
            The operator value at the current iterate. Can be omitted if
            unavailable, but some implementations may need it.

        Returns
        -------
        bool
            `True` if iterations should be stopped.
        """
        if self.triggered:
            return True
        self.x = x
        self.y = y
        self.triggered = self._stop(x, y)
        return self.triggered

    def _stop(self, x, y=None):
        """Check whether to stop iterations.

        This is an abstract method. Child classes should override it.

        Parameters and return values are the same as for the public interface
        method :meth:`stop`.

        This method will not be called again after returning `True`.

        Child classes that need `y` should raise :class:`MissingValueError` if
        called with `y=None`.
        """
        raise NotImplementedError

    def __add__(self, other):
        return CombineRules([self, other])


from .combine_rules import CombineRules
from .count_iterations import CountIterations
from .discrepancy import Discrepancy
