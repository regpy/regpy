import logging

from . import StopRule

__all__ = ['Discrepancy']


class Discrepancy(StopRule):
    """Morozov's discrepancy principle.

    Stops at the first iterate at which the residual is smaller than a
    pre-determined multiple of the noise level::

        norm(y - data) < tau * noiselevel

    This rule (obviously) :attr:`needs_y`.

    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side (noisy data).
    noiselevel : float
        An estimate of the distance from the noisy data to the exact data,
        measured in the norm of `op.domy`.
    tau : float, optional
        The multiplier; must be larger than 1. Defaults to 2.

    """

    def __init__(self, op, data, noiselevel, tau=2):
        super().__init__(logging.getLogger(__name__))
        self.op = op
        self.data = data
        self.noiselevel = noiselevel
        self.tau = tau
        self.needs_y = True

    def __repr__(self):
        return 'Discrepancy(noiselevel={}, tau={})'.format(
            self.noiselevel, self.tau)

    def stop(self, x, y=None):
        self.x = x
        if y is None:
            y = self.op(x)
        residual = self.data - y
        discrepancy = self.op.domy.norm(residual)
        self.log.info(
            'discrepancy = {}, noiselevel = {}, tau = {}'
            .format(discrepancy, self.noiselevel, self.tau))
        return discrepancy < self.noiselevel * self.tau
