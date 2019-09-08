from . import StopRule, MissingValueError


class Discrepancy(StopRule):
    """Morozov's discrepancy principle.

    Stops at the first iterate at which the residual is smaller than a
    pre-determined multiple of the noise level::

        ||y - data|| < tau * noiselevel

    Parameters
    ----------
    norm : callable
        The norm with respect to which the discrepancy should be measured.
        Usually this will be the `norm` method of some :class:`~itreg.spaces.Space`.
    data : array
        The right hand side (noisy data).
    noiselevel : float
        An estimate of the distance from the noisy data to the exact data.
    tau : float, optional
        The multiplier; must be larger than 1. Defaults to 2.
    """

    def __init__(self, norm, data, noiselevel, tau=2):
        super().__init__()
        self.norm = norm
        self.data = data
        self.noiselevel = noiselevel
        self.tau = tau

    def __repr__(self):
        return 'Discrepancy(noiselevel={}, tau={})'.format(
            self.noiselevel, self.tau)

    def _stop(self, x, y=None):
        if y is None:
            raise MissingValueError
        residual = self.data - y
        discrepancy = self.norm(residual)
        rel = discrepancy / self.noiselevel
        self.log.info('relative discrepancy = {}, tolerance = {}'.format(rel, self.tau))
        return rel < self.tau
