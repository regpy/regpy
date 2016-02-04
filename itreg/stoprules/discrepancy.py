import logging

from .stop_rule import StopRule

__all__ = ['Discrepancy']

log = logging.getLogger(__name__)


class Discrepancy(StopRule):
    def __init__(self, op, data, noiselevel, tau=2):
        super().__init__(log)
        self.op = op
        self.data = data
        self.noiselevel = noiselevel
        self.tau = tau
        self.needs_y = True

    def __repr__(self):
        return 'Discrepancy(noiselevel={}, tau={})'.format(
            self.noiselevel, self.tau)

    def stop(self, x, y=None):
        if y is None:
            y = self.op(x)
        residual = self.data - y
        discrepancy = self.op.domy.norm(residual)
        if discrepancy < self.noiselevel * self.tau:
            self.log.info(
                'Rule triggered: discrepancy = {}, noiselevel = {}, tau = {}'
                .format(discrepancy, self.noiselevel, self.tau))
            self.x = x
            return True
        return False
