import logging

from .stop_rule import StopRule

__all__ = ['CountIterations']

log = logging.getLogger(__name__)


class CountIterations(StopRule):
    def __init__(self, max_iterations):
        super().__init__(log)
        self.max_iterations = max_iterations
        self.iteration = 0

    def __repr__(self):
        return 'CountIterations(max_iterations={})'.format(self.max_iterations)

    def stop(self, x, y=None):
        self.iteration += 1
        if self.iteration > self.max_iterations:
            self.x = x
            return True
        return False
