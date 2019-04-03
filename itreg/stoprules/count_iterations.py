from . import StopRule


class CountIterations(StopRule):
    """Stopping rule based on number of iterations.

    Each call to :attr:`stop` increments the iteration count by 1.

    Parameters
    ----------
    max_iterations : int
        The number of iterations after which to stop.
    """

    def __init__(self, max_iterations):
        super().__init__()
        self.max_iterations = max_iterations
        self.iteration = 0

    def __repr__(self):
        return 'CountIterations(max_iterations={})'.format(self.max_iterations)

    def _stop(self, x, y=None):
        self.iteration += 1
        self.log.info(
            'iteration = {} / {}'
            .format(self.iteration, self.max_iterations))
        return self.iteration >= self.max_iterations
