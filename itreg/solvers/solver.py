import logging


class Solver(object):
    def __init__(self, log=logging.getLogger()):
        self.log = log

    def next(self):
        raise NotImplementedError()

    def run(self, stoprule):
        while True:
            if stoprule.stop(self.x, self.y):
                self.log.info('Stopping rule triggered.')
                return stoprule.select()
            if not self.next():
                self.log.info('Solver converged.')
                return self.x
