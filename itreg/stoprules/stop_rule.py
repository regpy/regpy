import logging


class StopRule(object):
    def __init__(self, log=logging.getLogger()):
        self.log = log
        self.needs_y = False

    def stop(self, x, y=None):
        raise NotImplementedError()

    def select(self):
        return self.x
