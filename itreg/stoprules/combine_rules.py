import logging

from .stop_rule import StopRule

__all__ = ['CombineRules']

log = logging.getLogger(__name__)


class CombineRules(StopRule):
    def __init__(self, rules, op=None):
        super().__init__(log)
        self.rules = rules
        self.needs_y = any([rule.needs_y for rule in self.rules])
        self.op = op

    def stop(self, x, y=None):
        if y is None and self.needs_y and self.op is not None:
            y = self.op(x)
        for rule in self.rules:
            if rule.stop(x, y):
                self.log.info('Stopping rule {} triggered.'.format(rule))
                self.active_rule = rule
                return True
        return False

    def select(self):
        return self.active_rule.select()
