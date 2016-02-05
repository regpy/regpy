import logging

from . import StopRule

__all__ = ['CombineRules']


class CombineRules(StopRule):
    """Combine several stopping rules into one.

    The resulting rule triggers when any of the given rules triggers and
    delegates selecting the solution to the active rule.

    It tries to handle the case when any rule :attr:`needs_y` appropriately.

    Parameters
    ----------
    rules : list of :class:`StopRule`
        The rules to be combined.
    op : :class:`Operator <itreg.operators.Operator>`, optional
        If any rule :attr:`needs_y` and none is given to :meth:`stop`, the
        operator is used to compute the value in advance.

    Attributes
    ----------
    rules : list of :class:`StopRule`
        The combined rules.
    op : :class:`Operator <itreg.operators.Operator>` or `None`
        The forward operator.
    active_rule : :class:`StopRule` or `None`
        The rule that triggered.

    """

    def __init__(self, rules, op=None):
        super().__init__(logging.getLogger(__name__))
        self.rules = rules
        self.needs_y = any([rule.needs_y for rule in self.rules])
        self.op = op
        self.active_rule = None

    def __repr__(self):
        return 'CombineRules({})'.format(self.rules)

    def stop(self, x, y=None):
        if y is None and self.needs_y and self.op is not None:
            y = self.op(x)
        for rule in self.rules:
            if rule.stop(x, y):
                self.log.info('Rule {} triggered.'.format(rule))
                self.active_rule = rule
                self.x = rule.x
                return True
        return False
