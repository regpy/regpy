from . import StopRule, MissingValueError


class CombineRules(StopRule):
    """Combine several stopping rules into one.

    The resulting rule triggers when any of the given rules triggers and
    delegates selecting the solution to the active rule.

    Parameters
    ----------
    rules : list of :class:`StopRule`
        The rules to be combined.
    op : :class:`~itreg.operators.Operator`, optional
        If any rule needs the operator value and none is given to :meth:`stop`,
        the operator is used to compute it.

    Attributes
    ----------
    rules : list of :class:`StopRule`
        The combined rules.
    op : :class:`~itreg.operators.Operator` or `None`
        The forward operator.
    active_rule : :class:`StopRule` or `None`
        The rule that triggered.
    """

    def __init__(self, rules, op=None):
        super().__init__()
        self.rules = []
        for rule in rules:
            if type(rule) is type(self) and rule.op is self.op:
                self.rules.extend(rule.rules)
            else:
                self.rules.append(rule)
        self.op = op
        self.active_rule = None

    def __repr__(self):
        return 'CombineRules({})'.format(self.rules)

    def _stop(self, x, y=None):
        for rule in self.rules:
            try:
                triggered = rule.stop(x, y)
            except MissingValueError:
                if self.op is None or y is not None:
                    raise
                y = self.op(x)
                triggered = rule.stop(x, y)
            if triggered:
                self.log.info('Rule {} triggered.'.format(rule))
                self.active_rule = rule
                self.x = rule.x
                self.y = rule.y
                return True
        return False
