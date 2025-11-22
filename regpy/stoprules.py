from regpy.util import ClassLogger, Errors
from regpy.operators import Operator

__all__ = ["CountIterations","Discrepancy","RelativeChangeData","RelativeChangeSol","Monotonicity","DualityGapStopping"]

class MissingValueError(Exception):
    pass

class StopRule:
    """Abstract base class for stopping rules.

    The attributes :attr:`x` and :attr:`y` are set to the current iterate from the solver. The method :meth:`stop` then checks whether the stopping rule should trigger using the private method :meth:`_stop_`. If it does, then the attribute :attr:`triggered` is set to true and the method :meth:`stop` returns `True`. Note that a later call to :meth:`stop` will not evaluate the rule again since the attribute :attr:`triggered` is set to `True`. 
    """

    log = ClassLogger()

    def __init__(self):
        self.x = None
        """The current iterate. This is set by the solver when calling :meth:`stop`."""
        self.y = None
        """The operator value at the current iterate. This is set by the solver when calling :meth:`stop`. Can be `None` if not available."""
        self.triggered = False
        """Whether the stopping rule decided to stop."""
        self.history_dict = {}
        """A place to save scalars for later use/analysis. An entry of the form {"parameter_name":[]} needs to be added in the implementation of the stopping rule."""

    def stop(self, x, y=None,dual=None):
        """Check whether to stop iterations.

        Parameters
        ----------
        x : array
            The current iterate.
        y : array, optional
            The operator value at the current iterate. Can be omitted if
            unavailable, but some implementations may need it.
        dual : array, optional
            The iterate of the dual problem. Can be omitted if
            unavailable, but some implementations may need it.

        Returns
        -------
        bool
            `True` if iterations should be stopped.
        """
        if self.triggered:
            return True
        # self.x = x
        # self.y = y
        self.triggered = self._stop(x, y, dual)
        return self.triggered

    def _stop(self, x, y=None,dual=None):
        """Check whether to stop iterations.

        This is an abstract method. Child classes should override it.

        Parameters and return values are the same as for the public interface
        method :meth:`stop`.

        This method will not be called again after returning `True`.

        Child classes that need `y` should raise :class:`MissingValueError` if
        called with `y=None`.
        """
        raise NotImplementedError

    def __add__(self, other):
        return CombineRules([self, other])


class NoneRule(StopRule):
    """Default stop rule that will never stop an iteration. The rule should not be used in normal setting
    it provides a default for the solvers that would stop by triggering their converged statement. 
    """

    def __init__(self):
        super().__init__()

    def _stop(self, x, y=None,dual=None):
        return False

class CombineRules(StopRule):
    """Combine several stopping rules into one.

    The resulting rule triggers when any of the given rules triggers and
    delegates selecting the solution to the active rule.

    Parameters
    ----------
    rules : list of :class:`StopRule`
        The rules to be combined.
    op : :class:`~regpy.operators.Operator`, optional
        If any rule needs the operator value and none is given to :meth:`stop`,
        the operator is used to compute it.
    """

    def __init__(self, rules, op=None):
        if not isinstance(rules,(list,tuple)) or any(not isinstance(rule,StopRule) for rule in rules):
            raise TypeError(Errors.type_error(f"Combining stopping rules is only supported for a list of StopRules!"))
        if op is not None and not isinstance(op,Operator):
            raise TypeError(Errors.type_error("The operator that is passed to the combined rules needs to be either None or an Operator!"))
        super().__init__()
        self.rules = []
        r"""List of :class:`StopRule` the combined rules.
        """
        self.op = op
        r""":class:`~regpy.operators.Operator` or `None`
        The forward operator.
        """
        for rule in rules:
            if type(rule) is type(self) and hasattr(rule,"op") and rule.op is self.op:
                self.rules.extend(rule.rules)
            else:
                self.rules.append(rule)
        self.active_rule = None
        r"""
        The rule that triggered the stop condition, or `None` if no rule has triggered yet.
        """

    def __repr__(self):
        return 'CombineRules({})'.format(self.rules)

    def _stop(self, x, y=None,dual=None):
        for rule in self.rules:
            try:
                triggered = rule.stop(x, y, dual)
            except MissingValueError:
                if self.op is None or y is not None:
                    raise
                y = self.op(x)
                triggered = rule.stop(x, y, dual)
            if triggered:
                self.log.info('Rule {} triggered.'.format(rule))
                self.active_rule = rule
                self.x = rule.x
                self.y = rule.y
                return True
        return False


class CountIterations(StopRule):
    """Stopping rule based on number of iterations.

    Each call to :attr:`stop` increments the iteration count by 1.

    Parameters
    ----------
    max_iterations : int
        The number of iterations after which to stop.
    """

    def __init__(self, max_iterations, while_type = True,logging_level= "INFO"):
        if not isinstance(max_iterations,int):
            raise TypeError(Errors.type_error("The maximal iteration in the CountIterations should be an integer!"))
        if max_iterations<0:
            raise ValueError(Errors.value_error("The maximal iteration in CountIteration needs to be at least zero (for no iteration)!"))
        super().__init__()
        self.max_iterations = max_iterations
        self.iteration = 0
        self.while_type = while_type
        self.log.setLevel(logging_level)

    def __repr__(self):
        return 'CountIterations(max_iterations={})'.format(self.max_iterations)

    def _stop(self, x, y=None,dual=None):
        if self.while_type:
            self.iteration += 1
            if  self.iteration <= self.max_iterations:
                self.log.info(
                    'iteration = {} / {}'
                    .format(self.iteration, self.max_iterations))
        else:
            self.log.info(
                'iteration = {} / {}'
                .format(self.iteration, self.max_iterations))
            self.iteration += 1
        return self.iteration > self.max_iterations

class Discrepancy(StopRule):
    """Morozov's discrepancy principle.

    Stops at the first iterate at which the residual is smaller than a
    pre-determined multiple of the noise level::

        ||y - data|| < tau * noiselevel

    Parameters
    ----------
    norm : callable
        The norm with respect to which the discrepancy should be measured.
        Usually this will be the `norm` method of some :class:`~regpy.spaces.Space`.
    data : array
        The right hand side (noisy data).
    noiselevel : float
        An estimate of the distance from the noisy data to the exact data.
    tau : float, optional
        The multiplier; must be larger than 1. Defaults to 2.
    """

    def __init__(self, norm, data, noiselevel, tau=2):
        if not callable(norm):
            raise TypeError(Errors.type_error("The norm in the discrepancy principle needs to be a callable!"))
        if not isinstance(noiselevel,(int,float)):
            raise TypeError(Errors.type_error("The noise level in the discrepancy principle should be real scalar!"))
        if noiselevel<=0:
            raise ValueError(Errors.value_error("The noise level in the discrepancy principle needs to be bigger then zero!"))
        if not isinstance(tau,(int,float)):
            raise TypeError(Errors.type_error("The multiplier in the discrepancy principle should be real scalar!"))
        if tau<=1:
            raise ValueError(Errors.value_error("The multiplier in the discrepancy principle needs to be bigger then one!"))
        super().__init__()
        self.norm = norm
        self.data = data
        self.noiselevel = noiselevel
        self.tau = tau
        self.history_dict["relative discrepancy"] = []
    def __repr__(self):
        return 'Discrepancy(noiselevel={}, tau={})'.format(
            self.noiselevel, self.tau)

    def _stop(self, x, y=None,dual=None):
        if y is None:
            raise MissingValueError
        residual = self.data - y
        discrepancy = self.norm(residual)
        rel = discrepancy / self.noiselevel
        self.history_dict["relative discrepancy"].append(rel)
        self.log.info('relative discrepancy = {:3.2f}, tolerance = {:1.2f}'.format(rel, self.tau))
        return rel < self.tau


class RelativeChangeData(StopRule):
    """Stops if the relative change in the residual becomes small

    Stops at the first iterate at which the difference between the old residual
    and the new residual is smaller than a pre-determined cutoff::

        ||y_k-y_{k+1}|| < delta

    Parameters
    ----------
    norm : callable
        The norm with respect to which the difference should be measured.
        Usually this will be the `norm` method of some :class:`~regpy.spaces.Space`.
    cutoff : float
        The cutoff value at which the iteration should be stopped
    data : np array
        The data array
    """

    def __init__(self, norm, data, cutoff):
        if not callable(norm):
            raise TypeError(Errors.type_error("The norm in the relative change of data stopping needs to be a callable!"))
        if not isinstance(cutoff,(int,float)):
            raise TypeError(Errors.type_error("The cutoff in the relative change of data stopping should be real scalar!"))
        if cutoff<=0:
            raise ValueError(Errors.value_error("The cutoff in the relative change of data stopping needs to be bigger then zero!"))
        super().__init__()
        self.norm = norm
        self.cutoff = cutoff
        self.data_old = data
        self.history_dict["relative change of y"] = []

    def __repr__(self):
        return 'RelativeChangeData(cutoff={})'.format(
            self.cutoff)

    def _stop(self, x, y=None,dual=None):
        if y is None:
            raise MissingValueError
        change = self.norm(y - self.data_old)
        self.data_old = y.copy()
        self.history_dict["relative change of y"].append(change)
        self.log.info('RelativeChangeData = {}, cutoff = {}'.format(
            change, self.cutoff))
        return change < self.cutoff


class RelativeChangeSol(StopRule):
    """Stops if the relative change in the solution space becomes small

    Stops at the first iterate at which the difference between the old estimate
    and the new estimate is smaller than a pre-determined cutoff::

        ||y_k-y_{k+1}|| < cutoff

    Parameters
    ----------
    norm : callable
        The norm with respect to which the difference should be measured.
        Usually this will be the `norm` method of some :class:`~regpy.spaces.Space`.
    cutoff : float
        The cutoff value at which the iteration should be stopped
    init : np array
        initial guess
    """

    def __init__(self, norm, init, cutoff):
        if not callable(norm):
            raise TypeError(Errors.type_error("The norm in the relative change of solution stopping needs to be a callable!"))
        if not isinstance(cutoff,(int,float)):
            raise TypeError(Errors.type_error("The cutoff in the relative change of solution stopping should be real scalar!"))
        if cutoff<=0:
            raise ValueError(Errors.value_error("The cutoff in the relative change of solution stopping needs to be bigger then zero!"))
        super().__init__()
        self.norm = norm
        self.cutoff = cutoff
        self.sol_old = init
        self.history_dict["relative change of x"] = []

    def __repr__(self):
        return 'RelativeChangeSol(cutoff={})'.format(
            self.cutoff)

    def _stop(self, x, y=None,dual=None):
        change = self.norm(x - self.sol_old)
        self.sol_old = x.copy()
        self.history_dict["relative change of x"].append(change)
        self.log.info('RelativeChangeSol = {}, cutoff = {}'.format(
            change, self.cutoff))
        return change < self.cutoff


class Monotonicity(StopRule):
    """Stops if the residual is growing again.

    Parameters
    ----------
    norm : callable
        The norm with respect to which the difference should be measured.
        Usually this will be the `norm` method of some :class:`~regpy.spaces.Space`.
    data : np array
        The data array
    init_data : np array
        initial guess in data space
    """

    def __init__(self, norm, data, init_data):
        if not callable(norm):
            raise TypeError(Errors.type_error("The norm in the monotonicity stopping needs to be a callable!"))
        super().__init__()
        self.norm = norm
        self.data = data
        self.residual = self.norm(self.data - init_data)
        self.history_dict["monotonicity"] = []
        self.history_dict["residual"] = []

    def __repr__(self):
        return 'Monotonicty'

    def _stop(self, x, y=None,dual=None):
        if y is None:
            raise MissingValueError
        residual = self.norm(self.data - y)
        change = self.residual - residual
        self.history_dict["monotonicity"].append(change)
        self.history_dict["residual"].append(residual)
        self.residual = residual
        self.log.info('Monotonicity = {}, residual = {}'.format(
            change, residual))
        #self.log.info('Monotonicity = {}'.format(
        #    change))
        return change < 0


class DualityGapStopping(StopRule):
    def __init__(self, setting, threshold = None,max_iter=1000, logging_level = "INFO",cutoff = 0.):
        from regpy.solvers import TikhonovRegularizationSetting
        if not isinstance(setting,TikhonovRegularizationSetting):
            raise TypeError(Errors.not_instance(setting,TikhonovRegularizationSetting,add_info="For the Duality gap stopping rule the setting needs to be a TikhonovRegularizationSetting!"))
        super().__init__()
        self.setting = setting
        if threshold is not None:
            self.cutoff= threshold
        else:
            self.cutoff = cutoff
        self.log.setLevel(logging_level)
        self.history_dict["duality gap"] = []

    def __repr__(self):
        return 'DualityGapStopping(cutoff={})'.format(
            self.cutoff)

    def _stop(self, x, y=None, dual=None):
        if dual is not None:
            gap = self.setting.dualityGap(primal = x, dual = dual)
        elif y is not None:
            gap = self.setting.dualityGap(primal = x,dual=self.setting.primalToDual(y,argumentIsOperatorImage=True))
        else:
            gap = self.setting.dualityGap(primal = x)
        self.history_dict["duality gap"].append(gap)
        gap_stop = gap<=self.cutoff
        self.log.info('duality gap={:.3e}, threshold  = {:.3e}'.format(gap,self.cutoff))      
        return gap_stop 