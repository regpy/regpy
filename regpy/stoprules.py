from copy import deepcopy
from regpy.util import ClassLogger, Errors
from regpy.operators import Operator
import numpy as np

__all__ = ["CountIterations","Discrepancy","RelativeChangeData","RelativeChangeSol","Monotonicity","DualityGapStopping"]

class MissingValueError(Exception):
    pass

class StopRule:
    """Abstract base class for stopping rules.

    The attributes :attr:`x` and :attr:`y` are set to the current iterate from the solver. The method :meth:`stop` then checks whether the stopping rule should trigger using the private method :meth:`_stop_`. If it does, then the attribute :attr:`triggered` is set to true and the method :meth:`stop` returns `True`. Note that a later call to :meth:`stop` will not evaluate the rule again since the attribute :attr:`triggered` is set to `True`. 
    """

    log = ClassLogger()

    def __init__(self):
        self.solver = None

        self.triggered = False
        """Whether the stopping rule decided to stop."""
        self.history_dict = {}
        """A place to save scalars for later use/analysis. An entry of the form {"parameter_name":[]} needs to be added in the implementation of the stopping rule."""
        self.is_main_rule = True
        r"""Whether this is the main stopping rule of the solver or a sub-rule used for example in a combined rule."""
        self.log.setLevel("INFO")

    def _complete_init_with_solver(self,solver):
        """Complete the initialisation of the stoprule by giving a solver. 
        A stoprule might reimplement this if this if the _stop_method 
        for example
        ``` 
        if isinstance(solver,specific_solver):
            self._stop = _specific_stop

        Args:
            solver (Solver): The solver the stopping rule applies to
        """
        if self.solver is not None:
            self.log.warning("the solver was already set and is now overwritten")
        self.solver = solver

    def copy_and_reset(self):
        """copy stopping rule and reset to the initial state
        """
        rule = self.copy()
        rule.reset()
        return rule
    
    def reset(self):
        """resets the stoprule to initial state
        (reset needs to be re-implemented in a stoprule if more parameters need to be reset)
        """
        self.solver = None
        self.triggered = False
        for key in self.history_dict.keys():
            self.history_dict[key] = []

    def copy(self):
        return deepcopy(self)

    


    def stop(self):
        """Check whether to stop iterations.

        Returns
        -------
        bool
            `True` if iterations should be stopped.
        """
        if self.triggered:
            return True
        self.triggered = self._stop()
        return self.triggered
    

    def _stop(self):
        """Check whether to stop iterations.

        This is an abstract method. Child classes should override it.

        Parameters and return values are the same as for the public interface
        method :meth:`stop`.

        This method will not be called again after returning `True`.


        """
        raise NotImplementedError

    def __add__(self, other):
        return CombineRules([self, other])

    def __or__(self, other):
        return CombineRules([self, other])

class NoneRule(StopRule):
    """Default stop rule that will never stop an iteration. The rule should not be used in normal setting
    it provides a default for the solvers that would stop by triggering their converged statement. 
    """

    def __init__(self):
        super().__init__()

    def _stop(self):
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
            raise TypeError(Errors.type_error(f"Combining stopping rules is only supported for a list of StopRules! You gave {rules} of type {type(rules)}"))
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
        self.history_dict = {}
        r"""Dictionary of the convergence histories of the rules."""
        for rule in rules:
            if type(rule) is type(self) and hasattr(rule,"op") and rule.op is self.op:
                self.rules.extend(rule.rules)
            else:
                self.rules.append(rule)
            self.history_dict.update(rule.history_dict)
            rule.is_main_rule = False
        
        self.active_rule = None
        r"""
        The rule that triggered the stop condition, or `None` if no rule has triggered yet.
        """
        


    def __repr__(self):
        return 'CombineRules({})'.format(self.rules)
    
    def reset(self):
        self.active_rule = None
        self.triggered = False
        self.history_dict.clear()
        for rule in self.rules:
            rule.reset()
            self.history_dict.update(rule.history_dict) 
    
    def _complete_init_with_solver(self, solver):
        self.solver = solver
        for rule in self.rules:
            rule._complete_init_with_solver(self.solver)

    def _stop(self):
        triggered =False
        self.log_info = ''
        for rule in self.rules:
            try:
                rule_triggered = rule.stop()
            except MissingValueError:
                if self.op is None or self.solver.y is not None:
                    raise
                self.solver.y = self.op(self.solver.x)
                rule_triggered = rule.stop()
            if rule_triggered:
                self.log_info += 'Rule {} triggered.'.format(rule)
                self.active_rule = rule
                triggered = True
            else:
                self.log_info = ''
        log_infos_rules = ''
        for rule in self.rules:
            log_infos_rules += rule.log_info + ' | '
        log_infos_rules = log_infos_rules[:-3]
        if self.is_main_rule:
            self.log.info(log_infos_rules+('\n' if self.log_info != '' else '')+self.log_info)
        else:
            self.log_info = '(' + log_infos_rules + (')' if self.log_info == '' else '['+self.log_info+'])')
        return triggered

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
    
    def reset(self):
        super().reset()
        self.iteration = 0

    def _stop(self):
        if self.while_type:
            self.iteration += 1
            if  self.iteration <= self.max_iterations:
                self.log_info = 'it. {}>={}'.format(self.iteration, self.max_iterations)
        else:
            self.log_info = 'it.{}>={}'.format(self.iteration, self.max_iterations)
            self.iteration += 1
        if self.is_main_rule:
            self.log.info(self.log_info)
        return self.iteration > self.max_iterations
    
######### StopRules for determining regularization parameters or for regularization by early stopping #########

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
        self.tol = self.tau
        self.history_dict["relative discrepancy"] = []

    def __repr__(self):
        return 'Discrepancy(noiselevel={}, tau={})'.format(
            self.noiselevel, self.tau)

    def _stop(self):
        if self.solver.y is None:
            raise MissingValueError
        residual = self.data - self.solver.y
        discrepancy = self.norm(residual)
        rel = discrepancy / self.noiselevel
        self.history_dict["relative discrepancy"].append(rel)
        self.log.info('relative discrepancy = {:3.2f}, tolerance = {:1.2f}'.format(rel, self.tau))
        return rel < self.tau


########## General StopRules based on relative change of data or solution ##########

class RelativeChangeData(StopRule):
    """Stops if the relative change in the residual becomes small

    Stops at the first iterate at which the difference between the old residual
    and the new residual is smaller than a pre-determined tol::

        ||y_k-y_{k+1}|| < tol

    Parameters
    ----------
    norm : callable
        The norm with respect to which the difference should be measured.
        Usually this will be the `norm` method of some :class:`~regpy.spaces.Space`.
    tol : float
        The tol value at which the iteration should be stopped
    data : np array
        The data array
    """

    def __init__(self, norm, data, tol):
        if not callable(norm):
            raise TypeError(Errors.type_error("The norm in the relative change of data stopping needs to be a callable!"))
        if not isinstance(tol,(int,float)):
            raise TypeError(Errors.type_error("The tol in the relative change of data stopping should be real scalar!"))
        if tol<=0:
            raise ValueError(Errors.value_error("The tol in the relative change of data stopping needs to be bigger then zero!"))
        super().__init__()
        self.norm = norm
        self.tol = tol
        self.data_old = data.copy()
        self.initial_data = data
        self.history_dict["relative change of y"] = []

    
    def reset(self):
        super().reset()
        self.data_old = self.initial_data
        

    def __repr__(self):
        return 'RelativeChangeData(tol={})'.format(
            self.tol)

    def _stop(self):
        if self.solver.y is None:
            raise MissingValueError
        change = self.norm(self.solver.y - self.data_old)
        self.data_old = self.solver.y.copy()
        self.history_dict["relative change of y"].append(change)
        self.log_info('rel. data change {}<{}'.format(change,self.tol))
        if self.is_main_rule:
            self.log.info(self.log_info)
        return change < self.tol


class RelativeChangeSol(StopRule):
    """Stops if the relative change in the solution space becomes small

    Stops at the first iterate at which the difference between the old estimate
    and the new estimate is smaller than a pre-determined tol::

        ||y_k-y_{k+1}|| < tol

    Parameters
    ----------
    norm : callable
        The norm with respect to which the difference should be measured.
        Usually this will be the `norm` method of some :class:`~regpy.spaces.Space`.
    tol : float
        The tol value at which the iteration should be stopped
    init : np array
        initial guess
    """

    def __init__(self, norm, init, tol):
        if not callable(norm):
            raise TypeError(Errors.type_error("The norm in the relative change of solution stopping needs to be a callable!"))
        if not isinstance(tol,(int,float)):
            raise TypeError(Errors.type_error("The tol in the relative change of solution stopping should be real scalar!"))
        if tol<=0:
            raise ValueError(Errors.value_error("The tol in the relative change of solution stopping needs to be larger than zero!"))
        super().__init__()
        self.norm = norm
        self.tol = tol
        self.sol_old = init.copy()
        self.sol_init = init
        self.history_dict["relative change of x"] = []


    def __repr__(self):
        return 'RelativeChangeSol(tol={})'.format(
            self.tol)
    
    def reset(self):
        super().reset()
        self.sol_old = self.sol_init

    def _stop(self,):
        change = self.norm(self.solver.x - self.sol_old)
        self.sol_old = self.solver.x.copy()
        self.history_dict["relative change of x"].append(change)
        self.log_info = 'rel. change sol: {}<{}'.format(change,self.tol)
        if self.is_main_rule:
            self.log.info(self.log_info)
        return change < self.tol

######### StopRules for convex optimization problems #########

class OptimalityCondStopping(StopRule):
    def __init__(self, setting, logging_level = "INFO",tol = 0.):
        if not setting.is_tikhonov and setting.is_convex:
            raise ValueError("For the optimality condition stopping rule the setting needs to be a convex and contain a regularization parameter!")
        super().__init__()
        self.setting = setting
        self.tol = tol
        self.log.setLevel(logging_level)
        self.history_dict["dSstar"] = []
        self.history_dict["dR"] = []


    def __repr__(self):
        return 'OptimalityCondStopping(tol={})'.format(
            self.tol)

    def _stop(self):
        self.solver.compute_dual()
        dSstar,dR = self.setting.violation_optimality_cond(self.solver.primal, self.solver.dual)
        self.history_dict["dSstar"].append(dSstar)
        self.history_dict["dR"].append(dR)
        stop = (dSstar+dR<=self.tol)
        self.log_info = '{:.3e} + {:.3e} = {:.3e}  <= {:.3e}'.format(dSstar,dR,dSstar+dR,self.tol)
        if self.is_main_rule:
            self.log.info(self.log_info)    
        return stop 
    
class DualityGapStopping(StopRule):
    def __init__(self, tol = 0., logging_level = "INFO"):
        super().__init__()
        self.tol = tol
        self.iteration=0
        self.log.setLevel(logging_level)
        self.history_dict["duality gap"] = []


    def __repr__(self):
        return 'DualityGapStopping(tol={})'.format(
            self.tol)
    
    def _complete_init_with_solver(self, solver):
        if not solver.setting.is_tikhonov and  solver.setting.is_convex:
            raise RuntimeError(Errors.generic_message("It is not possible to compute the dual in the implementation of this setting. The setting needs to be convex and contain a regularization parameter!"))
        return super()._complete_init_with_solver(solver)

    def _stop(self):
        self.solver.compute_dual() # sets self.primal and self.dual to new Values
        gap = self.solver.setting.duality_gap(primal = self.solver.primal, dual = self.solver.dual)
        self.history_dict["duality gap"].append(gap)
        stop = (gap<=self.tol) or (gap == np.inf)
        if gap==np.inf:
            self.log_info = 'duality gap: inf'
        else:
            self.log_info ='duality gap:{:.3e} <= {:.3e}'.format(gap,self.tol)
        if self.is_main_rule:
            self.log.info(self.log_info)    
        return stop 