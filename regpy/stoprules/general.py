"""General stopping-rule infrastructure.

This module contains the base class, rule combinators, and the iteration-count rule.
"""

from copy import deepcopy
import logging
import time

from regpy.util import ClassLogger, Errors

__all__ = [
    "MissingValueError", "StopRule", "NoneRule", "CombineRules",
    "AndCombineRules", "CountIterations",
]

class MissingValueError(Exception):
    pass


def _copy_vector(space, vector):
    result = space.empty()
    space.copyto(result, vector)
    return result

class StopRule:
    r"""Abstract base class for stopping rules.

    :meth:`stop` attaches the solver on first use and delegates the decision to
    :meth:`_stop`. Once a rule triggers, :attr:`triggered` remains true and
    subsequent calls return immediately without evaluating the rule again.
    """

    log = ClassLogger()

    def __init__(self, logging_level = "INFO"):
        self.solver = None

        self.triggered = False
        """Whether the stoprule decided to stop."""
        self.history_dict = {}
        """A place to save scalars for later use/analysis. An entry of the form {"parameter_name":[]} needs to be added in the implementation of the stoprule."""
        self.is_main_rule = True
        r"""Whether this is the main stoprule of the solver or a sub-rule used for example in a combined rule."""
        self.log.setLevel(logging_level)

    def _complete_init_with_solver(self,solver):
        """Attach a solver and complete rule-specific initialization.

        Subclasses may override this hook when their criterion needs
        information from the concrete solver.

        Parameters
        ----------
        solver : regpy.solvers.Solver
            Solver to which this rule is attached.
        """
        if self.solver is not None:
            self.log.warning("the solver was already set and is now overwritten")
        self.solver = solver

    def copy_and_reset(self):
        """copy stoprule and reset to the initial state
        """
        rule = self.copy()
        rule.reset()
        return rule
    
    def reset(self):
        """Reset the rule to its initial state.

        Subclasses that maintain additional state must override this method and
        reset that state as well.
        """
        self.solver = None
        self.triggered = False
        if hasattr(self,"x"):
            del self.x
        if hasattr(self,"y"):
            del self.y

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
        should_stop = self._stop()
        if should_stop and self.is_main_rule:
            self.trigger()
        return should_stop

    def trigger(self):
        """Force the stoprule to trigger at the current iterate.

        This sets the :attr:`triggered` attribute to `True` and stores
        the current iterate in :attr:`x` and :attr:`y`.
        """
        self.triggered = True
        self.x = self.solver.x
        self.y = self.solver.y if hasattr(self.solver,"y") else None

    def best_iterate(self):
        """Return the best iterate according to this stoprule. 
        By default, this is the last iterate computed before the stoprule triggered.

        However, since iterative methods for ill-posed problems typically exhibit a semi-convergent behaviour, the best iterate is not necessarily the last one computed before the stoprule triggered. 

        Returns
        -------
        If the connected solver converges or the stop rule was triggered:
        x : array
            The best solution found.
        y : array
            The image of the best solution under the operator.
        
        or 
        
        None
            If neither the solver converged nor the stop rule was triggered. 

        Raise
        -----
        RuntimeError
            Whenever the solver converged or the stop rule was triggered but the stop rule does not have an `x`
            attribute.
        """
        if self.solver is not None and self.solver.is_converged():
            self.x = self.solver.x
            self.y = self.solver.y if hasattr(self.solver,"y") else None
        elif not self.triggered:
            self.log.warning("The stoprule has not triggered yet and the solver has not converged, so no best iterate is available!")
            return None
        if not hasattr(self,"x"):
            raise RuntimeError(Errors.generic_message("The stoprule did not store self.x when triggered! Please re-implement the best_iterate method of the stoprule!"))
        if not hasattr(self,"y"):
            self.y = None
        return self.x, self.y

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
    
    def __radd__(self, other):
        # Adding zero should not change the rule, this allows to use sum() to combine rules.
        if other == 0:
            return self
        return NotImplemented

    def __or__(self, other):
        return CombineRules([self, other])

    def __and__(self, other):
        return AndCombineRules([self, other])

class NoneRule(StopRule):
    r"""Default stop rule that will never stop an iteration. The rule should not be used in normal setting
    it provides a default for the solvers that would stop by triggering their converged statement. 
    """

    def __init__(self):
        super().__init__()
        self.triggered = True

    def _stop(self):
        return False
    
    def best_iterate(self):
        return self.solver.x, self.solver.y if hasattr(self.solver,"y") else None

class CombineRules(StopRule):
    r"""Combine several stopping rules into one that stops if one of the rules stops. (logical OR)

    The resulting rule triggers when any of the given rules triggers.
    The first rule is responsible for selecting the best solution.

    Parameters
    ----------
    rules : list of :class:`StopRule`
        The rules to be combined.
    """

    def __init__(self, rules:list[StopRule],logging_level=None):
        if not isinstance(rules,(list,tuple)) or any(not isinstance(rule,StopRule) for rule in rules):
            raise TypeError(Errors.type_error(f"Combining stopping rules is only supported for a list of stopping rules! You gave {rules} of type {type(rules)}"))
        super().__init__()
        self.rules = []
        r"""List of :class:`StopRule` the combined rules.
        """
        self.history_dict = {}
        r"""Dictionary of the convergence histories of the rules."""
        for rule in rules:
            if type(rule) is type(self) and hasattr(rule,"solver") and rule.solver is self.solver:
                self.rules.extend(rule.rules)
            else:
                self.rules.append(rule)
            self.history_dict.update(rule.history_dict)
            rule.is_main_rule = False
        if logging_level is None:
            self.log.setLevel(min([rule.log.level if hasattr(rule,'log') else 50 for rule in self.rules]))
        else:
            self.log.setLevel(logging_level)
        
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
                if self.solver is None or (self.solver is not None and self.solver.op is None): 
                    raise RuntimeError(Errors.generic_message("One of the combined stopping rules needs the operator value to evaluate the stopping condition. Please provide the operator to the solver or make sure that the solver computes the operator value before calling the stopping rules."))
                self.solver.y = self.solver.op(self.solver.x)
                rule_triggered = rule.stop()
            if rule_triggered:
                self.log_info += 'Rule {} triggered.'.format(rule)
                self.active_rule = rule
                self.active_rule.trigger()
                self.triggered = True
                self.x, self.y  =self.active_rule.x, self.active_rule.y                
                triggered = True
            else:
                self.log_info = ''
        log_infos_rules = ''
        for rule in self.rules:
            log_infos_rules += rule.log_info + ' | '
        log_infos_rules = log_infos_rules[:-3]
        if self.is_main_rule:
            self.log.log(logging.INFO if self.triggered else logging.DEBUG,
                         log_infos_rules+('\n' if self.log_info != '' else '')+self.log_info
                         )
        else:
            self.log_info = '(' + log_infos_rules + (')' if self.log_info == '' else '['+self.log_info+'])')
        return triggered

class AndCombineRules(StopRule):
    r"""Combine several stopping rules into one that stops if all of the rules stop.

    The resulting rule triggers when all of the given rules trigger. 
    It delegates selecting the solution to the first rule.

    Parameters
    ----------
    rules : list of :class:`StopRule`
        The rules to be combined.
    """

    def __init__(self, rules:list[StopRule],logging_level = None):
        if not isinstance(rules,(list,tuple)) or any(not isinstance(rule,StopRule) for rule in rules):
            raise TypeError(Errors.type_error(f"Combining stopping rules is only supported for a list of stopping rules! You gave {rules} of type {type(rules)}"))
        super().__init__()
        self.rules = []
        r"""List of :class:`StopRule` the combined rules.
        """
        self.history_dict = {}
        r"""Dictionary of the convergence histories of the rules."""
        for rule in rules:
            if type(rule) is type(self) and hasattr(rule,"solver") and rule.solver is self.solver:
                self.rules.extend(rule.rules)
            else:
                self.rules.append(rule)
            self.history_dict.update(rule.history_dict)
            rule.is_main_rule = False
        if logging_level is None:         
            self.log.setLevel(min([rule.log.level if hasattr(rule,'log') else 50 for rule in self.rules]))
        else:
            self.log.setLevel(logging_level)

    def __repr__(self):
        return 'AndCombineRules({})'.format(self.rules)
    
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
        triggered =True
        self.log_info = ''
        for rule in self.rules:
            try:
                rule_triggered = rule.stop()
            except MissingValueError:
                if self.solver is None or (self.solver is not None and self.solver.op is None): 
                    raise RuntimeError(Errors.generic_message("One of the combined stopping rules needs the operator value to evaluate the stopping condition. Please provide the operator to the solver or make sure that the solver computes the operator value before calling the stopping rules."))
                self.solver.y = self.solver.op(self.solver.x)
                rule_triggered = rule.stop()
            if not rule_triggered:
                triggered = False
            else:
                self.log_info = ''
        if triggered:
            self.log_info += 'All rules triggered.'
            self.triggered = True
            self.rules[0].trigger() # first rule decides best iterate
            self.x, self.y = self.rules[0].x, self.rules[0].y
        log_infos_rules = ''
        for rule in self.rules:
            log_infos_rules += rule.log_info + ' & '
        log_infos_rules = log_infos_rules[:-3]
        if self.is_main_rule:
            self.log.log(logging.INFO if self.triggered else logging.DEBUG,
                        log_infos_rules+('\n' if self.log_info != '' else '')+self.log_info
                        )
        else:
            self.log_info = '(' + log_infos_rules + (')' if self.log_info == '' else '['+self.log_info+'])')
        return triggered
    

class CountIterations(StopRule):
    import time
    r"""Stopping rule based on number of iterations.

    Each call to :attr:`stop` increments the iteration count by 1.

    Parameters
    ----------
    max_iterations : int
        The number of iterations after which to stop.
    while_type :bool, optional 
        Indicates if this StopRule is used in a while (Default: True) or an until loop.
    measure_time: bool, optional
        If True, the cpu time elapsed since the start of the iteration is saved for each call in self.history_dict

    """

    def __init__(self, max_iterations:int, 
                 while_type:bool = True,
                 timing:bool=False,
                 logging_level:str= "INFO"):
        if not isinstance(max_iterations,int):
            raise TypeError(Errors.type_error("The maximal iteration in the CountIterations should be an integer!"))
        if max_iterations<0:
            raise ValueError(Errors.value_error("The maximal iteration in CountIteration needs to be at least zero (for no iteration)!"))
        super().__init__()
        self.max_iterations = max_iterations
        self.iteration = 0
        self.while_type = while_type
        self.timing = timing
        if timing: 
            self.history_dict['time'] = []
            self.start_time = time.process_time()
        self.log.setLevel(logging_level)


    def __repr__(self):
        return 'CountIterations(max_iterations={})'.format(self.max_iterations)
    
    def reset(self):
        super().reset()
        self.iteration = 0

    def _stop(self):
        if self.timing:
            self.history_dict['time'].append(time.process_time()-self.start_time)
        if self.while_type:
            triggered = self.iteration >= self.max_iterations
            self.log_info = 'it. {}>={}'.format(self.iteration, self.max_iterations)
            self.iteration += 1
        else:
            self.iteration += 1
            triggered = self.iteration >= self.max_iterations
            self.log_info = 'it. {}>={}'.format(self.iteration, self.max_iterations)
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)
        return triggered
    
######### stopping rules for determining regularization parameters or for regularization by early stopping #########

