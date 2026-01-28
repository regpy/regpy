from copy import deepcopy
from regpy.util import ClassLogger, Errors
from typing import Callable
import numpy as np

__all__ = ["CountIterations","Discrepancy","RelativeChangeData","RelativeChangeSol","Monotonicity","DualityGapStopping","LCurve","OptimalityCondStopping","CombineRules","AndCombineRules","NoneRule"]

class MissingValueError(Exception):
    pass

class StopRule:
    r"""Abstract base class for stopping rules.

    The attributes :attr:`x` and :attr:`y` are set to the current iterate from the solver. The method :meth:`stop` then checks whether the stopping rule should trigger using the private method :meth:`_stop_`. If it does, then the attribute :attr:`triggered` is set to true and the method :meth:`stop` returns `True`. Note that a later call to :meth:`stop` will not evaluate the rule again since the attribute :attr:`triggered` is set to `True`. 
    """

    log = ClassLogger()

    def __init__(self, logging_level = "INFO"):
        self.solver = None

        self.triggered = False
        """Whether the stopping rule decided to stop."""
        self.history_dict = {}
        """A place to save scalars for later use/analysis. An entry of the form {"parameter_name":[]} needs to be added in the implementation of the stopping rule."""
        self.is_main_rule = True
        r"""Whether this is the main stopping rule of the solver or a sub-rule used for example in a combined rule."""
        self.log.setLevel(logging_level)

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
        """Force the stopping rule to trigger at the current iterate.

        This sets the :attr:`triggered` attribute to `True` and stores
        the current iterate in :attr:`x` and :attr:`y`.
        """
        self.triggered = True
        self.x = self.solver.x
        self.y = self.solver.y if hasattr(self.solver,"y") else None

    def best_iterate(self):
        """Return the best iterate according to this stopping rule. 
        By default, this is the last iterate computed before the stopping rule triggered.

        However, since iterative methods for ill-posed problems typically exhibit a semi-convergent behaviour, the best iterate is not necessarily the last one computed before the stopping rule triggered. 

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
            self.log.warning("The stopping rule has not triggered yet and the solver has not converged, so no best iterate is available!")
            return None
        if not hasattr(self,"x"):
            raise RuntimeError(Errors.generic_message("The stopping rule did not store self.x when triggered! Please re-implement the best_iterate method of the stopping rule!"))
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

    def __init__(self, rules:list[StopRule]):
        if not isinstance(rules,(list,tuple)) or any(not isinstance(rule,StopRule) for rule in rules):
            raise TypeError(Errors.type_error(f"Combining stopping rules is only supported for a list of StopRules! You gave {rules} of type {type(rules)}"))
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
            self.log.info(log_infos_rules+('\n' if self.log_info != '' else '')+self.log_info)
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

    def __init__(self, rules:list[StopRule]):
        if not isinstance(rules,(list,tuple)) or any(not isinstance(rule,StopRule) for rule in rules):
            raise TypeError(Errors.type_error(f"Combining stopping rules is only supported for a list of StopRules! You gave {rules} of type {type(rules)}"))
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
            self.log.info(log_infos_rules+('\n' if self.log_info != '' else '')+self.log_info)
        else:
            self.log_info = '(' + log_infos_rules + (')' if self.log_info == '' else '['+self.log_info+'])')
        return triggered
    

class CountIterations(StopRule):
    r"""Stopping rule based on number of iterations.

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
            triggered = self.iteration >= self.max_iterations
            self.log_info = 'it. {}>={}'.format(self.iteration, self.max_iterations)
            self.iteration += 1
        else:
            self.iteration += 1
            triggered = self.iteration >= self.max_iterations
            self.log_info = 'it. {}>={}'.format(self.iteration, self.max_iterations)
        if self.is_main_rule:
            self.log.info(self.log_info)
        return triggered
    
######### StopRules for determining regularization parameters or for regularization by early stopping #########

class Discrepancy(StopRule):
    r"""Morozov's discrepancy principle.

    Stops at the first iterate at which the residual is smaller than a
    pre-determined multiple of the noise level::

        ||y - data|| < tau * noiselevel

    Parameters
    ----------
    noiselevel : float
        An estimate of the distance from the noisy data to the exact data.
    setting: Setting| None, optional
        setting, default: None. In the default case, data and norm must be given.  
    data : array or None, optional
        The right hand side (noisy data) or None (default). 
        In the default case, setting.data is used.   
    norm : callable or None, optional
        The norm with respect to which the discrepancy should be measured or None (default).
        In the default case, setting.h_codomain.norm is used.            
    tau : float, optional
        The multiplier; must be larger than 1. Defaults to 2.
    noise_level_is_relative: bool, optional
        Indicates whether the given noiselevel is a relative or absolute noise level. Defaults to False
    """
    #def __init__(self, 
    #             noiselevel:float, 
    #             tau:float=2.,
    #             setting =None,
    #             data=None,
    #             norm: Callable|None =None,
    #             noise_level_is_relative:bool=False
    #            ):
    def __init__(self,*args,**kwargs):
        defaults = {'tau':2.,'setting':None,'data':None,'norm':None,'noise_level_is_relative':False}
        if len(args)==3: 
            self.log.warning('Initialization of Discrepancy with three positional arguments (norm, data, noise_level) deprecated. Use one positional argument (noiselevel) and specifiy norm and data via a keyword argument setting!')
            norm,data,noiselevel = args
            p = {**defaults, **kwargs}
            tau,noise_level_is_relative,setting = p['tau'],p['noise_level_is_relative'],p['setting']
        elif len(args)==2:
            self.log.warning('Initialization of Discrepancy with two positional arguments (norm, data) deprecated. Use one positional argument (noiselevel) and specifiy norm and data via a keyword argument setting!')
            norm,data = args
            p = {**defaults, **kwargs}
            tau,noise_level_is_relative,setting,noiselevel = p['tau'],p['noise_level_is_relative'],p['setting'],p['noiselevel']          
        elif len(args)==1:
            noiselevel = args[0]
            p = {**defaults, **kwargs}
            tau,noise_level_is_relative,setting,norm,data = p['tau'],p['noise_level_is_relative'],p['setting'],p['norm'],p['data']
        else:
            raise ValueError(Errors.value_error('Discrepancy must have either two (deprecated) or one (recommended) positional arguments.'))
        from regpy.solvers import Setting
        if not isinstance(noise_level_is_relative,bool):
            raise TypeError(Errors.type_error("noise_level_is_relative must be boolean."))
        if not isinstance(noiselevel,(int,float)):  
            raise TypeError(Errors.type_error(f"The noise level in the discrepancy principle should be real scalar! Got {noiselevel}"))
        if noiselevel<=0:
            raise ValueError(Errors.value_error(f"The noise level in the discrepancy principle needs to be bigger then zero! Got {noiselevel}"))
        if not isinstance(tau,(int,float)):
            raise TypeError(Errors.type_error(f"The multiplier in the discrepancy principle should be real scalar! Got {tau}."))
        if tau<=1:
            self.log.warning("The multiplier in the discrepancy principle should be bigger than one!")
        if norm is not None:
            if not callable(norm):
                raise TypeError(Errors.type_error(f"The norm in the discrepancy principle needs to be a callable! Got {norm}."))
            self.norm = norm
        else:
            if setting is None:
                raise ValueError(Errors.value_error('If setting is not provided, then norm must be provided.')) 
            self.norm = setting.h_codomain.norm
        super().__init__()
        if data is not None:
            self.data = data
        else:
            if setting is None or setting.data is None:
                raise ValueError(Errors.value_error('If setting is not provided or setting has no data, then data must be provided.'))
            else:
                self.data = setting.data
            
        if noise_level_is_relative:
            self.noiselevel = noiselevel*self.norm(self.data)
        else:
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
        self.log_info = 'discr./noiselevel = {:3.2f}< {:1.2f}'.format(rel, self.tau)
        if self.is_main_rule:
            self.log.info(self.log_info)
        return rel < self.tau

class LCurve(StopRule):
    r"""L Curve method.

    Computes ||x|| and ||y-data|| for all available parameters
    and returns as best iterate that x where the curve (||x||,||y-data||) 
    has maximal curvature

    Parameters
    ----------
    setting: Setting
    solver: The solver used for computing the reconstructions x
    max_iter: int
        Maximal number of regularization parameters considered
    """
    def __init__(self, 
                 setting,
                 solver,
                 max_iter:int=1000
                ):
        from regpy.solvers import Setting
        super().__init__()
        self.data = setting.data
        self.setting = setting
        self.norm = setting.h_codomain.norm
        self.solver = solver
        self.history_dict["residual"] = []
        self.history_dict["norm"] = []
        self.history_dict["alphas"] = []
        self.recos=[]
        self.max_iter = max_iter
        self.it =0

    def __repr__(self):
        return 'L Curve'

    def _stop(self):
        self.it +=1
        if self.solver.y is None:
            raise MissingValueError
        if self.solver.x is None:
            raise MissingValueError     
        residual = self.data - self.solver.y
        norm_res = self.norm(residual)
        norm_x = self.norm(self.solver.x)
        self.history_dict["residual"].append(norm_res)
        self.history_dict["norm"].append(norm_x)     
        self.history_dict["alphas"].append(self.setting.regpar)
        self.recos.append(self.solver.x.copy())
        self.log_info = 'res {:.3e},norm {:.3e}'.format(norm_res,norm_x)
        return self.it >= self.max_iter
   
    def best_stopping_index(self):
        res = self.history_dict["residual"]
        norm_x = self.history_dict["norm"]
        alphas = self.history_dict["alphas"]
        xi = np.log(res)
        eta = np.log(norm_x)
        dxi = np.gradient(xi,alphas)
        d2xi = np.gradient(dxi,alphas)
        deta = np.gradient(eta,alphas)
        d2eta = np.gradient(deta,alphas)
        kappa = (d2xi*deta - dxi*d2eta)/(dxi**2 + deta**2)**(3/2)
        return np.argmax(kappa)
        
    def best_iterate(self):
        return self.recos[self.best_stopping_index()]

class QuasiOpt(StopRule):
    r"""Quasi-optimality principle.

    Computes x for all available parameters
    and returns as best iterate that x_{k+1} where the ||x_{k+1} - x_{k}|| 
    is minimal

    Parameters
    ----------
    setting: Setting
    solver: The solver used for computing the reconstructions x
    max_iter: int
        Maximal number of regularization parameters considered
    """
    def __init__(self, 
                 setting,
                 solver,
                 max_iter:int=1000
                ):
        from regpy.solvers import Setting
        super().__init__()
        self.data = setting.data
        self.norm = setting.h_codomain.norm
        self.history_dict["norm_diff"] = []
        self.history_dict["alphas"] = []
        self.recos=[]
        self.max_iter = max_iter
        self.it =0

    def __repr__(self):
        return 'Quasi optimality'

    def _stop(self):
        self.it +=1
        if self.solver.y is None:
            raise MissingValueError
        if self.solver.x is None:
            raise MissingValueError
        if self.it > 1:
            norm_diff = self.norm(self.solver.x-self.recos[-1])
        else:
            norm_diff = 0
        self.history_dict["norm_diff"].append(norm_diff)
        self.history_dict["alphas"].append(self.solver.alpha)
        self.recos.append(self.solver.x.copy())
        self.log_info = 'norm_diff {:.3e}'.format(norm_diff)
        return self.it >= self.max_iter
    
    def best_stopping_index(self):
        norm_diff = self.history_dict["norm_diff"]
        return np.argmin(norm_diff[1:])+1

    def best_iterate(self):
        return self.recos[self.best_stopping_index()]
    
class Lepskii(StopRule):
        """Lepskii principle.

        Computes x for all available parameters
        and returns as best iterate x_{\bar k} where
        \bar k =  max{ k=1,...,max_it | ||x_l - x_k|| <=  4 solver.error_prop(l)*noise_level for all l<= k} 
        The function error_prop needs to be decreasing, which is typically the case if the regularization parameters are increasing
        When selecting \bar k, the regularization parameters are hence sorted increasingly
        

        Parameters
        ----------
        setting: Setting
        noise_level: Noise level (absolute)
        solver: The solver used for computing the reconstructions x
        max_iter: int
            Maximal number of regularization parameters considered
        """
        def __init__(self, 
                     setting,
                     solver,
                     noise_level,
                     max_iter:int=1000
                    ):
            from regpy.solvers import Setting
            super().__init__()
            self.noise_level = noise_level
            self.data = setting.data
            self.norm = setting.h_codomain.norm
            self.history_dict["error_prop"] = []
            self.history_dict["alphas"] = []
            self.recos=[]
            self.max_iter = max_iter
            self.it =0

        def __repr__(self):
            return 'Lepskii'

        def _stop(self):
            self.it +=1
            if self.solver.y is None:
                raise MissingValueError
            if self.solver.x is None:
                raise MissingValueError
            self.history_dict["alphas"].append(self.solver.alpha)
            self.history_dict["error_prop"].append(self.solver.error_prop)
            self.recos.append(self.solver.x.copy())
            self.log_info = 'alpha {:.3e}'.format(self.solver.alpha)
            return self.it >= self.max_iter
        
        def best_stopping_index(self):
            # Check if regularization parameters are increasing
            alphas = self.history_dict["alphas"]
            idx = np.argsort(alphas)
            self.history_dict["error_prop"] = [self.history_dict["error_prop"][i] for i in idx]
            recos = [self.recos[i] for i in idx]
            # Lepskii
            bark = 1
            while bark <= self.max_iter-1:
                l = 0
                while l<bark:
                    if self.norm(recos[bark] - recos[l])>= 4*self.history_dict["error_prop"][l]*self.noise_level:
                        break;
                    l +=1;
                if l<= bark-1:
                    break;
                bark+=1;
            return bark-1;
        
        def best_iterate(self):
            return self.recos[self.best_stopping_index()]


class Oracle(StopRule):
    r"""Oracle stopping rule. Returns the iterate that is closest to the exact solution in terms of the given distance function.
    Useful for testing purposes and monitoring when the exact solution is known.


    Parameters
    ----------
    setting: Setting| None, optional
        setting, default: None. In the default case, data and norm must be given.  
    distance_function : callable, optional
        The distance function to measure the distance between the current iterate and the exact solution. 
        distance_function(x, exact_solution) -> float
        Defaults to None, in which case the norm of the operator's domain is applied to x-exact_solution.
    """
    def __init__(self, 
                 setting,
                 exact_solution=None,
                 distance_function:Callable = None
                ):
        from regpy.solvers import Setting
        super().__init__()
        self.data = setting.data
        self.history_dict["error"] = []
        self.history_dict["alphas"] = []
        self.recos=[]
        if not hasattr(setting,'exact_solution') and exact_solution is None:  
            raise ValueError(Errors.value_error('Oracle stopping rule needs the exact solution to be provided in the setting!'))
        if exact_solution is not None:
            setting.exact_solution = exact_solution
        if distance_function is None:
            self.dist = lambda x:setting.op.domain.norm(x - setting.exact_solution)
        else:
            self.dist = lambda x:distance_function(x, setting.exact_solution)

    def __repr__(self):
        return 'Oracle'

    def _stop(self):
        if self.solver.x is None:
            raise MissingValueError     
        error = self.dist(self.solver.x)
        self.history_dict["error"].append(error)
        self.history_dict["alphas"].append(self.solver.setting.regpar)
        self.recos.append(self.solver.x.copy())
        self.log_info = 'error {:.3e}'.format(error)
        return False

    def best_stopping_index(self):
        return np.argmin(self.history_dict["error"])

    def best_iterate(self):
        return self.recos[self.best_stopping_index()]

########## General StopRules based on relative change of data or solution ##########

class RelativeChangeData(StopRule):
    r"""Stops if the relative change in the residual becomes small

    Stops at the first iterate at which the difference between the old residual
    and the new residual is smaller than a pre-determined tol::

        ||y_k-y_{k+1}|| < tol

    Parameters
    ----------
    norm : callable [default=None]
        The norm with respect to which the difference should be measured.
        In the default case this is the `norm` method of some :class:`~self.solver.op.codomain`.
    tol : float [default=0.]
        The tol value at which the iteration should be stopped
        norm : callable
    """

    def __init__(self, norm=None, tol=0.):
        if not callable(norm) and not norm is None:
            raise TypeError(Errors.type_error("The norm in the relative change of data stopping needs to be a callable or None!"))
        if not isinstance(tol,(int,float)):
            raise TypeError(Errors.type_error("The tol in the relative change of data stopping should be real scalar!"))
        if tol<0:
            raise ValueError(Errors.value_error("The tol in the relative change of data stopping needs to be bigger or equal to zero!"))
        super().__init__()
        self.norm = norm
        self.tol = tol
        self.data_old = None
        self.history_dict["relative change of y"] = []

    def _complete_init_with_solver(self, solver):
        if self.norm is None:
            self.norm = solver.op.codomain.norm
        super()._complete_init_with_solver(solver)   

    def reset(self):
        super().reset()
        self.data_old = None
        
    def __repr__(self):
        return 'RelativeChangeData(tol={})'.format(
            self.tol)

    def _stop(self):
        if self.solver.y is None:
            raise MissingValueError
        if self.data_old is None:   
            self.data_old = self.solver.y.copy()
            self.log_info = 'First iteration, no change computed.'
            if self.is_main_rule:
                self.log.info(self.log_info)
            return False
        change = self.norm(self.solver.y - self.data_old)
        self.data_old = self.solver.y.copy()
        self.history_dict["relative change of y"].append(change)
        self.log_info = 'rel. data change {:.3e}<{:.3e}'.format(change,self.tol)
        if self.is_main_rule:
            self.log.info(self.log_info)
        return change < self.tol


class RelativeChangeSol(StopRule):
    r"""Stops if the relative change in the solution space becomes small

    Stops at the first iterate at which the difference between the old estimate
    and the new estimate is smaller than a pre-determined tol::

        ||x_k-x_{k+1}|| < tol

    Parameters
    ----------
    norm : callable [default=None]
        The norm with respect to which the difference should be measured.
        In the default case this is the `norm` method of some :class:`~self.solver.op.domain`.
    tol : float [default=0.]
        The tol value at which the iteration should be stopped
    """

    def __init__(self, norm=None, tol=0.):
        if not callable(norm) and not norm is None:
            raise TypeError(Errors.type_error("The norm in the relative change of solution stopping needs to be a callable!"))
        if not isinstance(tol,(int,float)):
            raise TypeError(Errors.type_error("The tol in the relative change of solution stopping should be real scalar!"))
        if tol<0:
            raise ValueError(Errors.value_error("The tol in the relative change of solution stopping needs to be larger or equal to zero!"))
        super().__init__()
        self.norm = norm
        self.tol = tol
        self.sol_old = None
        self.history_dict["relative change of x"] = []

    def _complete_init_with_solver(self, solver):
        if self.norm is None:
            self.norm = solver.op.domain.norm
        super()._complete_init_with_solver(solver)   

    def __repr__(self):
        return 'RelativeChangeSol(tol={})'.format(
            self.tol)
    
    def reset(self):
        super().reset()
        self.sol_old = None

    def _stop(self,):
        if self.sol_old is None:   
            self.sol_old = self.solver.x.copy()
            self.log_info = 'First iteration, no change computed.'
            if self.is_main_rule:
                self.log.info(self.log_info)
            return False
        change = self.norm(self.solver.x - self.sol_old)
        self.sol_old = self.solver.x.copy()
        self.history_dict["relative change of x"].append(change)
        self.log_info = 'rel. change sol: {:.3e}<{:.3e}'.format(change,self.tol)
        if self.is_main_rule:
            self.log.info(self.log_info)
        return change < self.tol

######### StopRules for convex optimization problems #########

class OptimalityCondStopping(StopRule):
    def __init__(self, logging_level = "INFO",tol = 0.):
        r"""Stopping rule based on optimality condition violation.
        
        Parameters
        ----------
        tol : float [default=0.]
            The tolerance for the duality gap.
        logging_level : str
            The logging level for the stopping rule.
        """
        super().__init__()
        self.tol = tol
        self.log.setLevel(logging_level)
        self.history_dict["dSstar"] = []
        self.history_dict["dR"] = []

    def __repr__(self):
        return 'OptimalityCondStopping(tol={})'.format(
            self.tol)

    def _complete_init_with_solver(self, solver):
        if not solver.setting.is_tikhonov and  solver.setting.is_convex:
            raise RuntimeError(Errors.generic_message("It is not possible to compute the dual in the implementation of this setting. The setting needs to be convex and contain a regularization parameter!"))
        super()._complete_init_with_solver(solver)

    def _stop(self):
        primal = self.solver.primal() if hasattr(self.solver,"primal") and callable(self.solver.primal) else None
        dual = self.solver.dual() if hasattr(self.solver,"dual") and callable(self.solver.dual) else None
        if primal is None and dual is None:
            raise RuntimeError(Errors.generic_message("The solver needs to provide at least one of the methods 'primal' or 'dual'."))
        dSstar,dR = self.solver.setting.violation_optimality_cond(primal, dual)
        self.history_dict["dSstar"].append(dSstar)
        self.history_dict["dR"].append(dR)
        stop = (dSstar+dR<=self.tol)
        self.log_info = '{:.1e} + {:.1e} = {:.2e}  <= {:.1e}'.format(dSstar,dR,dSstar+dR,self.tol)
        if self.is_main_rule:
            self.log.info(self.log_info)    
        return stop 
    
class DualityGapStopping(StopRule):
    r"""Stopping rule based on duality gap.

    Parameters
    ----------
    tol : float [default=0.]
        The tolerance for the duality gap.
    logging_level : str
        The logging level for the stopping rule.
    """    
    def __init__(self, tol = 0, logging_level = "INFO"):
        super().__init__()
        self.tol = tol
        self.log.setLevel(logging_level)
        self.history_dict["duality gap"] = []

    def __repr__(self):
        return 'DualityGapStopping(tol={})'.format(
            self.tol)
    
    def _complete_init_with_solver(self, solver):
        if not solver.setting.is_tikhonov and  solver.setting.is_convex:
            raise RuntimeError(Errors.generic_message("It is not possible to compute the dual in the implementation of this setting. The setting needs to be convex and contain a regularization parameter!"))
        super()._complete_init_with_solver(solver)

    def _stop(self):
        primal = self.solver.primal() if hasattr(self.solver,"primal") and callable(self.solver.primal) else None
        dual = self.solver.dual() if hasattr(self.solver,"dual") and callable(self.solver.dual) else None
        if primal is None and dual is None:
            raise RuntimeError(Errors.generic_message("The solver needs to provide at least one of the methods 'primal' or 'dual'."))        
        gap = self.solver.setting.duality_gap(primal = primal, dual = dual)
        self.history_dict["duality gap"].append(gap)
        stop = (gap<=self.tol) or (gap == np.nan)
        if gap==np.nan:
            self.log_info = 'duality gap is NaN'
        else:
            self.log_info ='duality gap:{:.2e} <= {:.1e}'.format(gap,self.tol)
        if self.is_main_rule:
            self.log.info(self.log_info)    
        return stop 