from regpy.stoprules import StopRule,CountIterations,DualityGapStopping
from ..general import RegSolver, Setting
from typing import Iterable, Tuple
from regpy.util import Errors

__all__ =["GeneralizedTikhonov","GeometricSequence"]
class GeometricSequence:
    r"""Iterator generating a geometric sequence
    
    Parameters
    ----------
    alpha0 : float
        :math:`\alpha_0` the initial regularization parameter 
    q : float
        Rate of the geometric sequence

    Notes
    ----- 
    Sequence defined recursively by
    
    .. math::
        \alpha_0 &= \alpha_0 \\
        \alpha_{n+1} &= q*\alpha_n
    """    
    def __init__(self, alpha0:float=1.0,q:float=0.5):
        self.alpha, self.alpha0, self.q = alpha0, alpha0, q

    def __iter__(self):
        return self

    def __next__(self)->float:
        result = self.alpha
        self.alpha *= self.q
        return result

class GeneralizedTikhonov(RegSolver):
    r"""Class runnning generalized Tikhonov regularization with some inner solver for different regularization parameters.
    This can serve as an interface to stopping rules for the selection of the regularization parameters 

    Parameters
    ----------
    setting:  regpy.solvers.Setting
        The setting of the forward problem.
    inner_solver: regpy.solvers.RegSolver class name
        The right hand side.
    inner_stoprule_class_list: list[regpy.stoprules.StopRule]        
        List of stop rules to be combined for the inner solver. Default is [regpy.stoprules.DualityGapStopping,regpy.stoprules.CountIterations].
    inner_stoprule_param_list: list[dict]
        List of parameter dicts for the inner stop rules. Default is [{'tol':1e-3},{'max_iterations':1000}].
    alphas: iterable or tuple, optional
        Either an iterable giving the grid of alphas or a tuple (alpha0,q).
        In the second case the seuqence :math:`(alpha0*q^n)_{n=0,1,2,...}` is generated.
        Default is (1.0,0.5).
    """
    def __init__(self,setting:Setting, 
                 inner_solver_class:type[RegSolver],
                 inner_solver_is_primal:bool=True,
                 inner_solver_params:dict={},
                 inner_stoprule_class_list:list[type[StopRule]]=[DualityGapStopping,CountIterations],
                 inner_stoprule_param_list:list[dict]=[{'tol':1e-3},{'max_iterations':1000}],
                 alphas: Iterable[float] | Tuple[float,float] = (1e0,0.5),
                 logging_level:str= "INFO",
                 inner_logging_level:str="WARNING"
                 ):
                 
        super().__init__(setting)
        self.log.setLevel(logging_level) 

        self.inner_solver_class = inner_solver_class
        self.inner_solver_is_primal = inner_solver_is_primal
        self.inner_stoprule_class_list = inner_stoprule_class_list
        self.inner_solver_params = inner_solver_params 
        self.inner_logging_level = inner_logging_level
        self.inner_stoprule_param_list = inner_stoprule_param_list
        if isinstance(alphas,tuple) and len(alphas)==2:
            self._alpha_iterator = GeometricSequence(alphas[0],alphas[1])
        else:
            try:
                self._alpha_iterator = iter(alphas)
            except TypeError:
                raise TypeError(Errors.value_error("alphas must be either an iterable or a tuple (alpha0,q).",alphas))
        
        exhausted = self.set_next_alpha()
        if exhausted:
            raise ValueError(Errors.value_error("No valid regularization parameter found in alphas.",alphas))
        the_setting = self.setting if self.inner_solver_is_primal else self.setting.get_dual_setting()
        self.inner_solver = self.inner_solver_class(the_setting,**self.inner_solver_params | {"logging_level":self.inner_logging_level}) 
        self.inner_stoprule = sum([rule(**params | {"logging_level":self.inner_logging_level}) for rule,params in zip(self.inner_stoprule_class_list,self.inner_stoprule_param_list )])
        self.x,self.y = self.inner_solver.run(self.inner_stoprule)
        if not self.inner_solver_is_primal:
            self.x = self.setting.dual_to_primal((self.x,self.y))
            self.y = self.setting.op(self.x)
        self.log.info('alpha = {}'.format(self.setting.regpar))
        self._external_alpha_call = False
        self.setting.get_or_update_initial_guess(self.x,update=True)
        """use the computed solution as initial guess for next iteration."""
    

    def _next(self):
        the_setting = self.setting if self.inner_solver_is_primal else self.setting.get_dual_setting()
        self.inner_solver = self.inner_solver_class(the_setting,**self.inner_solver_params | {"logging_level":self.inner_logging_level}) 
        self.inner_stoprule = sum([rule(**params | {"logging_level":self.inner_logging_level}) for rule,params in zip(self.inner_stoprule_class_list,self.inner_stoprule_param_list )])

        if not self._external_alpha_call:
            exhausted = self.set_next_alpha()
            if exhausted:
               self.converge()
        self.x,self.y = self.inner_solver.run(self.inner_stoprule)
        if not self.inner_solver_is_primal:
            self.x = self.setting.dual_to_primal((self.x,self.y))
            self.y = self.setting.op(self.x)
        self.log.info('alpha = {}'.format(self.setting.regpar))
        self._external_alpha_call = False
        self.setting.get_or_update_initial_guess(self.x,update=True)
        """use the computed solution as initial guess for next iteration."""        

    def set_next_alpha(self,alpha:float=None)->bool:
        r"""" Set the next regularization parameter.

        Parameters
        ----------    
        alpha: float, optional
            If given, this regularization parameter is set.
            Otherwise, the next parameter from the internal iterator is used.
        Returns
        -------   
        bool
            False, if a new regularization parameter was set, True if the iterator is exhausted.
        """ 
        if alpha is None:
            self._external_alpha_call = False
            try:     
                alpha = next(self._alpha_iterator)
            except StopIteration:
               return True
        else:
            self._external_alpha_call = True
        self.setting.regpar = alpha
        return False
     