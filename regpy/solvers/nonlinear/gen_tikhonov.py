from regpy.stoprules import StopRule,CountIterations
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
    inner_stoprule: regpy.stoprules.StopRule        
    alphas: iterable or tuple, optional
        Either an iterable giving the grid of alphas or a tuple (alpha0,q).
        In the second case the seuqence :math:`(alpha0*q^n)_{n=0,1,2,...}` is generated.
        Default is (1.0,0.5).
    """
    def __init__(self,setting:Setting, 
                 inner_solver_class:type[RegSolver],
                 inner_solver_params:dict={},
                 inner_stoprule_class:type[StopRule]=CountIterations,
                 inner_stoprule_params:dict={'max_iterations':100},
                 alphas: Iterable[float] | Tuple[float,float] = (1e0,0.5),
                 logging_level:str= "INFO",
                 ):
                 
        super().__init__(setting)
        self.log.setLevel(logging_level) 

        self.inner_solver_class = inner_solver_class
        self.inner_stoprule_class = inner_stoprule_class
        self.inner_solver_params = inner_solver_params 
        self.inner_stoprule_params = inner_stoprule_params
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
        self.inner_solver = self.inner_solver_class(self.setting,**self.inner_solver_params) 
        self.inner_stoprule = self.inner_stoprule_class(**self.inner_stoprule_params)
        self.x,self.y = self.inner_solver.run(self.inner_stoprule)
        self.log.info('alpha = {}'.format(self.setting.regpar))
        self._external_alpha_call = False
        self.setting.get_or_update_initial_guess(self.x,update=True)
        """use the computed solution as initial guess for next iteration."""

    def _next(self):
        self.inner_solver = self.inner_solver_class(self.setting,**self.inner_solver_params) 
        self.inner_stoprule = self.inner_stoprule_class(**self.inner_stoprule_params)

        if not self._external_alpha_call:
            exhausted = self.set_next_alpha()
            if exhausted:
               self.converge()
        self.x,self.y = self.inner_solver.run(self.inner_stoprule)
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
     