"""Stopping rules used inside iterative optimization methods.
"""

import logging
from typing import Iterable

import numpy as np

from regpy.util import Errors
from .general import MissingValueError, StopRule, _copy_vector

__all__ = [
    "RelativeChangeData", "RelativeChangeSol", "LinearSequence",
    "OptimalityCondStopping", "DualityGapStopping",
]

class RelativeChangeData(StopRule):
    r"""Stops if the relative change in the residual becomes small

    Stops at the first iterate at which the difference between the old residual
    and the new residual is smaller than a pre-determined tol::

        ||y_k-y_{k+1}|| < tol max(||y_k||,||y_{k+1}||)

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
            self.data_old = _copy_vector(self.solver.op.codomain, self.solver.y)
            self.log_info = 'First iteration, no change computed.'
            if self.is_main_rule:
                self.log.debug(self.log_info)
            return False
        change = self.norm(self.solver.y - self.data_old)
        ref = max([self.norm(self.solver.y),self.norm(self.data_old)])
        self.data_old = _copy_vector(self.solver.op.codomain, self.solver.y)
        self.history_dict["relative change of y"].append(change/ref)
        self.log_info = 'rel. data change {:.3e}<{:.3e}'.format(change,self.tol*ref)
        triggered =  change < self.tol*ref
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)
        return triggered


class RelativeChangeSol(StopRule):
    r"""Stops if the relative change in the solution space becomes small

    Stops at the first iterate at which the difference between the old estimate
    and the new estimate is smaller than a pre-determined tol::

        ||x_k-x_{k+1}|| < tol max(||x_k||,||x_{k+1}||)

 a   Parameters
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
            self.sol_old = _copy_vector(self.solver.op.domain, self.solver.x)
            self.log_info = 'First iteration, no change computed.'
            if self.is_main_rule:
                self.log.debug(self.log_info)
            return False
        change = self.norm(self.solver.x - self.sol_old)
        ref = max([self.norm(self.solver.x),self.norm(self.sol_old)])
        self.sol_old = _copy_vector(self.solver.op.domain, self.solver.x)
        self.history_dict["relative change of x"].append(change/ref)
        self.log_info = 'rel. change sol: {:.3e}<{:.3e}'.format(change,self.tol*ref)
        triggered = change < self.tol*ref
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)
        return triggered

######### stopping rules for convex optimization problems #########

class LinearSequence:
    r"""Iterator generating a linear sequence (c,c+a,c+2a,c+3a, ....)
    
    Parameters
    ----------
    a: int, optional
        difference between neighboring sequence elements. Defaults to 1 
    start: int, optional
        first element of the sequence. Defaults to 0
    """    
    def __init__(self, a:int=1,start:int=0):
        self.a=a, self.x= start

    def __iter__(self):
        return self

    def __next__(self)->int:
        result = self.x
        self.x += self.a
        return result

class OptimalityCondStopping(StopRule):
    r"""Stopping rule based on optimality condition violation.
        
        Parameters
        ----------
        tol : float [default=0.]
            The tolerance for the duality gap.
        logging_level : str
            The logging level for the stoprule.
    """
    def __init__(self, 
                 tol = 0., 
                 steps: int | Iterable[int]=1,
                 logging_level = "INFO"):   
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
        triggered = (dSstar+dR<=self.tol)
        self.log_info = '{:.1e} + {:.1e} = {:.2e}  <= {:.1e}'.format(dSstar,dR,dSstar+dR,self.tol)
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)    
        return triggered 
    
class DualityGapStopping(StopRule):
    r"""Stopping rule based on duality gap.

    Parameters
    ----------
    tol : float [default=0.]
        The tolerance for the duality gap.
    logging_level : str
        The logging level for the stoprule.
    """    
    def __init__(self, tol = 0, stop_at_infty = True, logging_level = "INFO"):
        super().__init__()
        self.tol = tol
        self.log.setLevel(logging_level)
        self.history_dict["duality gap"] = []
        self.stop_at_infty = stop_at_infty

    def __repr__(self):
        return 'DualityGapStopping(tol={})'.format(
            self.tol)
    
    def _complete_init_with_solver(self, solver):
        if not solver.setting.is_tikhonov and  solver.setting.is_convex:
            raise RuntimeError(Errors.generic_message("It is not possible to compute the dual in the implementation of this setting. The setting needs to be convex and contain a regularization parameter!"))
        solver.setting.duality_gap_warning_incr = 1.
        super()._complete_init_with_solver(solver)

    def _stop(self):
        primal = self.solver.primal() if hasattr(self.solver,"primal") and callable(self.solver.primal) else None
        dual = self.solver.dual() if hasattr(self.solver,"dual") and callable(self.solver.dual) else None
        if primal is None and dual is None:
            raise RuntimeError(Errors.generic_message("The solver needs to provide at least one of the methods 'primal' or 'dual'."))        
        gap = self.solver.setting.duality_gap(primal = primal, dual = dual)
        self.history_dict["duality gap"].append(gap)
        triggered = (gap<=self.tol) or (gap == np.nan) or (gap == np.inf and self.stop_at_infty)
        if gap==np.nan:
            self.log_info = 'duality gap is NaN'
        else:
            self.log_info ='duality gap:{:.2e} <= {:.1e}'.format(gap,self.tol)
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)    
        return triggered
