"""Stopping rules for choosing regularization parameters.
"""

import logging
from typing import Callable

import numpy as np

from regpy.util import Errors
from .general import MissingValueError, StopRule, _copy_vector

__all__ = ["Discrepancy", "LCurve", "QuasiOpt", "Lepskii", "Oracle"]

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
        The right-hand side (noisy data) or None (default). 
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
            raise ValueError(Errors.value_error(f"The noise level in the discrepancy principle needs to be greater than zero! Got {noiselevel}"))
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
        triggered =  rel < self.tau
        if self.is_main_rule:
            self.log.log(logging.INFO if triggered else logging.DEBUG,self.log_info)
        return triggered

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
        self.recos.append(_copy_vector(self.solver.op.domain, self.solver.x))
        self.log_info = 'res {:.3e},norm {:.3e}'.format(norm_res,norm_x)
        return self.it >= self.max_iter
   
    def best_stopping_index(self):
        res = self.history_dict["residual"]
        norm_x = self.history_dict["norm"]
        #alphas = self.history_dict["alphas"]
        #xi = np.log(res)
        #eta = np.log(norm_x)
        #dxi = np.gradient(xi,alphas)
        #d2xi = np.gradient(dxi,alphas)
        #deta = np.gradient(eta,alphas)
        #d2eta = np.gradient(deta,alphas)
        #kappa = (d2xi*deta - dxi*d2eta)/(dxi**2 + deta**2)**(3/2)
        #dydx = np.gradient(eta,xi)
        #d2yd2x = np.gradient(dydx,xi)
        #kappa = d2yd2x/(1+dydx**2)**(3/2)
        #return np.argmax(abs(kappa))
        return np.argmin(np.array(res)*np.array(norm_x))
        
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
        self.history_dict["alphas"].append(self.solver.setting.regpar)
        self.recos.append(_copy_vector(self.solver.op.domain, self.solver.x))
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
        solver: The solver used for computing the reconstructions x
        error_propagation: anonymous function used to estimate the error propagation in terms of alpha
        max_iter: int
            Maximal number of regularization parameters considered
        """
        def __init__(self, 
                     setting,
                     solver,
                     error_prop:Callable = None,
                     max_iter:int=100
                    ):
            from regpy.solvers import Setting
            super().__init__()
            self.data = setting.data
            self.norm = setting.h_codomain.norm
            #self.history_dict["error_prop"] = []
            self.history_dict["alphas"] = []
            self.recos=[]
            self.max_iter = max_iter
            self.it =0
            if error_prop is not None:
                self.Psi = error_prop
            #else:
                #TODO: Implement estimation of error propagation

        def __repr__(self):
            return 'Lepskii'

        def _stop(self):
            self.it +=1
            if self.solver.y is None:
                raise MissingValueError
            if self.solver.x is None:
                raise MissingValueError
            self.history_dict["alphas"].append(self.solver.setting.regpar)
            self.recos.append(_copy_vector(self.solver.op.domain, self.solver.x))
            self.log_info = 'iter {:d}, alpha {:.3e}'. format(self.it, self.solver.setting.regpar)
            return self.it > self.max_iter
        
        def best_stopping_index(self):
            # Check if regularization parameters are increasing
            alphas = self.history_dict["alphas"]
            if not np.all(alphas[:-1] <= alphas[1:]):
                idx = np.argsort(alphas)
            else:
                idx = np.arange(len(alphas))
            # Lepskii
            bark = 1
            while bark <= self.max_iter-1:
                l = 0
                while l<bark:
                    if self.norm(self.recos[idx[bark]] - self.recos[idx[l]])>= 4*self.Psi(alphas[idx[l]]):
                        break
                    l +=1
                if l<= bark-1:
                    break
                bark+=1
            return idx[bark-1]
        
        def best_iterate(self):
            return self.recos[self.best_stopping_index()]


class Oracle(StopRule):
    r"""Oracle stoprule. Returns the iterate that is closest to the exact solution in terms of the given distance function.
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
            raise ValueError(Errors.value_error('Oracle stoprule needs the exact solution to be provided in the setting!'))
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
        self.recos.append(_copy_vector(self.solver.op.domain, self.solver.x))
        self.log_info = 'error {:.3e}'.format(error)
        return False

    def best_stopping_index(self):
        return np.argmin(self.history_dict["error"])

    def best_iterate(self):
        return self.recos[self.best_stopping_index()]

########## General stopping rules based on relative change of data or solution ##########

