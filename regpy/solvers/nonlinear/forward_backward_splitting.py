import logging
import numpy as np

from regpy.solvers import Solver, RegularizationSetting

"""
Minimizes data_fidelity(f)+regpar*penalty(f) with forward backward splitting

Parameters
----------
setting : regpy.solvers.RegularizationSetting
    The setting of the forward problem. Includes both penalty and data fidelity functional. 
init : array-like
    The initial guess. 
tau : float , optional
    The parameter to compute the proximal operator of the penalty term. Must be positive.
regpar : float, optional
    The regularization parameter. Must be positive.
proximal_pars: dict, optional
    Parameter dictionary passed to the computation of the prox-operator.
"""

class ForwardBackwardSplitting(Solver):
    def __init__(self, setting, init, tau = 1, regpar = 1, proximal_pars = None):
        assert isinstance(setting,RegularizationSetting), "Setting is not a RegularizationSetting instance."
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.regpar = regpar
        """The regularization parameter."""
        self.tau = tau
        """The proximal operator parameter"""
        self.proximal_pars = proximal_pars

        
        self.x = init
        self.y = self.setting.op(self.x)
        
    def _next(self):
        self.x-=self.tau*self.setting.h_domain.gram_inv(self.setting.data_fid.gradient(self.x)) 
        self.x = self.setting.penalty.proximal(self.x, self.regpar*self.tau, self.proximal_pars)
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        
        self.y = self.setting.op(self.x)
        
