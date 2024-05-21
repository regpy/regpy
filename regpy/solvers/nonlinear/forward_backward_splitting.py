import logging
import numpy as np

from regpy.solvers import RegSolver, RegularizationSetting

class ForwardBackwardSplitting(RegSolver):
    r"""
    Minimizes \(\mathcal{S}(f)+r\alpha*\mathcal{R}(f)\) with forward backward splitting. 

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem. Includes both penalty \(\mathcal{R}\) and data fidelity \(\mathcal{S}\) functional. 
    init : array-like
        The initial guess. Must be in setting.op.domain 
    tau : float , optional
        The parameter to compute the proximal operator of the penalty term. Must be positive.
    regpar : float, optional
        The regularization parameter \(\alpha\). Must be positive.
    proximal_pars: dict, optional
        Parameter dictionary passed to the computation of the prox-operator.
    """
    def __init__(self, setting, init, tau = 1, regpar = 1, proximal_pars = None):
        assert isinstance(setting,RegularizationSetting), "Setting is not a RegularizationSetting instance."
        super().__init__(setting)
        assert regpar > 0
        assert tau > 0
        assert init in setting.op.domain
        self.regpar = regpar
        """The regularization parameter."""
        self.tau = tau
        """The proximal operator parameter"""
        self.proximal_pars = proximal_pars

        
        self.x = init
        self.y = self.op(self.x)
        
    def _next(self):
        self.x-=self.tau*self.h_domain.gram_inv(self.data_fid.subgradient(self.x)) 
        self.x = self.penalty.proximal(self.x, self.regpar*self.tau, self.proximal_pars)
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        
        self.y = self.op(self.x)
        
