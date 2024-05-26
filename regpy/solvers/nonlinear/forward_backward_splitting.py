import logging
import numpy as np

from regpy.solvers import RegSolver, TikhonovRegularizationSetting


class ForwardBackwardSplitting(RegSolver):
    r"""
    Minimizes \(\mathcal{S}(Tf)+\alpha*\mathcal{R}(f)\) with forward backward splitting. 

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem. Includes both penalty \(\mathcal{R}\) and data fidelity \(\mathcal{S}\) functional. 
    init : setting.domain
        The initial guess. 
    tau : float , optional
        The step size parameter. Must be positive. 
        Default is the operator norm of \(T^*T\) 
    regpar : float, optional
        The regularization parameter \(\alpha\). Must be positive.
    proximal_pars: dict, optional
        Parameter dictionary passed to the computation of the prox-operator.
    """
    def __init__(self, setting, init, tau = None, proximal_pars = None):
        assert isinstance(setting,TikhonovRegularizationSetting), "Setting is not a TikhnoovRegularizationSetting instance."
        super().__init__(setting)
        assert init in self.op.domain
        self.regpar = setting.regpar
        """The regularization parameter."""

        self.x = init
        self.y, self.deriv = self.op.linearize(self.x)

        assert tau is None or tau>0
        if tau is None:
            self.tau = setting.op_norm(self.deriv)
        else:
            self.tau = tau
            """The step size parameter"""
        self.proximal_pars = proximal_pars

        
    def _next(self):
        self.x-=self.tau*self.h_domain.gram_inv(self.deriv.adjoint(self.data_fid.subgradient(self.y)))
        self.x = self.penalty.proximal(self.x, self.regpar*self.tau, self.proximal_pars)
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        self.y,self.deriv = self.op.linearize(self.x)
 