import logging
import numpy as np

from regpy.solvers import Solver, RegularizationSetting

class FISTA(Solver):
    r"""
    The generalized FISTA algorithm for minimization of $\alpha * \mathcal{G}+\mathcal{H}$, where $\mathcal{G},\mathcal{H}: H -> \mathbb{R}$ 
    are the penalty term and the data fidelity term respectively.

    We assume:
        -> $\mathcal{G}, \mathcal{H}$ are convex
        -> grad $\mathcal{H}$ is L-Lipschitz continuous
        -> $\mathcal{G}$ is $\mu_\mathcal{G}$-convex with $\mu_\mathcal{G} \geq 0$ 
        -> $\mathcal{H}$ is $\mu_\mathcal{H}$-convex with $\mu_\mathcal{H} \geq 0$ 

    Parameters:
    -----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem. Includes the penalty and data fidelity functionals. 
    init : array-like
        The initial guess
    tau : float, optional 
        step size of minimization procedure. Needs to be in (0, 1/L) where grad H is assumed to be L-Lipschitz 
    regpar : float, optional
        The regularization parameter
    mu_data_fidelity : float, optional
        The convexity constant of the data fidelity term. Matches $\mu_\mathcal{H}$.
    mu_penalty : float, optional
        The convexity constant of the penalty term. Matches $\mu_\mathcal{G}$
    proximal_pars : dict, optional
        Parameter dictionary passed to the computation of the prox-operator for the penalty term. 

    Notes
    -----
    The data fidelity in the setting has to be defined on the domain of the operator i.e. of the type $S(T(\cdot))$. 
    """
    def __init__(self, setting, init, tau = 1, regpar = 1, mu_data_fidelity = 1, mu_penalty = 1, proximal_pars=None):
        super().__init__()
        self.setting = setting
        """Regularization setting. Includes Operator, penalty and data fidelity functional and corresponding Hilbert Spaces.
        """
        assert isinstance(setting,RegularizationSetting)
        self.x = init
        self.y = self.setting.op(self.x)

        self.tau = tau
        """Step size of minimization procedure. 
        """
        self.regpar = regpar
        """Regularization parameter.
        """
        self.mu_data_fidelity = mu_data_fidelity
        """The convexity constant of the data fidelity term.
        """
        self.mu_penalty = mu_penalty
        """The convexity constant of the penalty term.
        """
        self.proximal_pars = proximal_pars
        """Proximal parameters that are passed to prox-operator of penalty term. 
        """

        self.t = 0
        self.t_old = 0
        self.mu = self.mu_data_fidelity+self.mu_penalty

        self.x_old = self.x
        self.q = (self.tau * self.mu) / (1+self.tau*self.mu_penalty)

    def _next(self):
        if self.mu == 0:
            self.t = (1 + np.sqrt(1+4*self.t_old**2))/2
            beta = (self.t_old-1) / self.t
        else: 
            self.t = (1-self.q*self.t_old**2+np.sqrt((1-self.q*self.t_old**2)**2+4*self.t_old**2))/2
            beta = (self.t_old-1)/self.t * (1+self.tau*self.mu_penalty-self.t*self.tau*self.mu)/(1-self.tau*self.mu_data_fidelity)

        h = self.x+beta*(self.x-self.x_old)

        self.x_old = self.x
        self.t_old = self.t

        self.x = self.setting.penalty.proximal(h-self.tau*self.setting.h_domain.gram_inv(self.setting.data_fid.gradient(h)), self.tau * self.regpar, self.proximal_pars)
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        self.y = self.setting.op(self.x)
