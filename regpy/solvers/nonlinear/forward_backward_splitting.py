import logging
import numpy as np

from regpy.solvers import RegSolver, TikhonovRegularizationSetting

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

class ForwardBackwardSplitting(RegSolver):
    r"""
    Minimizes :math:`\mathcal{S}(Tf)+\alpha*\mathcal{R}(f)` with forward backward splitting. 

    Parameters
    ----------
    setting : regpy.solvers.TikhonovRegularizationSetting
        The setting of the forward problem. Includes both penalty :math:`\mathcal{R}` and data fidelity :math:`\mathcal{S}` functional. 
    init : setting.domain [default: domain.zeros()]
        The initial guess. 
    tau : float , optional
        The step size parameter. Must be positive. 
        Default is the reciprocal of the operator norm of :math:`T^*T` 
    regpar : float, optional
        The regularization parameter :math:`\alpha`. Must be positive.
    proximal_pars: dict, optional
        Parameter dictionary passed to the computation of the prox-operator.
    logging_level: int [default: logging.INFO]
        logging level
    """
    def __init__(self, setting, init=None, tau = None, proximal_pars = {}, logging_level = logging.INFO):
        assert isinstance(setting,TikhonovRegularizationSetting), "Setting is not a TikhonovRegularizationSetting instance."
        super().__init__(setting)

        self.x = self.op.domain.zeros() if init is None else init
        assert self.x in self.op.domain
        self.y, self.deriv = self.op.linearize(self.x)
        self.tau = 1/setting.op_norm(op=self.deriv)**2 if tau is None else tau
        """The step size parameter"""
        assert self.tau>0
        self.proximal_pars = proximal_pars
        self.log.setLevel(logging_level)
        
    def _next(self):
        self.x-=self.tau*self.h_domain.gram_inv(self.deriv.adjoint(self.data_fid.subgradient(self.y)))
        self.x = self.penalty.proximal(self.x, self.regpar*self.tau, **self.proximal_pars)
        r"""Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        self.y,self.deriv = self.op.linearize(self.x)
 