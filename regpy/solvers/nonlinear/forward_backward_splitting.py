from regpy.util import Errors

from ..general import RegSolver

__all__ = ["ForwardBackwardSplitting"]

class ForwardBackwardSplitting(RegSolver):
    r"""
    Minimizes :math:`\mathcal{S}(Tf)+\alpha*\mathcal{R}(f)` with forward backward splitting. 

    Parameters
    ----------
    setting : regpy.solvers.Setting
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
    def __init__(self, setting, init=None, tau = None, proximal_pars = {}, logging_level = "INFO"):
        if not setting.is_tikhonov:
            raise ValueError(Errors.value_error("ForwardBackwardSplitting requires the setting to contain a regularization parameter!")) 
        super().__init__(setting)
        if self.op.linear:
            self.log.warning("Using non-linear ForwardBackwardSplitting with a linear Operator! Consider using the linear ForwardBackwardSplitting in the module solvers.linear")
        if init is not None and init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(init,self.op.domain,vec_name="initial guess",space_name="domain"))
        self.x = self.op.domain.zeros() if init is None else init
        self.y, self.deriv = self.op.linearize(self.x)
        self.tau = 1/self.deriv.norm(setting.h_domain,setting.h_codomain)**2 if tau is None else tau
        """The step size parameter"""
        if tau<=0:
            raise ValueError(Errors.value_error("tau the step size needs to be positive!")) 
        self.proximal_pars = proximal_pars
        self.log.setLevel(logging_level)
        
    def _next(self):
        self.x-=self.tau*self.h_domain.gram_inv(self.deriv.adjoint(self.data_fid.subgradient(self.y)))
        self.x = self.penalty.proximal(self.x, self.regpar*self.tau, **self.proximal_pars)
        r"""Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        self.y,self.deriv = self.op.linearize(self.x)
 