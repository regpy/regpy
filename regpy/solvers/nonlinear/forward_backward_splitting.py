import logging
import numpy as np

from regpy.solvers import RegSolver, TikhonovRegularizationSetting

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

class ForwardBackwardSplitting(RegSolver):
    r"""
    Minimizes \(\mathcal{S}(Tf)+\alpha*\mathcal{R}(f)\) with forward backward splitting. 

    Parameters
    ----------
    setting : regpy.solvers.TikhonovRegularizationSetting
        The setting of the forward problem. Includes both penalty \(\mathcal{R}\) and data fidelity \(\mathcal{S}\) functional. 
    init : setting.domain [default: domain.zeros()]
        The initial guess. 
    tau : float , optional
        The step size parameter. Must be positive. 
        Default is the reciprocal of the operator norm of \(T^*T\) 
    regpar : float, optional
        The regularization parameter \(\alpha\). Must be positive.
    proximal_pars: dict, optional
        Parameter dictionary passed to the computation of the prox-operator.
    logging_level: int [default: logging.INFO]
        logging level
    """
    def __init__(self, setting, init=None, tau = None, proximal_pars = {}, logging_level = logging.INFO):
        assert isinstance(setting,TikhonovRegularizationSetting), "Setting is not a TikhonovRegularizationSetting instance."
        super().__init__(setting)
        self.regpar = setting.regpar
        """The regularization parameter."""

        if init is None:
            self.x = self.op.domain.zeros()
        else:
            assert init in self.op.domain
            self.x = init
        self.y, self.deriv = self.op.linearize(self.x)

        assert tau is None or tau>0
        if tau is None:
            self.tau = 1/setting.op_norm(op=self.deriv)
        else:
            self.tau = tau
            """The step size parameter"""
        self.proximal_pars = proximal_pars
        self.log.setlevel(logging_level)

        try:
            gap=self.setting.dualityGap(primal = self.x)
            self.dualityGapWorks =True
            self.log.info('initial duality gap: {}'.format(gap))
        except NotImplementedError:
            self.dualityGapWorks = False
        
    def _next(self):
        self.x-=self.tau*self.h_domain.gram_inv(self.deriv.adjoint(self.data_fid.subgradient(self.y)))
        self.x = self.penalty.proximal(self.x, self.regpar*self.tau, **self.proximal_pars)
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        self.y,self.deriv = self.op.linearize(self.x)
 
        if self.dualityGapWorks:
            gap=self.setting.dualityGap(primal = self.x,dual=self.setting.primalToDual(self.y,argumentIsOperatorImage=True) )
            self.log.debug('it.{}: duality gap={:.3e}'.format(self.iteration_step_nr,gap))
            