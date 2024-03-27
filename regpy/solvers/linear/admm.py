import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util
from regpy.functionals import Functional
from regpy.solvers import RegularizationSetting

from regpy.solvers.linear.tikhonov import TikhonovCG

"""The ADMM algorithm"""

class ADMM(Solver):
    """The ADMM method for minimizing S(Tf) + regpar * R(f)

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem. Includes the penalty and data fidelity functionals.
    init : dict
        The initial guess. Must contain v1, v2, p1 and p2 keys. 
    gamma : float, optional
        Must be strictly greater than zero. 
    regpar : float
        The regularization parameter. Must be positive.
    proximal_pars_data_fidelity : dict, optional
        Parameter dictionary passed to the computation of the prox-operator for the data fidelity term
    proximal_pars_penalty : dict, optional
        Parameter dictionary passed to the computation of the prox-operator for the penalty term
    cg_pars : dict, optional
        Parameter dictionary passed to the inner `regpy.solvers.linear.tikhonov.TikhonovCG` solver.
    """
    def __init__(self,  setting, init, gamma = 1, regpar = 1, proximal_pars_data_fidelity = None, proximal_pars_penalty = None, cg_pars = None):
        assert isinstance(setting,RegularizationSetting)
        super().__init__()
        self.setting = setting
        assert self.setting.op.linear

        self.v1 = init['v1']
        self.v2 = init['v2']
        self.p1 = init['p1']
        self.p2 = init['p2']

        self.gamma = gamma
        self.regpar = regpar
        self.proximal_pars_data_fidelity = proximal_pars_data_fidelity
        self.proximal_pars_penalty = proximal_pars_penalty

        if cg_pars is None:
            cg_pars = {}
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""

        self.x, self.y = TikhonovCG(
            setting=RegularizationSetting(self.setting.op, self.setting.h_domain, self.setting.h_codomain),
            data=self.v1+self.p1,
            xref=self.v2+self.p2,
            regpar=1,
            **self.cg_pars
        ).run()

    def _next(self):
        self.v1 = self.setting.data_fid.proximal(self.setting.op(self.x)-self.p1, 1/self.gamma, self.proximal_pars_data_fidelity)
        self.v2 = self.setting.penalty.proximal(self.setting.op(self.x)-self.p2, self.regpar/self.gamma, self.proximal_pars_penalty)
        self.p1 -= self.gamma*(self.setting.op(self.x)-self.v1)
        self.p2 -= self.gamma*(self.setting.op(self.x)-self.v2)

        self.x, self.y = TikhonovCG(
            setting=RegularizationSetting(self.setting.op, self.setting.h_domain, self.setting.h_codomain),
            data=self.v1+self.p1,
            xref=self.v2+self.p2,
            regpar=1,
            **self.cg_pars
        ).run()
