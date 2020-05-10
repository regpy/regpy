import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util

from regpy.solvers.tikhonov import TikhonovCG

"""The ADMM algorithm"""

class ADMM(Solver):
    def __init__(self,  setting, data_fidelity, penalty, init_v1, init_v2, init_p1, init_p2, gamma = 1, regpar = 1, proximal_pars_data_fidelity = None, proximal_pars_penalty = None, cgpars = None):
        super().__init__()
        self.setting = setting
        assert self.setting.op.linear
        self.data_fidelity = data_fidelity
        self.penalty = penalty

        self.v1 = init_v1
        self.v2 = init_v2
        self.p1 = init_p1
        self.p2 = init_p2

        self.gamma = gamma
        self.regpar = regpar
        self.proximal_pars_data_fidelity = proximal_pars_data_fidelity
        self.proximal_pars_penalty = proximal_pars_penalty

        if cgpars is None:
            cgpars = {}
        self.cgpars = cgpars
        """The additional `regpy.solvers.tikhonov.TikhonovCG` parameters."""

        self.x, self.y = TikhonovCG(
            setting=HilbertSpaceSetting(self.setting.op, self.setting.Hdomain, self.setting.Hcodomain),
            data=self.v1+self.p1,
            xref=self.v2+self.p2
            regpar=1,
            **self.cgpars
        ).run()

    def _next(self):
        self.v1 = self.data_fidelity.proximal(self.setting.op(self.x)-self.p1, 1/self.gamma, self.proximal_pars_data_fidelity)
        self.v2 = self.penalty.proximal(self.setting.op(self.x)-self.p2, self.regpar/self.gamma, self.proximal_pars_penalty)
        self.p1 -= self.gamma*(self.setting.op(self.x)-self.v1)
        self.p2 -= self.gamma*(self.setting.op(self.x)-self.v2)

        self.x, self.y = TikhonovCG(
            setting=HilbertSpaceSetting(self.setting.op, self.setting.Hdomain, self.setting.Hcodomain),
            data=self.v1+self.p1,
            xref=self.v2+self.p2
            regpar=1,
            **self.cgpars
        ).run()
