import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util
from regpy.functionals import Functional
from regpy.solvers import HilbertSpaceSetting

from regpy.solvers.linear.tikhonov import TikhonovCG

"""The ADMM algorithm"""

class ADMM(Solver):
    """The ADMM method for minimizing S(Tf) + regpar * R(f)

    Parameters
    ----------
    setting : regpy.solvers.HilbertSpaceSetting
        The setting of the forward problem.
    data_fidelity : regpy.functionals.Functional
        The data fidelity term. Needs to have a prox-operator defined. Matches S.
    penalty : regpy.functionals.Functional
        The penalty term. Needs to have a prox-operator defined. Matches R.
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
    def __init__(self,  setting, data_fidelity, penalty, init, gamma = 1, regpar = 1, proximal_pars_data_fidelity = None, proximal_pars_penalty = None, cg_pars = None):
        super().__init__()
        self.setting = setting
        assert self.setting.op.linear
        self.data_fidelity = data_fidelity
        self.penalty = penalty
        assert isinstance(self.data_fidelity, Functional)
        assert isinstance(self.penalty, Functional)
        assert self.data_fidelity.h_domain == self.setting.h_codomain
        assert self.penalty.h_domain == self.setting.h_domain

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
            setting=HilbertSpaceSetting(self.setting.op, self.setting.h_domain, self.setting.h_codomain),
            data=self.v1+self.p1,
            xref=self.v2+self.p2,
            regpar=1,
            **self.cg_pars
        ).run()

    def _next(self):
        self.v1 = self.data_fidelity.proximal(self.setting.op(self.x)-self.p1, 1/self.gamma, self.proximal_pars_data_fidelity)
        self.v2 = self.penalty.proximal(self.setting.op(self.x)-self.p2, self.regpar/self.gamma, self.proximal_pars_penalty)
        self.p1 -= self.gamma*(self.setting.op(self.x)-self.v1)
        self.p2 -= self.gamma*(self.setting.op(self.x)-self.v2)

        self.x, self.y = TikhonovCG(
            setting=HilbertSpaceSetting(self.setting.op, self.setting.h_domain, self.setting.h_codomain),
            data=self.v1+self.p1,
            xref=self.v2+self.p2,
            regpar=1,
            **self.cg_pars
        ).run()
