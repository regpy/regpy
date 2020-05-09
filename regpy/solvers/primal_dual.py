import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util

"""The Chambolle-Pock Algorithm"""
"""For theta==0 thisis the Arrow-Hurwicz-Uzawa algorithm"""
class PDHG(Solver):
    def __init__(self,  setting, data_fidelity_conjugate, penalty, init_domain, init_codomain, tau = 1, sigma = 1, regpar = 1, theta= 0, proximal_pars_data_fidelity_conjugate = None, proximal_pars_penalty = None):
        super().__init__()
        self.setting = setting
        assert self.setting.op.linear
        self.data_fidelity_conjugate = data_fidelity_conjugate
        self.penalty = penalty

        self.x = init_domain
        self.x_old = self.x
        self.y = self.setting.op(self.x)
        self.p = init_codomain

        self.tau = tau
        self.sigma = sigma
        self.regpar = regpar
        self.theta = theta
        self.proximal_pars_data_fidelity_conjugate = proximal_pars_data_fidelity_conjugate
        self.proximal_pars_penalty = proximal_pars_penalty

    def _next(self):
        primal_step = self.x - self.tau * self.setting.Hdomain.gram_inv(self.setting.op.adjoint(self.setting.Hcodomain.gram(self.p)))
        self.x = self.penalty.proximal(primal_step, self.regpar * self.tau, self.proximal_pars_penalty)
        dual_step = self.p + self.sigma * self.setting.op( self.x+self.theta*(self.x-self.x_old) )
        self.p = self.data_fidelity_conjugate.proximal(dual_step, self.sigma, self.proximal_pars_data_fidelity_conjugate)
        self.x_old = self.x
        self.y = self.setting.op(self.x)

"""The Douglas-Rashford Algorithm"""
class Douglas_Rashford(Solver):
    def __init__(self,  setting, data_fidelity, penalty, init_h, tau = 1, regpar = 1, proximal_pars_data_fidelity = None, proximal_pars_penalty = None):
        super().__init__()
        self.setting = setting
        self.data_fidelity = data_fidelity
        self.penalty = penalty

        self.h = init_h

        self.tau = tau
        self.regpar = regpar
        self.proximal_pars_data_fidelity = proximal_pars_data_fidelity
        self.proximal_pars_penalty = proximal_pars_penalty

        self.x = self.penalty.proximal(self.h, self.tau*self.regpar, self.proximal_pars_penalty)
        self.y = self.setting.op(self.x)

    def _next(self):
        self.h += self.data_fidelity.proximal(2*self.x-self.h, self.tau, self.proximal_pars_data_fidelity) - self.x
        self.x = self.penalty.proximal(self.h, self.tau*self.regpar, self.proximal_pars_penalty)
        self.y = self.setting.op(self.x)