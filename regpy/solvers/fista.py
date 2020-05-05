import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util

class FISTA(Solver):
    def __init__(self, setting, data_fidelity, penalty, init, tau = 1, regpar = 1, mu_data_fidelity = 1, mu_penalty = 1):
        self.setting = setting
        self.data_fidelity = data_fidelity
        self.penalty = penalty

        self.x = init
        self.y = self.setting.op(self.x)

        self.tau = tau
        self.regpar = regpar
        self.mu_data_fidelity = mu_data_fidelity
        self.mu_penalty = mu_penalty

        self.t = 0
        self.t_old = 0
        self.mu = self.mu_data_fidelity+self.mu_penalty

        self.x_old = self.x
        self.q = (self.tau * self.mu) / (1+self.tau*self.mu_penalty)

    def _next(self):
        if mu = 0:
            self.t = (1 + np.sqrt(1+4*self.t_old*self.t_old))/2
            beta = (self.t_old-1) / self.t
        else: 
            self.t = (1-self.q*self.t_old+np.sqrt((1-self.q*self.t_old**2)**2)+4*self.t_old**2)/2
            beta = (self.t_old-1)/self.t * (1+self.tau*self.mu_penalty-self.t*self.tau*self.mu)/(1-self.tau*self.mu_data_fidelity)

        h = self.x+beta*(self.x-self.x_old)

        self.x_old = self.x
        self.t_old = self.t

        self.x = self.penalty.proximal(h-self.tau*self.setting.Hdomain.gram_inv(self.data_fidelity.gradient(h)))
        self.y = self.setting.op(self.x)
