# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:08:40 2019

@author: Hendrik Müller
"""

import setpath  # NOQA

from itreg.BIP.BIP_gaussian_approximation import BayesianIP
from itreg.operators.Volterra.volterra import Volterra
from itreg.spaces import L2
from itreg.grids import Square_1D
from itreg.solvers import IRGNM_CG
import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')


spacing=2*np.pi/200
grid=Square_1D((200,), np.pi, spacing)
op = Volterra(L2(grid), spacing=spacing)

exact_solution = np.sin(grid.coords)
exact_data = op(exact_solution)
noise = 0.1 * np.random.normal(size=grid.shape)
data = exact_data + noise

noiselevel = op.range.norm(noise)

bip = BayesianIP(op, 'gaussian', 'gaussian', data, exact_data+0.1*np.ones(exact_data.shape), gamma_prior=np.eye(200), gamma_d=np.eye(200))
