#!/usr/bin/python3

import setpath  # NOQA

from itreg.operators import Volterra
from itreg.spaces import L2
from itreg.solvers import IRGNM_L1_fid
import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

xs = np.linspace(0, 2 * np.pi, 200)
spacing = xs[1] - xs[0]

op = Volterra(L2(len(xs)), spacing=spacing)

exact_solution = np.sin(xs)
exact_data = op(exact_solution)
noise = 0.1 * np.random.normal(size=xs.shape)
data = exact_data + noise

noiselevel = op.domy.norm(noise)

irgnm_l1_fid = IRGNM_L1_fid(op, data, np.zeros(xs.shape), alpha0 = 1, alpha_step = 2/3., alpha_l1 = 1e-4)
stoprule = rules.CombineRules(
    [rules.CountIterations(100),
     rules.Discrepancy(op, data, noiselevel, tau=1.1)],
    op=op)

plt.plot(xs, exact_solution)
plt.plot(xs, exact_data)
plt.plot(xs, irgnm_l1_fid.run(stoprule))
plt.plot(xs, data)
plt.show()
#GGWP