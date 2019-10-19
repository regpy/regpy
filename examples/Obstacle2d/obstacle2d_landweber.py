import logging

import numpy as np

import regpy.stoprules as rules
from regpy.discrs import UniformGrid
from regpy.hilbert import L2
from regpy.operators.obstacle2d import PotentialOp, plots
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

op = PotentialOp(
    domain=UniformGrid(np.linspace(0, 2 * np.pi, 200)),
    radius=1.5,
    nmeas=64,
)

exact_solution = np.ones(200)
exact_data = op(exact_solution)
noise = 0 * op.codomain.randn()
data = exact_data + noise

init = 1.1 * op.domain.ones()

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)

landweber = Landweber(setting, data, init, stepsize=0.1)
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(
        setting.Hcodomain.norm, data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=1.1
    )
)

reco, reco_data = landweber.run(stoprule)

plots(op, reco, reco_data, data, exact_data, exact_solution)
