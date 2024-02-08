from regpy.operators.volterra import Volterra
from regpy.hilbert import L2
from regpy.vecsps import UniformGridFcts
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.tikhonov import TikhonovCG
import regpy.stoprules as rules

import numpy as np
import logging

def test_volterra_tikhonov_cg():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s')

    grid = UniformGridFcts((0, 2*np.pi, 200))
    op = Volterra(grid)

    exact_solution = np.sin(grid.coords[0])
    exact_data = op(exact_solution)
    noise = 0.03 * op.domain.randn()
    data = exact_data + noise
    init = op.domain.ones()

    setting = HilbertSpaceSetting(op=op, h_domain=L2, h_codomain=L2)

    solver = TikhonovCG(setting, data, regpar=0.01)
    stoprule = (
        rules.CountIterations(1000) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=1.1
        )
    )

    reco, reco_data = solver.run(stoprule)
