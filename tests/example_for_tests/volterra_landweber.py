import logging

import numpy as np

import regpy.stoprules as rules
from examples.volterra.volterra import Volterra
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.landweber import Landweber
from regpy.hilbert import L2, Sobolev
from regpy.vecsps import UniformGridFcts


def test_volterra_landweber():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
    )

    grid = UniformGridFcts(np.linspace(0, 2 * np.pi, 200))
    op = Volterra(grid, exponent=3)

    exact_solution = np.sin(grid.coords[0])
    exact_data = op(exact_solution)
    noise = 0.03 * op.domain.randn()
    data = exact_data + noise
    init = op.domain.ones()

    setting = RegularizationSetting(op=op, penalty=Sobolev, data_fid=L2)

    landweber = Landweber(setting, data, init, stepsize=0.01)
    stoprule = (
        # Landweber is slow, so need to use large number of iterations
        rules.CountIterations(max_iterations=100000) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=1.1
        )
    )

    reco, reco_data = landweber.run(stoprule)

