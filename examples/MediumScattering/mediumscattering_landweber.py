import setpath

import itreg

from itreg.operators import MediumScattering
# TODO from itreg.spaces import Sobolev
from itreg.spaces import L2
from itreg.solvers import Landweber, HilbertSpaceSetting
# TODO from itreg.util import test_adjoint
import itreg.stoprules as rules
import itreg.util as util

import numpy as np
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

op = MediumScattering(
    gridshape=(65, 65),
    radius=1,
    wave_number=1,
    inc_directions=util.linspace_circle(16),
    meas_directions=util.linspace_circle(16),
    support=lambda grid, radius: np.max(np.abs(grid.coords), axis=0) <= radius,
    amplitude=False)

exact_solution = op.domain.ones()
exact_data = op(exact_solution)
noise = 0.03 * op.codomain.rand(np.random.randn)
data = exact_data + noise
noiselevel = np.linalg.norm(noise)
init = 1.1 * op.domain.ones()

setting = HilbertSpaceSetting(
    op=op,
    domain=L2,
    codomain=L2)

landweber = Landweber(setting, data, init, stepsize=0.01)
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0.1, tau=1))

reco, reco_data = landweber.run(stoprule)
