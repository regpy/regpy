import setpath

import itreg

from itreg.operators import MediumScattering, CoordinateProjection
from itreg.spaces import L2, H1, HilbertPullBack
from itreg.solvers import Landweber, HilbertSpaceSetting
# TODO from itreg.util import test_adjoint
import itreg.stoprules as rules
import itreg.util as util

from functools import partial
import numpy as np
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s')

scattering = MediumScattering(
    gridshape=(65, 65),
    radius=1,
    wave_number=1,
    inc_directions=util.linspace_circle(16),
    meas_directions=util.linspace_circle(16),
    # support=lambda grid, radius: np.max(np.abs(grid.coords), axis=0) <= radius,
    # coarseshape=(17, 17),
    amplitude=False)

contrast = scattering.domain.zeros()
r = np.linalg.norm(scattering.domain.coords, axis=0)
contrast[r < 1] = np.exp(-1/(1 - r[r < 1]**2))

projection = CoordinateProjection(
    scattering.domain,
    scattering.support)
embedding = projection.adjoint

op = scattering * embedding

exact_solution = projection(contrast)
exact_data = op(exact_solution)
noise = 0.03 * op.codomain.randn()
data = exact_data + noise
init = 1.1 * op.domain.ones()

setting = HilbertSpaceSetting(
    op=op,
    domain=HilbertPullBack(partial(H1, index=1), embedding, inverse='cholesky'),
    codomain=L2)

landweber = Landweber(setting, data, init, stepsize=0.01)
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(setting.codomain.norm, data,
                      noiselevel=setting.codomain.norm(noise),
                      tau=1))

reco, reco_data = landweber.run(stoprule)
solution = embedding(reco)

import matplotlib.pylab as plt
plt.imshow(np.abs(solution))
plt.show()
