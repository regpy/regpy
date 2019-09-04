import setpath

from itreg.operators import MediumScatteringFixed, CoordinateProjection
from itreg.spaces import L2, Sobolev, HilbertPullBack
from itreg.solvers import Landweber, HilbertSpaceSetting
import itreg.stoprules as rules
import itreg.util as util

from functools import partial
import numpy as np
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s')

scattering = MediumScatteringFixed(
    gridshape=(65, 65),
    radius=1,
    wave_number=1,
    inc_directions=util.linspace_circle(16),
    farfield_directions=util.linspace_circle(16),
    # support=lambda grid, radius: np.max(np.abs(grid.coords), axis=0) <= radius,
    # coarseshape=(17, 17),
)

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
    # TODO
    # Hdomain=HilbertPullBack(partial(Sobolev, index=1), embedding, inverse='cholesky'),
    Hdomain=L2,
    Hcodomain=L2
)

landweber = Landweber(setting, data, init, stepsize=0.001)
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(setting.Hcodomain.norm, data,
                      noiselevel=setting.Hcodomain.norm(noise),
                      tau=1)
)

reco, reco_data = landweber.run(stoprule)
solution = embedding(reco)

import matplotlib.pyplot as plt
plt.imshow(np.abs(solution))
plt.show()
