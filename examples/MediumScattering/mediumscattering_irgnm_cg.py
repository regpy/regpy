from regpy.operators.mediumscattering import MediumScatteringFixed
from regpy.operators import CoordinateProjection
from regpy.hilbert import L2, Sobolev, HilbertPullBack
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.irgnm_cg import IrgnmCG
import regpy.stoprules as rules
import regpy.util as util

import numpy as np
import logging

import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar


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
init = op.domain.zeros()

setting = HilbertSpaceSetting(
    op=op,
    Hdomain=HilbertPullBack(Sobolev(index=2), embedding, inverse='cholesky'),
    Hcodomain=L2
)

solver = IrgnmCG(
    setting, data,
    regpar=10, regpar_step=0.8,
    init=init,
    cgpars=dict(
        tol=1e-8,
        reltolx=1e-8,
        reltoly=1e-8
    )
)
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(
        setting.Hcodomain.norm, data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=1.1
    )
)

plt.ion()
fig, axes = plt.subplots(ncols=3, nrows=2, constrained_layout=True)
bars = np.vectorize(lambda ax: cbar.make_axes(ax)[0], otypes=[object])(axes)

def show(i, j, x):
    im = axes[i, j].imshow(x)
    bars[i, j].clear()
    fig.colorbar(im, cax=bars[i, j])

show(0, 0, np.abs(contrast))
show(1, 0, np.abs(exact_data))
for reco, reco_data in solver.until(stoprule):
    solution = embedding(reco)
    show(0, 1, np.abs(solution))
    show(1, 1, np.abs(reco_data))
    show(0, 2, np.abs(solution - contrast))
    show(1, 2, np.abs(exact_data - reco_data))
    plt.pause(0.5)

plt.ioff()
plt.show()
