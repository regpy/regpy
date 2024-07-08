from examples.medium_scattering.mediumscattering import MediumScatteringFixed
from regpy.operators import CoordinateProjection, CoordinateMask
from regpy.hilbert import L2, HmDomain, Sobolev, HilbertPullBack
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
import regpy.stoprules as rules
import regpy.util as util

import numpy as np
import logging

import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

radius = 1
scattering = MediumScatteringFixed(
    gridshape=(64, 64),
    radius=radius,
    wave_number=1,
    inc_directions=util.linspace_circle(16),
    farfield_directions=util.linspace_circle(16),
)

contrast = scattering.domain.zeros()
r = np.linalg.norm(scattering.domain.coords, axis=0)
contrast[r < radius] = np.exp(-1/(radius - r[r < radius]**2))

projection = CoordinateProjection(
    scattering.domain,
    scattering.support
)
embedding = projection.adjoint

op = scattering * embedding

exact_solution = projection(contrast)
exact_data = op(exact_solution)
noise = 0.01 * op.codomain.randn()
data = exact_data + noise
init = op.domain.zeros()

myh_domain = HmDomain(scattering.domain,scattering.support,dtype=complex,index=2)
setting = RegularizationSetting(
    op=op,
    # Define Sobolev norm on support via embedding
    #h_domain=HilbertPullBack(Sobolev(index=2), embedding, inverse='cholesky'),
    penalty = myh_domain, 
    data_fid =L2
)

solver = IrgnmCG(
    setting, data,
    regpar=0.0001, regpar_step=0.8,
    init=init,
    cg_pars=dict(
        tol=1e-8,
        reltolx=1e-8,
        reltoly=1e-8
    )
)
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(
        setting.h_codomain.norm, data,
        noiselevel=setting.h_codomain.norm(noise),
        tau=1.1
    )
)

plt.ion()
fig, axes = plt.subplots(ncols=3, nrows=2, constrained_layout=True)
bars = np.vectorize(lambda ax: cbar.make_axes(ax)[0], otypes=[object])(axes)

axes[0, 0].set_title('exact contrast')
axes[1, 0].set_title('exact data')
axes[0, 1].set_title('reco contrast')
axes[1, 1].set_title('reco data')
axes[0, 2].set_title('difference')

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
