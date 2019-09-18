import setpath

import numpy as np
import matplotlib.pyplot as plt

from itreg.operators.mediumscattering import MediumScatteringOneToMany, Normalization
from itreg.solvers.iterative_born import IterativeBorn

from itreg.util import potentials
from itreg.stoprules import CountIterations

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)
log = logging.getLogger()

inc_directions, farfield_directions = MediumScatteringOneToMany.generate_directions(
    ninc=32,
    nfarfield=256
)

op = MediumScatteringOneToMany(
    gridshape=(100, 100),
    radius=1,
    wave_number=10,
    inc_directions=inc_directions,
    farfield_directions=farfield_directions,
    normalization=Normalization.Schroedinger
)

noiselevel = 0.00

exact_solution = potentials.bell(op.domain)
exact_data = op(exact_solution)
noise = noiselevel * np.max(np.abs(exact_data)) * op.codomain.randn()
data = exact_data + noise

# Initializing the inversion method
solver = IterativeBorn(
    op=op,
    data=data,
    cutoffs=np.linspace(0.3, 0.5, 5),
    p=dict(
        GRID_SHAPE=op.domain.shape,
        WAVE_NUMBER=op.wave_number,
        SUPPORT_RADIUS=1,
    )
)
stoprule = CountIterations(20)

for x, y in solver.until(stoprule):
    r = y - solver.rhs
    e = x - exact_solution
    r_norm = solver.NFFT.norm(r) / solver.NFFT.norm(solver.rhs)
    e_norm = np.linalg.norm(e) / np.linalg.norm(exact_solution)
    log.info('|r|={:.2g}, |e|={:.2g}'.format(r_norm, e_norm))

plt.figure()
plt.imshow(np.abs(exact_solution))
plt.title('potential')
plt.colorbar()

plt.figure()
plt.imshow(np.abs(x))
plt.title('solution')
plt.colorbar()

plt.figure()
plt.imshow(np.abs(e))
plt.title('error')
plt.colorbar()
plt.show()
