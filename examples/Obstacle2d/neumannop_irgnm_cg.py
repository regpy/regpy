import regpy

from regpy.operators.obstacle2d import NeumannOp
from regpy.operators.obstacle2d.NeumannOp import create_synthetic_data

from regpy.hilbert import L2
from regpy.discrs import UniformGrid
from regpy.solvers import HilbertSpaceSetting

import regpy.stoprules as rules
from regpy.operators.obstacle2d import plots

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

xs = np.linspace(0, 2 * np.pi, 64)
spacing = xs[1] - xs[0]
ys=np.linspace(0, 63, 64)

grid=UniformGrid(xs)
grid_codomain=UniformGrid(ys, dtype=complex)


op=NeumannOp(grid, codomain=grid_codomain)

init = 1*op.domain.ones()

setting=HilbertSpaceSetting(op=op, domain=L2, codomain=L2)
#setting=HilbertSpaceSetting(op=op, domain=L2, codomain=L2)
#exact_data=setting.op.create_synthetic_data()
exact_data=create_synthetic_data(setting)
exact_solution=setting.op.obstacle.bd_ex.q[0, :]
data=exact_data


newton = IRGNM_CG(setting, data, init)
stoprule = (
    rules.CountIterations(1000) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0.1, tau=1.1))

reco, reco_data = newton.run(stoprule)

plotting=plots(op, reco, reco_data, data, exact_data, exact_solution)
plotting.plotting()
