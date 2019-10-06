import setpath

import itreg

from itreg.operators.obstacle2d import PotentialOp

from itreg.hilbert import L2
from itreg.discrs import UniformGrid
from itreg.solvers import HilbertSpaceSetting
from itreg.solvers.landweber import Landweber

import itreg.stoprules as rules
from itreg.operators.obstacle2d import plots

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

xs = np.linspace(0, 2 * np.pi, 200)
spacing = xs[1] - xs[0]
ys=np.linspace(0, 63, 64)

grid=UniformGrid(xs)
grid_codomain=UniformGrid(ys)

op=PotentialOp(grid, codomain=grid_codomain)
#op = PotentialOp(L2(grid))

exact_solution = np.ones(200)
exact_data = op(exact_solution)
#noise = 0.03 * op.domain.rand(np.random.randn)
data = exact_data

#noiselevel = op.range.norm(noise)

init = 1.1*op.domain.ones()

#_, deriv = op.linearize(init)
#test_adjoint(deriv)

setting=HilbertSpaceSetting(op=op, domain=L2, codomain=L2)

landweber = Landweber(setting, data, init, stepsize=0.1)
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0.1, tau=1.1))

reco, reco_data = landweber.run(stoprule)

plotting=plots(op, reco, reco_data, data, exact_data, exact_solution)
plotting.plotting()
