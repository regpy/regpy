import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import regpy.stoprules as rules
from regpy.hilbert import L2
from regpy.operators.obstacle2d import Potential
from regpy.discrs.obstacles import StarTrigDiscr
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

op = Potential(
    domain=StarTrigDiscr(200),
    radius=1.5,
    nmeas=64,
)

exact_solution = np.ones(200)
exact_data = op(exact_solution)
noise = 0 * op.codomain.randn()
data = exact_data + noise

init = 1.1 * op.domain.ones()

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)

landweber = Landweber(setting, data, init, stepsize=0.1)
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(
        setting.Hcodomain.norm, data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=1.1
    )
)

reco, reco_data = landweber.run(stoprule)

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 8))
axs[0].set_title('Domain')
axs[1].set_title('Heat source')
axs[1].plot(exact_data)
axs[1].plot(data)
axs[1].plot(reco_data)
ymin = 0.7 * min(reco_data.min(), data.min(), exact_data.min())
ymax = 1.3 * max(reco_data.max(), data.max(), exact_data.max())
axs[1].set_ylim((ymin, ymax))
pts = op.domain.eval_curve(reco, 64).z
pts_2 = op.domain.eval_curve(exact_solution, 64).z
poly = Polygon(np.column_stack([pts[0, :], pts[1, :]]), animated=True, fill=False)
poly_2 = Polygon(np.column_stack([pts_2[0, :], pts_2[1, :]]), animated=True, fill=False)
axs[0].add_patch(poly)
axs[0].add_patch(poly_2)
xmin = 1.5 * min(pts[0, :].min(), pts_2[0, :].min())
xmax = 1.5 * max(pts[0, :].max(), pts_2[0, :].max())
ymin = 1.5 * min(pts[1, :].min(), pts_2[1, :].min())
ymax = 1.5 * max(pts[1, :].max(), pts_2[1, :].max())
axs[0].set_xlim((xmin, xmax))
axs[0].set_ylim((ymin, ymax))
plt.show()
