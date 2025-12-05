import sys

import numpy as np

from regpy.vecsps import UniformGridFcts
from regpy.hilbert import L2, Sobolev
from regpy.functionals import HilbertNorm, TV
from regpy.solvers import Setting
import regpy.stoprules as rules
from regpy.solvers.linear.tikhonov import TikhonovCG
from regpy.solvers.nonlinear.landweber import Landweber
from regpy.solvers.nonlinear.fista import FISTA
import regpy.util as util

util.set_rng_seed(15873098306879350073259142812684978477)

from . import import_example_package

import_example_package("./examples/volterra/")

from volterra import Volterra

def test_volterra():
    grid = UniformGridFcts(np.linspace(0, 2 * np.pi, 200))

    exact_solution = (1-np.cos(grid.coords[0]))**2/4 



    op = Volterra(grid)

    exact_data = op(exact_solution)
    noise = 0.03 * op.domain.randn()
    data = exact_data + noise

    setting = Setting(op, L2, L2)

    solver = TikhonovCG(setting, data, regpar=0.01)
    stoprule = (
        rules.CountIterations(1000) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=1.1
        )
    )

    _, _ = solver.run(stoprule)

    op = Volterra(grid,exponent=2)

    exact_data = op(exact_solution)
    noise = 0.03 * op.domain.randn()
    data = exact_data + noise
    init = op.domain.ones()*0.05

    setting = Setting(op, Sobolev, L2)

    solver = Landweber(setting, init,data, stepsize=0.01)
    stoprule = (
        # Landweber is slow, so need to use large number of iterations
        rules.CountIterations(max_iterations=100000) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=1.1
        )
    )

    _, _ = solver.run(stoprule)

    op = Volterra(grid, exponent=2)

    # Impulsive Noise
    sigma = 0.01*np.ones(grid.coords[0].shape[0])
    sigma[100:110] = 0.5

    exact_data = op(exact_solution)
    noise = sigma * op.domain.randn()
    data = exact_data + noise
    init = op.domain.ones()

    #The penalty term |f|_{TV}
    setting = Setting(
        op=op, 
        penalty=TV(grid), 
        data_fid=HilbertNorm(h_space=L2), 
        data = data,
        regpar = 0.01
    )

    proximal_pars = {
            'stepsize' : 0.001,
            'maxiter' : 100
            }
    # """Parameters for the inner computation of the proximal operator with the Chambolle algorithm"""

    solver = FISTA(setting, init, proximal_pars=proximal_pars)
    stoprule = (
        # Method is slow, so need to use large number of iterations
        rules.CountIterations(max_iterations=100000) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=1.1
        )
    )

    _, _ = solver.run(stoprule)


sys.path.pop(0)