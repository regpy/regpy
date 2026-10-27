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
    setting = Setting(op, L2, L2, exact_data = exact_data,  regpar=0.01)
    setting.add_Gaussian_noise(absolute_noise_level=0.03,white_noise=False)
    setting.run('TikhCG')
    assert setting.get_stoprule('TikhCG').main_rule.converged

    op = Volterra(grid,exponent=2)
    exact_data = op(exact_solution)
    init = op.domain.ones()*0.05

    setting = Setting(op, Sobolev, L2, exact_data = exact_data, init = init)
    setting.add_Gaussian_noise(absolute_noise_level=0.03,white_noise=False)    
    setting.run('nlLW')
    assert setting.get_stoprule('nlLW').main_rule.converged

    # Impulsive Noise
    sigma = 0.01*np.ones(grid.coords[0].shape[0])
    sigma[100:110] = 0.5
    noise = sigma * op.domain.randn()
    data = exact_data + noise
    init = op.domain.ones()

    #The penalty term |f|_{TV}
    setting = Setting(op, TV, L2,  data = data, regpar = 0.01, init = init)
    setting.run('gFISTA',max_it = 100)
    assert setting.get_stoprule('gFISTA').main_rule.converged    


sys.path.pop(0)