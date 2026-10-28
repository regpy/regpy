import logging
import sys

import numpy as np

from regpy.solvers.nonlinear.newton import NewtonCG

import regpy.stoprules as rules
from regpy.hilbert import L2, Sobolev
from regpy.solvers import Setting
import regpy.util as util

util.set_rng_seed(15873098306879350073259142812684978477)

from . import import_example_package

import_example_package("./examples/potential/")

from potential import Potential

def test_potential():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
    )

    #Forward operator
    op = Potential(
        radius=1.3
    )

    setting = Setting(op=op, penalty=Sobolev, data_fid=L2)

    #Exact data and Poission data
    exact_solution = op.domain.radialfct2curve(lambda t: np.sqrt(6*np.cos(1.5*t)**2+1)/3)
    exact_data = op(exact_solution.radial())
    #Initial guess
    init = op.domain.circle(radius=0.2)
    setting = Setting(op, L2, L2, exact_data=exact_data,  init = init.coeff)
    setting.add_Gaussian_noise(relative_noise_level=0.03,white_noise=False)

    setting.run('NewtCG',max_it=50)


sys.path.pop(0)
