import logging
import sys

import numpy as np

from regpy.hilbert import L2, Sobolev
from regpy.solvers import Setting
from regpy.vecsps.curve import Apple
import regpy.util as util

util.set_rng_seed(15873098306879350073259142812684978477)

from . import import_example_package

import_example_package("./examples/obstacles2d/")

from dirichlet_op import DirichletOp

def test_obstacle():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
    )

    #Forward operator
    op = DirichletOp(
        kappa = 3,
        inc_waves = 4
    )

    #Exact data
    farfield, _ = op.create_synthetic_data(Apple)
    #Initial guess
    init = op.domain.circle(radius = 0.45)   
    setting = Setting(op, Sobolev(index=1.6), L2, 
                      exact_data= farfield,
                      init = init.coeff
                      )

    noiselevel=0.01
    setting.add_Gaussian_noise(relative_noise_level=noiselevel,white_noise=False)

    setting.run('NewtCG',max_it = 20)


sys.path.pop(0)
