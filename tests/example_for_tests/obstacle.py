import logging
import sys

import numpy as np

from regpy.solvers.nonlinear.newton import NewtonCG
import regpy.stoprules as rules
from regpy.hilbert import L2, Sobolev
from regpy.solvers import Setting
from regpy.vecsps.curve import Apple
import regpy.util as util

util.set_rng_seed(15873098306879350073259142812684978477)

from . import import_example_package

import_example_package("./examples/obstacle/")

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

    setting = Setting(op=op, penalty=Sobolev(index=1.6), data_fid=L2)

    #Exact data
    farfield, _ = op.create_synthetic_data(Apple)

    # Gaussian data 
    noiselevel=0.01
    noise = op.codomain.randn()
    noise = noiselevel*setting.h_codomain.norm(farfield)/setting.h_codomain.norm(noise)*noise
    data = farfield+noise

    #Initial guess
    init = op.domain.circle(radius = 0.45)    


    solver = NewtonCG(
        setting, data, init = init.coeff,
            cgmaxit=50, rho=0.8
    )

    stoprule = (
        rules.CountIterations(10) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=noiselevel,
            tau=2.4
        )
    )

    solver.run(stoprule)


sys.path.pop(0)
