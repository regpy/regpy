import logging
import sys

import numpy as np

from regpy.solvers.nonlinear.newton import NewtonCG
import regpy.stoprules as rules
from regpy.hilbert import L2, Sobolev
from regpy.solvers import Setting
from regpy.vecsps.curve import apple
import regpy.util as util

util.set_rng_seed(15873098306879350073259142812684978477)

from . import import_example_package

import_example_package("./examples/obstacle/")

from dirichlet_op import DirichletOp, create_synthetic_data


def test_obstacle():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
    )

    #Forward operator
    op = DirichletOp(
        kappa = 3,
        N_inc = 4
    )

    setting = Setting(op=op, penalty=Sobolev, data_fid=L2)

    #Exact data
    farfield, _ = create_synthetic_data(op, apple(64,der=3))

    # Gaussian data 
    noiselevel=0.01
    noise = op.codomain.randn()
    noise = noiselevel*setting.h_codomain.norm(farfield)/setting.h_codomain.norm(noise)*noise
    data = farfield+noise

    #Initial guess
    t = 2*np.pi*np.arange(0, op.N_FK)/op.N_FK
    init = 0.45*np.append(np.cos(t), np.sin(t)).reshape((2, op.N_FK))
    init = init.flatten()

    solver = NewtonCG(
        setting, data, init = init,
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
