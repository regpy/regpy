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
    noise = op.codomain.randn()
    noise = 0.01*setting.h_codomain.norm(exact_data)/setting.h_codomain.norm(noise)*noise
    data = exact_data + noise

    #Initial guess
    init = op.domain.circle(radius=0.2)

    #Solver: NewtonCG or IrgnmCG
    solver = NewtonCG(
        setting, data, init = init.coeff,
            cgmaxit=50, rho=0.6
    )

    """
    solver = IrgnmCG(
        setting, data,
        regpar = 1,
        regpar_step = 0.5,
        init = init,
        cg_pars = dict(
            tol = 1e-4
        )
    )
    """
    stoprule = (
        rules.CountIterations(100) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel = setting.h_codomain.norm(noise),
            tau=2.1
        )
    )

    #Plot function


    solver.run(stoprule)


sys.path.pop(0)
