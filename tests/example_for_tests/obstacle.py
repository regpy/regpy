import logging

import numpy as np

from regpy.solvers.nonlinear.irgnm import IrgnmCG
from regpy.solvers.nonlinear.newton import NewtonCG
import regpy.stoprules as rules
from regpy.hilbert import L2, Sobolev
from regpy.solvers import RegularizationSetting
from regpy.vecsps.curve import apple
from examples.obstacle.dirichlet_op import DirichletOp
from examples.obstacle.dirichlet_op import create_synthetic_data


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

    setting = RegularizationSetting(op=op, penalty=Sobolev, data_fid=L2)

    #Exact data
    farfield, exact_solution = create_synthetic_data(op, apple(64,der=3))

    # Gaussian data 
    noiselevel=0.01
    noise = op.codomain.randn()
    noise = noiselevel*setting.h_codomain.norm(farfield)/setting.h_codomain.norm(noise)*noise
    data = farfield+noise

    #Initial guess
    t = 2*np.pi*np.arange(0, op.N_FK)/op.N_FK
    init = 0.45*np.append(np.cos(t), np.sin(t)).reshape((2, op.N_FK))
    init = init.flatten()

    #Solver: NewtonCG or IrgnmCG
    solver = NewtonCG(
        setting, data, init = init,
            cgmaxit=50, rho=0.6
    )

    """
    solver = IrgnmCG(
        setting, data,
        regpar=1.,
        regpar_step=0.5,
        init=init,
        cg_pars=dict(
            tol=1e-4
        )
    )
    """
    stoprule = (
        rules.CountIterations(100) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=noiselevel,
            tau=2.1
        )
    )

    solver.run(stoprule)




