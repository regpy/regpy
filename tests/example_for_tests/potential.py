import logging

import numpy as np
from regpy.solvers.nonlinear.irgnm import IrgnmCG
from regpy.solvers.nonlinear.newton import NewtonCG

import regpy.stoprules as rules
from regpy.hilbert import L2, Sobolev
from regpy.operators.obstacles import Potential
from regpy.vecsps.obstacles import StarTrigDiscr
from regpy.solvers import HilbertSpaceSetting


def test_potential():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
    )

    op = Potential(
        domain=StarTrigDiscr(200),
        radius=1.2,
        nmeas=64,
    )

    setting = HilbertSpaceSetting(op=op, h_domain=Sobolev, h_codomain=L2)

    exact_solution = op.domain.sample(lambda t: np.sqrt(3 * np.cos(t)**2 + 1) / 2)
    exact_data = op(exact_solution)
    noise = op.codomain.randn()
    noise = 0.01*setting.h_codomain.norm(exact_data)/setting.h_codomain.norm(noise) * noise
    data = exact_data + noise

    init = op.domain.sample(lambda t: 1)

    solver = NewtonCG(
        setting, data, init = init,
            cgmaxit=50, rho=0.8
    )
    """solver = IrgnmCG(
        setting, data,
        regpar=10,
        regpar_step=0.8,
        init=init,
        cg_pars=dict(
            tol=1e-4
        )
    )"""
    stoprule = (
        rules.CountIterations(100) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=2.1
        )
    )


    for n, (reco, reco_data) in enumerate(solver.until(stoprule)):
        pass

