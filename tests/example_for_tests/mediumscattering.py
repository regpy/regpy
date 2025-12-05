import sys
import logging

import numpy as np


from regpy.operators import CoordinateProjection
from regpy.hilbert import L2, Hm
from regpy.solvers import Setting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
import regpy.stoprules as rules
import regpy.util as util

util.set_rng_seed(15873098306879350073259142812684978477)


from . import import_example_package

import_example_package("./examples/medium_scattering/")

from mediumscattering import MediumScatteringFixed


def test_mediumscattering():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
    )

    radius = 1
    scattering = MediumScatteringFixed(
        gridshape=(64, 64),
        radius=radius,
        wave_number=1,
        inc_directions=util.linspace_circle(16),
        farfield_directions=util.linspace_circle(16),
    )

    contrast = scattering.domain.zeros()
    r = scattering.domain.coord_distances()
    contrast[r < radius] = np.exp(-1/(radius - r[r < radius]**2))

    op = scattering

    exact_solution = contrast
    exact_data = op(exact_solution)
    noise = 0.001 * op.codomain.randn()
    data = exact_data + noise
    init = op.domain.zeros()

    myh_domain = Hm(mask = scattering.support,dtype=complex,index=2)
    setting = Setting(
        op=op,
        # Define Sobolev norm on support via embedding
        #h_domain=HilbertPullBack(Sobolev(index=2), embedding, inverse='cholesky'),
        penalty = myh_domain, 
        data_fid =L2
    )

    solver = IrgnmCG(
        setting, data,
        regpar=0.0001, regpar_step=0.8,
        init=init,
        cg_pars=dict(
            tol=1e-8,
            reltolx=1e-8,
            reltoly=1e-8
        )
    )
    stoprule = (
        rules.CountIterations(100) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=1.1
        )
    )

    _, _ =  solver.run(stoprule)

    assert stoprule.rules[1].triggered


sys.path.pop(0)