from examples.medium_scattering.mediumscattering import MediumScatteringFixed
from regpy.operators import CoordinateProjection
from regpy.hilbert import L2, HmDomain, Sobolev
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
import regpy.stoprules as rules
import regpy.util as util

import numpy as np
import logging


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
    r = np.linalg.norm(scattering.domain.coords, axis=0)
    contrast[r < radius] = np.exp(-1/(radius - r[r < radius]**2))

    projection = CoordinateProjection(
        scattering.domain,
        scattering.support
    )
    embedding = projection.adjoint

    op = scattering * embedding

    exact_solution = projection(contrast)
    exact_data = op(exact_solution)
    noise = 0.001 * op.codomain.randn()
    data = exact_data + noise
    init = op.domain.zeros()

    myh_domain = HmDomain(scattering.domain,scattering.support,dtype=complex,index=2)
    setting = RegularizationSetting(
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



    for reco, reco_data in solver.until(stoprule):
        solution = embedding(reco)

    assert stoprule.rules[1].triggered



