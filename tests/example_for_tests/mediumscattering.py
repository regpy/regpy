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

    exact_solution = contrast
    exact_data = scattering(exact_solution)
   
    init = scattering.domain.zeros()

    myh_domain = Hm(mask = scattering.support,dtype=complex,index=2)
    setting = Setting(
        op=scattering,
        # Define Sobolev norm on support via embedding
        #h_domain=HilbertPullBack(Sobolev(index=2), embedding, inverse='cholesky'),
        penalty = myh_domain, 
        data_fid =L2, 
        exact_data=exact_data
    )
    setting.add_Gaussian_noise(relative_noise_level=0.1,white_noise=False)

    solver = IrgnmCG(setting, 
        regpar=0.0001, regpar_step=0.8,
        init=init,
        cg_pars=dict(
            tol=1e-8,
            reltolx=1e-8
        )
    )
    stoprule = rules.Discrepancy()

    _, _ =  solver.run(stoprule, max_it = 100)

    assert stoprule.main_rule.converged

sys.path.pop(0)