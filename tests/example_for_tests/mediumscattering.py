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

    myh_domain = Hm(mask = scattering.support,dtype=complex,index=2)
    setting = Setting(
        op=scattering,
        # Define Sobolev norm on support via embedding
        #h_domain=HilbertPullBack(Sobolev(index=2), embedding, inverse='cholesky'),
        penalty = myh_domain, 
        data_fid =L2, 
        exact_data=exact_data, 
        init = scattering.domain.zeros()
    )
    setting.add_Gaussian_noise(relative_noise_level=0.1,
                               white_noise=False,
                               seed = 42)

    setting.set_method_params('IRGN_CG',{'regpar':1e-4, 'regpar_step':2/3})
    _, _ =  setting.run('IRGN_CG', max_it = 30)

    assert setting.get_stoprule('IRGN_CG').main_rule.converged

sys.path.pop(0)