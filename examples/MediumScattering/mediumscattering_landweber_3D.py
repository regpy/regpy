from itreg.operators.MediumScattering.mediumscattering import MediumScattering
from itreg.hilbert import Sobolev
from itreg.grids import Square_3D
from itreg.solvers.landweber import Landweber
import itreg.stoprules as rules

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

#domain needs to be specified
N=(4, 4, 4)
rho=1
sobo_index=0

grid=Square_3D(N, 0, 4*rho/N[0])
grid.support_sphere(rho)

#x_coo=(4*rho/N[0])*np.arange(-N[0]/2, (N[0]-1)/2, 1)
#y_coo=(4*rho/N[1])*np.arange(-N[1]/2, (N[1]-1)/2, 1)
#coords=np.array([x_coo, y_coo])

domain=Sobolev(grid, sobo_index)
op = MediumScattering(domain)

length_exact_solution=np.size(domain.parameters_domain.ind_support)
exact_solution=np.linspace(1, length_exact_solution, num=length_exact_solution)

exact_data=op(exact_solution)

data=exact_data
#print(exact_data)

#noise=0.03 *op.domain.rand(np.random.randn)
#data=exact_data+noise

#noiselevel = op.codomain.norm(noise)

#init=op.initguess_func

init=(1+0j)*np.ones((length_exact_solution, 1))
init_data=op(init)

_, deriv=op.linearize(init)
#test_adjoint(deriv, 0.1)

landweber= Landweber(op, data, init, stepsize=0.01)
stoprule=(
    rules.CountIterations(20)+
    rules.Discrepancy(op.codomain.norm, data, noiselevel=0, tau=2))

reco, reco_data=landweber.run(stoprule)
