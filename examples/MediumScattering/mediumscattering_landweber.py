import setpath

import itreg

from itreg.operators.MediumScattering.mediumscattering import MediumScattering
from itreg.spaces import Sobolev
from itreg.spaces import L2
from itreg.grids import Square_2D
from itreg.solvers import Landweber
from itreg.util import test_adjoint
import itreg.stoprules as rules

import numpy as np 
import logging
import matplotlib.pyplot as plt 



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

#domain needs to be specified
N=(64, 64)
rho=1
sobo_index=0

grid=Square_2D(N, 0, 4*rho/N[0])
grid.support_circle(rho)

#x_coo=(4*rho/N[0])*np.arange(-N[0]/2, (N[0]-1)/2, 1)
#y_coo=(4*rho/N[1])*np.arange(-N[1]/2, (N[1]-1)/2, 1)
#coords=np.array([x_coo, y_coo])

domain=Sobolev(grid, sobo_index)
op = MediumScattering(domain)

length_exact_solution=np.size(domain.parameters_domain.ind_support)
#exact_solution=np.linspace(1, length_exact_solution, num=length_exact_solution)
exact_solution=np.ones(length_exact_solution)

exact_data=op(exact_solution) 

data=exact_data

#print(exact_data)

#noise=0.03 *op.domain.rand(np.random.randn)

#data=exact_data+noise

#noiselevel = op.range.norm(noise)
#print(noiselevel)
#init=op.initguess_func

init=1.1*np.ones((length_exact_solution, 1), dtype=complex)
#init=(1+0j)*np.ones(length_exact_solution)
init_data=op(init)

_, deriv=op.linearize(init)
#_, deriv=op.linearize(exact_solution)

testderivative=deriv(init)
#testderivative=deriv(np.reshape(np.linspace(1, 13, num=13), (1, 13)))
#testderivative=deriv(np.reshape(init, (1, 197)))
#testadjoint=deriv.adjoint(np.reshape(np.array([1, 2, 3, 4]),(4,1)))
#test_adjoint(deriv, 0.1)

landweber= Landweber(op, data, init, stepsize=0.01)
stoprule=(
    rules.CountIterations(100)+
    rules.Discrepancy(op.range.norm, data, noiselevel=0.1, tau=1))

reco, reco_data=landweber.run(stoprule)
op.params.scattering.plotX(grid, reco)
op.params.scattering.plotX(grid, exact_solution)




