import setpath

from itreg.operators.NGSolveProblems.EIT import EIT
from itreg.spaces import UniformGrid
from itreg.solvers import Landweber, HilbertSpaceSetting

import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

xs = np.linspace(0, 1, 441)
x2 = np.linspace(0, 1, 8)

pts=[(0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 1), (0, 0.5)]

grid = UniformGrid(xs)
grid2= UniformGrid(x2)



from ngsolve import *
g=sin(x)
op = EIT(grid, g, pts)#, codomain=grid2)

exact_solution_coeff = cos(x)
gfu_exact_solution=GridFunction(op.fes)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution=gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data=exact_data

#gfu=GridFunction(op.fes)
#for i in range(441):
#    gfu.vec[i]=data[i]
    
#Symfunc=CoefficientFunction(gfu)
#func=np.zeros((21, 21))
#for j in range(0, 21):
#    for k in range(0, 21):
#        mip=op.mesh(j/20, k/20)
#        func[j][k]=Symfunc(mip)
        
#plt.contourf(func)
#plt.colorbar()      
#plt.show()

#############################################################################################
#_, deriv = op.linearize(exact_solution)
#adj=deriv.adjoint(np.linspace(1, 2, 8))

#init=np.concatenate((np.linspace(1, 2, 101), np.ones(100)))
init=cos(0.1*x)
init_gfu=GridFunction(op.fes)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_sol=init_solution.copy()
init_data=op(init_solution)

from itreg.spaces import L2
setting = HilbertSpaceSetting(op=op, domain=L2, codomain=L2)

landweber = Landweber(setting, data, init_solution, stepsize=10)
#irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
    rules.CountIterations(10) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)
