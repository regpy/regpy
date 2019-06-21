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

xs = np.linspace(0, 1, 61)
x2 = np.linspace(0, 1, 12)

pts=[(0, 0), (0.1, 0), (0.2, 0), (0.3, 0), (0.4, 0), (0.5, 0), (0.6, 0), (0.7, 0), (0.8, 0), (0.9, 0),
     (1, 0), (1, 0.1), (1, 0.2), (1, 0.3), (1, 0.4), (1, 0.5), (1, 0.6), (1, 0.7), (1, 0.8), (1, 0.9),
     (1, 1), (0.9, 1), (0.8, 1), (0.7, 1), (0.6, 1), (0.5, 1), (0.4, 1), (0.3, 1), (0.2, 1), (0.1, 1),
     (0, 1), (0, 0.9), (0, 0.8), (0, 0.7), (0, 0.6), (0, 0.5), (0, 0.4), (0, 0.3), (0, 0.2), (0, 0.1)]



grid = UniformGrid(xs)
grid2= UniformGrid(x2)



from ngsolve import *
g=0.1*(x-0.5)
op = EIT(grid, g, pts, codomain=grid2)

exact_solution_coeff = 1
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
#adj=deriv.adjoint(np.linspace(1, 2, 40))

#init=np.concatenate((np.linspace(1, 2, 101), np.ones(100)))
init=cos(x)
init_gfu=GridFunction(op.fes)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_sol=init_solution.copy()
init_data=op(init_solution)

from itreg.spaces import L2
setting = HilbertSpaceSetting(op=op, domain=L2, codomain=L2)

landweber = Landweber(setting, data, init_solution, stepsize=0.1)
#irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
    rules.CountIterations(30) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)
