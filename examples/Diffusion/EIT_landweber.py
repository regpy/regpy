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

grid = UniformGrid(xs)
grid2= UniformGrid(x2)





from ngsolve import *
mesh=Mesh('..\..\itreg\meshes_ngsolve\meshes\circle.vol.gz')

g=0.1*(x-0.5)*(y-0.5)
op = EIT(grid, g, mesh, codomain=grid2)
pts=np.array(op.pts)
nr_points=pts.shape[0]

exact_solution_coeff = sin(y)
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
init=0.5*y
init_gfu=GridFunction(op.fes)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_sol=init_solution.copy()
init_data=op(init_solution)

from itreg.spaces import L2
setting = HilbertSpaceSetting(op=op, domain=L2, codomain=L2)

landweber = Landweber(setting, data, init_solution, stepsize=0.01)
#irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)

############################################################################################
#Plotting with the help of matplotlib
gfu=GridFunction(op.fes)
gfu2=GridFunction(op.fes)
gfu3=GridFunction(op.fes)
for i in range(61):
    gfu.vec[i]=reco[i]
    gfu2.vec[i]=exact_solution[i]
    gfu3.vec[i]=init_sol[i]
    
Symfunc=CoefficientFunction(gfu)
Symfunc2=CoefficientFunction(gfu2)
Symfunc3=CoefficientFunction(gfu3)
func=np.zeros(nr_points)
func2=np.zeros(nr_points)
func3=np.zeros(nr_points)
for i in range(nr_points):
    mip=mesh(pts[i, 0], pts[i, 1])
    func[i]=Symfunc(mip)
    func2[i]=Symfunc2(mip)
    func3[i]=Symfunc3(mip)
    
plt.tricontourf(pts[:,0], pts[:,1], func, label='reco')
plt.legend()
plt.colorbar()
plt.show()

plt.tricontourf(pts[:,0], pts[:,1], func2, label='exact')
plt.legend()
plt.colorbar()
plt.show()

plt.tricontourf(pts[:,0], pts[:,1], func3, label='init')
plt.legend()
plt.colorbar()
plt.show()



    
plt.plot(reco_data, label='reco')
plt.plot(exact_data, label='exact')
plt.plot(init_data, label='init')
plt.legend()
plt.show()