#TODO: Works properly in data space, but not as well in solution space

import setpath

from itreg.operators.NGSolveProblems.Coefficient import Coefficient
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

grid=UniformGrid(xs)

meshsize=10

from ngsolve import *
rhs=10*sin(x)*sin(y)
op = Coefficient(grid, meshsize, rhs, bc_left=0, bc_right=1, bc_bottom=sin(y), bc_top=sin(y), dim=2)

#exact_solution = np.linspace(1, 2, 201)
exact_solution_coeff = cos(x)
gfu_exact_solution=GridFunction(op.fes)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution=gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data=exact_data

gfu=GridFunction(op.fes)
for i in range(441):
    gfu.vec[i]=data[i]
    
Symfunc=CoefficientFunction(gfu)
func=np.zeros((21, 21))
for j in range(0, 21):
    for k in range(0, 21):
        mip=op.mesh(j/20, k/20)
        func[j][k]=Symfunc(mip)
        
plt.contourf(func)
plt.colorbar()      
plt.show()







_, deriv = op.linearize(exact_solution)
adj=deriv.adjoint(np.linspace(1, 2, 441))

#init=np.concatenate((np.linspace(1, 2, 101), np.ones(100)))
init=1
init_gfu=GridFunction(op.fes)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_data=op(init_solution)

from itreg.spaces import L2
setting = HilbertSpaceSetting(op=op, domain=L2, codomain=L2)

landweber = Landweber(setting, data, init_solution, stepsize=3)
#irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
    rules.CountIterations(3000) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)

plt.contourf(reco.reshape(21, 21))
plt.colorbar()
plt.show()

plt.contourf(exact_solution.reshape(21, 21))
plt.colorbar()
plt.show()


gfu=GridFunction(op.fes)
gfu2=GridFunction(op.fes)
#gfu3=GridFunction(op.params.fes)
for i in range(441):
    gfu.vec[i]=reco[i]
    gfu2.vec[i]=exact_solution[i]
#    gfu3.vec[i]=init_solution[i]
    
Symfunc=CoefficientFunction(gfu)
Symfunc2=CoefficientFunction(gfu2)
#Symfunc3=CoefficientFunction(gfu3)
func=np.zeros((21, 21))
func2=np.zeros((21, 21))
#func3=np.zeros(201)
for i in range(21):
    for j in range(21):
        mip=op.mesh(i/20, j/20)
        func[i, j]=Symfunc(mip)
        func2[i, j]=Symfunc2(mip)
#    func3[i]=Symfunc3(mip)
    
plt.contourf(func)
plt.colorbar()
plt.show()

plt.contourf(func2)
plt.colorbar()
plt.show()






gfu=GridFunction(op.fes)
gfu2=GridFunction(op.fes)
#gfu3=GridFunction(op.params.fes)
for i in range(441):
    gfu.vec[i]=reco_data[i]
    gfu2.vec[i]=exact_data[i]
#    gfu3.vec[i]=init_data[i]
    
Symfunc=CoefficientFunction(gfu)
Symfunc2=CoefficientFunction(gfu2)
#Symfunc3=CoefficientFunction(gfu3)
func=np.zeros((21, 21))
func2=np.zeros((21, 21))
#func3=np.zeros(201)
for i in range(0, 21):
    for j in range(0, 21):
        mip=op.mesh(i/20, j/20)
        func[i, j]=Symfunc(mip)
        func2[i, j]=Symfunc2(mip)
#    func3[i]=Symfunc3(mip)
    
plt.contourf(func)
plt.colorbar()
plt.show()

plt.contourf(func2)
plt.colorbar()
plt.show()







