#TODO: Works properly in data space, but not as well in solution space

import setpath

from itreg.operators.NGSolveProblems.Coefficient import Coefficient
from itreg.spaces import L2
from itreg.solvers import Landweber
from itreg.solvers import IRGNM_CG
from itreg.util import test_adjoint
import itreg.stoprules as rules
from itreg.grids import User_Defined
from itreg.spaces import H1_NGSolve
from ngsolve.meshes import Make1DMesh
from ngsolve import CoefficientFunction, GridFunction

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

xs = np.linspace(0, 1, 441)

grid=User_Defined(xs, xs.shape)

domain=L2(grid)
meshsize=10

from ngsolve import *
rhs=10*sin(x)*sin(y)
op = Coefficient(domain, meshsize, rhs=rhs, bc_left=0, bc_right=1, bc_bottom=sin(y), bc_top=sin(y), dim=2)

#exact_solution = np.linspace(1, 2, 201)
exact_solution_coeff = cos(x)
gfu_exact_solution=GridFunction(op.params.fes)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution=gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data=exact_data

gfu=GridFunction(op.params.fes)
for i in range(441):
    gfu.vec[i]=data[i]
    
Symfunc=CoefficientFunction(gfu)
func=np.zeros((21, 21))
for j in range(0, 21):
    for k in range(0, 21):
        mip=op.params.mesh(j/20, k/20)
        func[j][k]=Symfunc(mip)
        
plt.contourf(func)
plt.colorbar()      
plt.show()







_, deriv = op.linearize(exact_solution)
adj=deriv.adjoint(np.linspace(1, 2, 441))

#init=np.concatenate((np.linspace(1, 2, 101), np.ones(100)))
init=1
init_gfu=GridFunction(op.params.fes)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_data=op(init_solution)

landweber = Landweber(op, data, init_solution, stepsize=3)
#irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
    rules.CountIterations(3000) +
    rules.Discrepancy(op.range.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)

plt.contourf(reco.reshape(21, 21))
plt.colorbar()
plt.show()

plt.contourf(exact_solution.reshape(21, 21))
plt.colorbar()
plt.show()


gfu=GridFunction(op.params.fes)
gfu2=GridFunction(op.params.fes)
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
        mip=op.params.mesh(i/20, j/20)
        func[i, j]=Symfunc(mip)
        func2[i, j]=Symfunc2(mip)
#    func3[i]=Symfunc3(mip)
    
plt.contourf(func)
plt.colorbar()
plt.show()

plt.contourf(func2)
plt.colorbar()
plt.show()






gfu=GridFunction(op.params.fes)
gfu2=GridFunction(op.params.fes)
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
        mip=op.params.mesh(i/20, j/20)
        func[i, j]=Symfunc(mip)
        func2[i, j]=Symfunc2(mip)
#    func3[i]=Symfunc3(mip)
    
plt.contourf(func)
plt.colorbar()
plt.show()

plt.contourf(func2)
plt.colorbar()
plt.show()







